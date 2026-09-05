#!/usr/bin/env python3
"""Regression tests for the no-odometry v2 control boundary."""

from __future__ import annotations

import ast
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uav_no_odometry_optimizer as v2


def angle_signature(receiver: int, emitters: tuple[int, ...], target: float):
    pairs = {
        tuple(sorted((emitters[i], emitters[j]))): target
        for i in range(len(emitters))
        for j in range(i + 1, len(emitters))
    }
    return v2.TargetSignature(receiver=receiver, emitters=emitters, pair_targets=pairs)


def angle_batch(receiver: int, emitters: tuple[int, ...], value: float):
    pairs = {
        tuple(sorted((emitters[i], emitters[j]))): value
        for i in range(len(emitters))
        for j in range(i + 1, len(emitters))
    }
    return v2.AngleBatch(receiver=receiver, emitters=emitters, angles=pairs)


def reachable_type_names(root: object, limit: int = 1000) -> set[str]:
    names: set[str] = set()
    stack: list[object] = [root]
    seen: set[int] = set()
    while stack and len(seen) < limit:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        names.add(type(obj).__name__)
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.append(obj.__dict__)
    return names


class ImmutableBoundaryTests(unittest.TestCase):
    def test_angle_batch_copies_and_freezes_mapping(self) -> None:
        raw = {(0, 2): 0.71, (0, 3): 0.32, (2, 3): 0.39}
        batch = v2.AngleBatch(receiver=1, emitters=(0, 2, 3), angles=raw)
        raw[(0, 2)] = 99.0
        self.assertEqual(batch.angles[(0, 2)], 0.71)
        with self.assertRaises(TypeError):
            batch.angles[(0, 2)] = 99.0
        with self.assertRaises(Exception):
            batch.receiver = 2

    def test_target_signature_copies_and_freezes_mapping(self) -> None:
        raw = {(0, 2): 0.3, (0, 5): 0.4, (2, 5): 0.5}
        signature = v2.TargetSignature(receiver=1, emitters=(0, 2, 5), pair_targets=raw)
        raw[(0, 2)] = 9.0
        self.assertEqual(signature.pair_targets[(0, 2)], 0.3)
        with self.assertRaises(TypeError):
            signature.pair_targets[(0, 2)] = 9.0

    def test_pulse_inverse_keeps_level(self) -> None:
        pulse = v2.Pulse("F", 7)
        self.assertEqual(pulse.inverse(), v2.Pulse("B", 7))
        self.assertEqual(pulse.inverse().inverse(), pulse)


class ControlBoundaryTests(unittest.TestCase):
    def test_controller_source_has_no_environment_or_truth_leaks(self) -> None:
        source = inspect.getsource(v2.Controller)
        tree = ast.parse(source)
        controller_nodes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Controller"
        ]
        self.assertEqual(len(controller_nodes), 1)
        forbidden = (
            "World", "BearingSensor", "ProbeRunner", "ActuatorModel",
            "Evaluator", "callable", "Callable", "measure(", "position",
            "coordinate", "delta", "dx", "dy", "distance", "l_pos",
            "similarity", "true_x", "true_y", "np.",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_controller_object_graph_has_no_truth_types(self) -> None:
        signature = angle_signature(1, (0, 2), 0.5)
        controller = v2.Controller({1: signature}, level0=2)
        names = reachable_type_names(controller)
        forbidden = {
            "World", "BearingSensor", "ProbeRunner", "ActuatorModel",
            "Evaluator", "ndarray",
        }
        self.assertFalse(forbidden & names, names)

    def test_controller_is_driven_by_a_pre_recorded_tape(self) -> None:
        emitters = (0, 2)
        signature = angle_signature(1, emitters, 0.50)
        base = angle_batch(1, emitters, 0.70)
        candidates_values = {
            v2.Pulse("F", 2): 0.60,
            v2.Pulse("B", 2): 0.75,
            v2.Pulse("L", 2): 0.55,
            v2.Pulse("R", 2): 0.65,
        }
        candidates = {
            pulse: angle_batch(1, emitters, value)
            for pulse, value in candidates_values.items()
        }
        controller = v2.Controller({1: signature}, level0=2)
        proposed = [controller.propose(1, base) for _ in range(4)]
        self.assertEqual(proposed, list(candidates))
        selected = controller.select(1, base, candidates)
        self.assertEqual(selected, v2.Pulse("L", 2))

    def test_no_improvement_shrinks_level_after_full_cycle(self) -> None:
        emitters = (0, 2)
        signature = angle_signature(1, emitters, 0.50)
        base = angle_batch(1, emitters, 0.70)
        controller = v2.Controller({1: signature}, level0=4)
        pulses = [controller.propose(1, base) for _ in range(4)]
        self.assertEqual(
            pulses,
            [v2.Pulse("F", 4), v2.Pulse("B", 4),
             v2.Pulse("L", 4), v2.Pulse("R", 4)],
        )
        candidates = {pulse: base for pulse in pulses}
        self.assertIsNone(controller.select(1, base, candidates))
        self.assertEqual(controller.propose(1, base), v2.Pulse("F", 5))

    def test_invalid_angle_batch_is_not_an_improvement(self) -> None:
        emitters = (0, 2)
        signature = angle_signature(1, emitters, 0.50)
        base = angle_batch(1, emitters, 0.70)
        candidate = angle_batch(1, emitters, float("nan"))
        controller = v2.Controller({1: signature})
        selected = controller.select(1, base, {v2.Pulse("F", 0): candidate})
        self.assertIsNone(selected)


class ActuationTests(unittest.TestCase):
    def test_repeated_probes_leave_no_drift(self) -> None:
        world = v2.World(v2.initial_positions())
        actuator = v2.ActuatorModel.deterministic(5.0, 10)
        events: list[tuple[str, str]] = []

        def listener(receiver: int, pulse: v2.Pulse, role: str, **kwargs: object):
            events.append((role, pulse.primitive))

        runner = v2.ProbeRunner(world, actuator, move_listener=listener)
        base = world.position(3).copy()
        for i in range(400):
            level = i % 8
            runner.probe(3, v2.Pulse("L", level), (0, 2, 5), "test", 0, i)
            np.testing.assert_array_equal(world.position(3), base)
        self.assertEqual(runner.probe_count, 400)
        self.assertEqual(runner.rollback_failures, 0)
        for i in range(400):
            self.assertEqual(events[2 * i][0], "probe")
            self.assertEqual(events[2 * i + 1][0], "probe_inverse")

    def test_deeper_level_moves_less_in_hidden_model(self) -> None:
        actuator = v2.ActuatorModel.deterministic(5.0, 12)
        norms = [
            float(np.linalg.norm(actuator.delta(4, v2.Pulse("F", level))))
            for level in (0, 2, 5, 9)
        ]
        self.assertTrue(all(a > b for a, b in zip(norms, norms[1:])))

    def test_fixed_ids_do_not_move_during_probe_or_commit(self) -> None:
        world = v2.World(v2.initial_positions())
        actuator = v2.ActuatorModel.deterministic(5.0, 8)
        runner = v2.ProbeRunner(world, actuator)
        before = world.positions()
        runner.probe(5, v2.Pulse("R", 0), (0, 2, 8), "test", 0, 0)
        runner.commit(5, v2.Pulse("R", 0), "test", 1)
        after = world.positions()
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[2], before[2])
        np.testing.assert_array_equal(actuator.delta(0, v2.Pulse("F", 0)), np.zeros(2))


class GeometryAndEvaluationTests(unittest.TestCase):
    def test_four_thirty_degree_angles_do_not_imply_equilateral(self) -> None:
        origin = np.array([0.0, 0.0])
        second = np.array([1.0, 0.0])
        u = math.radians(100.0)
        radius = 2.0 * math.cos(math.radians(40.0))
        fifth = radius * np.array([math.cos(u), math.sin(u)])
        eighth = radius * np.array([
            math.cos(u - 2.0 * math.pi / 3.0),
            math.sin(u - 2.0 * math.pi / 3.0),
        ])
        angles = {
            "250": v2.pair_angle(fifth, second, origin),
            "850": v2.pair_angle(fifth, origin, eighth),
            "280": v2.pair_angle(eighth, second, origin),
            "580": v2.pair_angle(eighth, origin, fifth),
            "258": v2.pair_angle(fifth, second, eighth),
            "285": v2.pair_angle(eighth, second, fifth),
        }
        thirty = math.pi / 6.0
        for name in ("250", "850", "280", "580"):
            self.assertLess(abs(angles[name] - thirty), 1e-12)
        for name in ("258", "285"):
            self.assertLess(angles[name], 1e-12)

    def test_global_scan_is_readonly_and_has_108_residuals(self) -> None:
        world = v2.World(v2.initial_positions())
        compiler = v2.SignatureCompiler()
        before = world.positions()
        residuals, measurements = v2.global_angle_residuals(world, compiler)
        self.assertEqual(len(residuals), 108)
        self.assertEqual(measurements, 18)
        value = v2.global_l_ang(world, compiler)
        self.assertTrue(math.isfinite(value))
        np.testing.assert_array_equal(world.positions(), before)

    def test_l_pos_is_similarity_invariant_and_rotation_has_positive_det(self) -> None:
        evaluator = v2.Evaluator()
        p = v2.initial_positions()
        theta = 0.41
        rotation = 1.25 * np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ])
        shifted = p @ rotation.T + np.array([7.0, -3.0])
        fit = evaluator.similarity_fit(shifted)
        self.assertGreater(float(fit["determinant"]), 0.0)
        self.assertAlmostEqual(evaluator.l_pos(p), evaluator.l_pos(shifted), places=12)

    def test_probe_selection_is_covariant_under_similarity(self) -> None:
        theta = 0.37
        scale = 1.3
        rotation = scale * np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ])
        translation = np.array([5.0, -2.0])

        def transform_point(p: np.ndarray) -> np.ndarray:
            return p @ rotation.T + translation

        def transform_vector(v: np.ndarray) -> np.ndarray:
            return v @ rotation.T

        world = v2.World(v2.initial_positions())
        transformed_world = v2.World(transform_point(v2.initial_positions()))
        actuator = v2.ActuatorModel.deterministic(5.0, 8)

        class TransformedActuator:
            def delta(self, receiver: int, pulse: v2.Pulse) -> np.ndarray:
                return transform_vector(actuator.delta(receiver, pulse))

        compiler = v2.SignatureCompiler()
        signature = v2.stage_i_signatures(compiler)[5]

        def select_once(w, a):
            controller = v2.Controller({5: signature}, level0=3, level_min=8)
            recorder = v2.RunRecorder()
            runner = v2.ProbeRunner(w, a)
            base = w.sensor.batch(5, signature.emitters)
            candidates = {}
            for i in range(4):
                pulse = controller.propose(5, base)
                candidates[pulse] = runner.probe(
                    5, pulse, signature.emitters, "test", 0, i
                )
            return controller.select(5, base, candidates)

        selected = select_once(world, actuator)
        transformed_selected = select_once(
            transformed_world, TransformedActuator()
        )
        self.assertEqual(selected, transformed_selected)


if __name__ == "__main__":
    unittest.main()
