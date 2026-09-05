#!/usr/bin/env python3
"""No-odometry, angle-only formation correction.

The Controller sees only tagged unsigned angles, target angle scalars, and
dimensionless pulse tokens.  Positions, hidden actuator gains, global angle
diagnostics, and Procrustes diagnostics remain outside the control boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np


N_UAV = 10
PERIPHERAL = tuple(range(1, 10))
EMITTER_GROUPS = ((2, 5, 8), (3, 6, 9), (1, 4, 7))
ANCHOR_IDS = (0, 2, 5, 8)
STAGE_I_RECEIVERS = (5, 8)
STAGE_II_RECEIVERS = (1, 3, 4, 6, 7, 9)
PRIMITIVE_ORDER = ("F", "B", "L", "R")
PRIMITIVE_INVERSE = {"F": "B", "B": "F", "L": "R", "R": "L"}
PRIMITIVE_VECTORS = {
    "F": (0.0, 1.0),
    "B": (0.0, -1.0),
    "L": (-1.0, 0.0),
    "R": (1.0, 0.0),
}

INITIAL_POLAR = {
    0: (0.0, 0.0),
    1: (100.0, 0.00),
    2: (98.0, 40.10),
    3: (112.0, 80.21),
    4: (105.0, 119.75),
    5: (98.0, 159.86),
    6: (112.0, 199.96),
    7: (105.0, 240.07),
    8: (98.0, 280.17),
    9: (112.0, 320.28),
}

STAGE_I_KEY_NAMES = {
    (5, (0, 2)): "key_250_rad",
    (5, (0, 8)): "key_850_rad",
    (5, (2, 8)): "key_258_rad",
    (8, (0, 2)): "key_280_rad",
    (8, (0, 5)): "key_580_rad",
    (8, (2, 5)): "key_285_rad",
}


def initial_positions() -> np.ndarray:
    p = np.zeros((N_UAV, 2), dtype=float)
    for k, (radius, angle_deg) in INITIAL_POLAR.items():
        angle = math.radians(angle_deg)
        p[k] = (radius * math.cos(angle), radius * math.sin(angle))
    return p


def ideal_template() -> np.ndarray:
    q = np.zeros((N_UAV, 2), dtype=float)
    for k in PERIPHERAL:
        angle = 2.0 * math.pi * (k - 1) / 9.0
        q[k] = (math.cos(angle), math.sin(angle))
    return q


def pair_angle(receiver: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    ua = first - receiver
    ub = second - receiver
    na = math.hypot(float(ua[0]), float(ua[1]))
    nb = math.hypot(float(ub[0]), float(ub[1]))
    if na == 0.0 or nb == 0.0 or not (math.isfinite(na) and math.isfinite(nb)):
        return float("nan")
    dot = float(ua[0] * ub[0] + ua[1] * ub[1])
    cross = float(ua[0] * ub[1] - ua[1] * ub[0])
    return math.atan2(abs(cross), dot)


@dataclass(frozen=True)
class AngleBatch:
    receiver: int
    emitters: tuple[int, ...]
    angles: Mapping[tuple[int, int], float]

    def __post_init__(self) -> None:
        emitters = tuple(sorted({int(e) for e in self.emitters}))
        if self.receiver in emitters:
            raise ValueError("receiver may not be an emitter")
        copied = {tuple(key): float(value) for key, value in dict(self.angles).items()}
        expected = {
            tuple(sorted((emitters[i], emitters[j])))
            for i in range(len(emitters))
            for j in range(i + 1, len(emitters))
        }
        if set(copied) != expected:
            raise ValueError("AngleBatch pair keys do not match its emitter set")
        object.__setattr__(self, "receiver", int(self.receiver))
        object.__setattr__(self, "emitters", emitters)
        object.__setattr__(self, "angles", MappingProxyType(copied))


@dataclass(frozen=True)
class TargetSignature:
    receiver: int
    emitters: tuple[int, ...]
    pair_targets: Mapping[tuple[int, int], float]

    def __post_init__(self) -> None:
        emitters = tuple(sorted({int(e) for e in self.emitters}))
        if self.receiver in emitters:
            raise ValueError("receiver may not be an emitter")
        copied = {tuple(key): float(value) for key, value in dict(self.pair_targets).items()}
        expected = {
            tuple(sorted((emitters[i], emitters[j])))
            for i in range(len(emitters))
            for j in range(i + 1, len(emitters))
        }
        if set(copied) != expected:
            raise ValueError("TargetSignature pair keys do not match its emitter set")
        if not all(math.isfinite(value) for value in copied.values()):
            raise ValueError("TargetSignature contains a non-finite angle")
        object.__setattr__(self, "receiver", int(self.receiver))
        object.__setattr__(self, "emitters", emitters)
        object.__setattr__(self, "pair_targets", MappingProxyType(copied))


@dataclass(frozen=True)
class Pulse:
    primitive: str
    level: int

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVE_INVERSE:
            raise ValueError("unknown pulse primitive")
        if int(self.level) < 0:
            raise ValueError("pulse level must be nonnegative")
        object.__setattr__(self, "primitive", str(self.primitive))
        object.__setattr__(self, "level", int(self.level))

    def inverse(self) -> "Pulse":
        return Pulse(PRIMITIVE_INVERSE[self.primitive], self.level)


class World:
    def __init__(self, positions: np.ndarray):
        self._pos = np.asarray(positions, dtype=float).reshape(N_UAV, 2)
        self.sensor = BearingSensor(self)

    def position(self, k: int) -> np.ndarray:
        return self._pos[int(k)].copy()

    def positions(self) -> np.ndarray:
        return self._pos.copy()

    def set_position(self, k: int, value: np.ndarray) -> None:
        self._pos[int(k)] = np.asarray(value, dtype=float).reshape(2)

    def commit_deltas(self, deltas: Mapping[int, np.ndarray]) -> None:
        updated = self._pos.copy()
        for k, delta in deltas.items():
            updated[int(k)] += np.asarray(delta, dtype=float).reshape(2)
        self._pos = updated


class BearingSensor:
    def __init__(self, world: World):
        self._world = world

    def batch(self, receiver: int, emitters: Sequence[int]) -> AngleBatch:
        em = tuple(sorted({int(e) for e in emitters}))
        k = int(receiver)
        if k in em:
            raise ValueError("receiver may not be an emitter")
        pk = self._world.position(k)
        angles = {}
        for i in range(len(em)):
            for j in range(i + 1, len(em)):
                key = (em[i], em[j])
                angles[key] = pair_angle(
                    pk, self._world.position(key[0]), self._world.position(key[1])
                )
        return AngleBatch(receiver=k, emitters=em, angles=angles)


@dataclass(frozen=True)
class ActuatorModel:
    orientations: Mapping[int, float]
    base_gain_m: float
    level_depth: int

    @staticmethod
    def deterministic(base_gain_m: float, level_depth: int) -> "ActuatorModel":
        orientations = {k: 0.73 + 0.41 * k for k in range(1, N_UAV)}
        return ActuatorModel(
            orientations=MappingProxyType(dict(orientations)),
            base_gain_m=float(base_gain_m),
            level_depth=int(level_depth),
        )

    def delta(self, receiver: int, pulse: Pulse) -> np.ndarray:
        if pulse.level > self.level_depth:
            raise ValueError("pulse level exceeds actuator depth")
        if int(receiver) == 0:
            return np.zeros(2, dtype=float)
        angle = self.orientations[int(receiver)]
        cosine, sine = math.cos(angle), math.sin(angle)
        local_x, local_y = PRIMITIVE_VECTORS[pulse.primitive]
        magnitude = self.base_gain_m / (2.0 ** pulse.level)
        return magnitude * np.array(
            [cosine * local_x - sine * local_y,
             sine * local_x + cosine * local_y],
            dtype=float,
        )


class RollbackError(RuntimeError):
    pass


class ProbeRunner:
    def __init__(
        self,
        world: World,
        actuator: ActuatorModel,
        move_listener=None,
        restore_tolerance: float = 1e-9,
    ):
        self.world = world
        self.actuator = actuator
        self.move_listener = move_listener
        self.restore_tolerance = float(restore_tolerance)
        self.probe_count = 0
        self.rollback_failures = 0

    def _record(
        self,
        receiver: int,
        pulse: Pulse,
        role: str,
        phase: str,
        state_step: int,
        candidate_id: int,
    ) -> None:
        delta = self.actuator.delta(receiver, pulse)
        if self.move_listener is not None:
            self.move_listener(
                receiver=int(receiver),
                pulse=pulse,
                role=role,
                phase=phase,
                state_step=int(state_step),
                candidate_id=int(candidate_id),
                dx=float(delta[0]),
                dy=float(delta[1]),
            )

    def probe(
        self,
        receiver: int,
        pulse: Pulse,
        emitters: Sequence[int],
        phase: str,
        state_step: int,
        candidate_id: int,
    ) -> AngleBatch:
        base = self.world.position(receiver)
        delta = self.actuator.delta(receiver, pulse)
        self.probe_count += 1
        try:
            self.world.set_position(receiver, base + delta)
            self._record(
                receiver, pulse, "probe", phase, state_step, candidate_id
            )
            measured = self.world.sensor.batch(receiver, emitters)
        finally:
            self.world.set_position(receiver, self.world.position(receiver) - delta)
            self._record(
                receiver,
                pulse.inverse(),
                "probe_inverse",
                phase,
                state_step,
                candidate_id,
            )
            restored = self.world.position(receiver)
            drift = float(np.linalg.norm(restored - base))
            if drift > self.restore_tolerance:
                self.rollback_failures += 1
                raise RollbackError(
                    f"FY{receiver:02d} probe rollback drift {drift:.3e} m"
                )
            self.world.set_position(receiver, base)
        return measured

    def commit(
        self,
        receiver: int,
        pulse: Pulse,
        phase: str,
        state_step: int,
    ) -> None:
        delta = self.actuator.delta(receiver, pulse)
        self.world.set_position(receiver, self.world.position(receiver) + delta)
        self._record(receiver, pulse, "commit", phase, state_step, -1)

    def commit_many(
        self,
        pulses: Mapping[int, Pulse],
        phase: str,
        state_step: int,
    ) -> None:
        deltas = {
            int(receiver): self.actuator.delta(receiver, pulse)
            for receiver, pulse in pulses.items()
        }
        self.world.commit_deltas(deltas)
        for receiver, pulse in pulses.items():
            self._record(receiver, pulse, "commit", phase, state_step, -1)


@dataclass
class _ControlState:
    level: int
    attempt: int = 0


class Controller:
    """Pattern-search controller over pulse tokens and tagged angles only."""

    def __init__(
        self,
        signatures: Mapping[int, TargetSignature],
        level0: int = 0,
        level_min: int = 24,
        shrink: int = 1,
        improvement_tol: float = 1e-14,
    ):
        self.signatures = {
            int(k): sig for k, sig in signatures.items()
        }
        self.level0 = int(level0)
        self.level_min = int(level_min)
        self.shrink = int(shrink)
        self.improvement_tol = float(improvement_tol)
        self._states = {
            int(k): _ControlState(level=int(level0)) for k in self.signatures
        }

    @staticmethod
    def _residuals(batch: AngleBatch, signature: TargetSignature) -> tuple[float, ...]:
        if batch.receiver != signature.receiver:
            raise ValueError("batch receiver does not match signature")
        if batch.emitters != signature.emitters:
            raise ValueError("batch emitters do not match signature")
        values = []
        for key, target in signature.pair_targets.items():
            observed = batch.angles.get(key, float("nan"))
            values.append(observed - target)
        return tuple(values)

    @classmethod
    def mse(cls, batch: AngleBatch, signature: TargetSignature) -> float:
        residuals = cls._residuals(batch, signature)
        if not residuals or not all(math.isfinite(value) for value in residuals):
            return float("inf")
        return math.fsum(value * value for value in residuals) / len(residuals)

    @classmethod
    def rms(cls, batch: AngleBatch, signature: TargetSignature) -> float:
        value = cls.mse(batch, signature)
        return math.sqrt(value) if math.isfinite(value) else float("inf")

    def _state(self, receiver: int) -> _ControlState:
        receiver = int(receiver)
        if receiver not in self._states:
            raise KeyError(f"no control session for FY{receiver:02d}")
        return self._states[receiver]

    def at_minimum(self, receiver: int) -> bool:
        return self._state(receiver).level >= self.level_min

    def propose(self, receiver: int, current: AngleBatch) -> Pulse:
        receiver = int(receiver)
        state = self._state(receiver)
        signature = self.signatures[receiver]
        self.mse(current, signature)
        primitive = PRIMITIVE_ORDER[state.attempt % len(PRIMITIVE_ORDER)]
        state.attempt += 1
        return Pulse(primitive, state.level)

    def select(
        self,
        receiver: int,
        base: AngleBatch,
        candidates: Mapping[Pulse, AngleBatch],
    ) -> Pulse | None:
        receiver = int(receiver)
        state = self._state(receiver)
        signature = self.signatures[receiver]
        base_loss = self.mse(base, signature)
        best_pulse: Pulse | None = None
        best_loss = base_loss
        for pulse, batch in candidates.items():
            candidate_loss = self.mse(batch, signature)
            if candidate_loss < best_loss - self.improvement_tol:
                best_loss = candidate_loss
                best_pulse = pulse
        state.attempt = 0
        if best_pulse is None:
            state.level = min(state.level + self.shrink, self.level_min)
            return None
        return best_pulse


class SignatureCompiler:
    def __init__(self, template: np.ndarray | None = None):
        self.template = ideal_template() if template is None else np.asarray(template, float)

    def __call__(self, receiver: int, emitters: Sequence[int]) -> TargetSignature:
        em = tuple(sorted({int(e) for e in emitters}))
        k = int(receiver)
        if k in em:
            raise ValueError("receiver may not be an emitter")
        kr = self.template[k]
        targets = {}
        for i in range(len(em)):
            for j in range(i + 1, len(em)):
                key = (em[i], em[j])
                targets[key] = pair_angle(kr, self.template[key[0]], self.template[key[1]])
        return TargetSignature(receiver=k, emitters=em, pair_targets=targets)


def stage_i_signatures(compiler: SignatureCompiler) -> dict[int, TargetSignature]:
    return {
        5: compiler(5, (0, 2, 8)),
        8: compiler(8, (0, 2, 5)),
    }


def stage_ii_signatures(compiler: SignatureCompiler) -> dict[int, TargetSignature]:
    return {k: compiler(k, ANCHOR_IDS) for k in STAGE_II_RECEIVERS}


class Evaluator:
    def __init__(self, template: np.ndarray | None = None):
        self.template = ideal_template() if template is None else np.asarray(template, float)

    def similarity_fit(self, positions: np.ndarray) -> dict[str, object]:
        p = np.asarray(positions, dtype=float).reshape(N_UAV, 2)
        q = self.template
        p_center = p.mean(axis=0)
        q_center = q.mean(axis=0)
        pc = p - p_center
        qc = q - q_center
        m = pc.T @ qc
        u, singular, vt = np.linalg.svd(m)
        sign = float(np.sign(np.linalg.det(u @ vt)))
        if sign == 0.0:
            sign = 1.0
        rotation = u @ np.diag([1.0, sign]) @ vt
        q_norm_sq = float((qc ** 2).sum())
        scale = float((singular[0] + sign * singular[1]) / q_norm_sq)
        center = p_center - scale * (rotation @ q_center)
        fitted = center + scale * (q @ rotation.T)
        residual = float(((p - fitted) ** 2).sum())
        return {
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "fitted": fitted,
            "residual": residual,
            "determinant": float(np.linalg.det(rotation)),
        }

    def l_pos(self, positions: np.ndarray, eps: float = 1e-12) -> float:
        fit = self.similarity_fit(positions)
        scale = float(fit["scale"])
        return float(fit["residual"]) / (10.0 * scale * scale + eps)

    def diagnostics(self, positions: np.ndarray) -> dict[str, float | int | bool]:
        p = np.asarray(positions, dtype=float).reshape(N_UAV, 2)
        fit = self.similarity_fit(p)
        center = np.asarray(fit["center"], dtype=float)
        fitted = np.asarray(fit["fitted"], dtype=float)
        errors = np.linalg.norm(p - fitted, axis=1)
        phases = {
            k: math.atan2(float(p[k, 1] - center[1]), float(p[k, 0] - center[0]))
            for k in PERIPHERAL
        }
        gap_errors = []
        for i, k in enumerate(PERIPHERAL):
            nxt = PERIPHERAL[(i + 1) % 9]
            gap = (phases[nxt] - phases[k]) % (2.0 * math.pi)
            gap_errors.append(gap - 2.0 * math.pi / 9.0)
        relative = {
            k: (phases[k] - phases[1]) % (2.0 * math.pi) for k in PERIPHERAL
        }
        order = sorted(PERIPHERAL, key=lambda k: (relative[k], k))
        radii = np.linalg.norm(p[list(PERIPHERAL)] - p[0], axis=1)
        center_offset = float(np.linalg.norm(p[0] - center))
        return {
            "rho_fit": float(fit["scale"]),
            "center_offset": center_offset,
            "err_max": float(errors.max()),
            "err_mean": float(errors.mean()),
            "gap_err_max_deg": float(max(abs(v) for v in gap_errors) * 180.0 / math.pi),
            "radius_mean": float(radii.mean()),
            "radius_rms": float(math.sqrt(float(np.mean(radii ** 2)))),
            "label_order_ok": int(order == list(PERIPHERAL)),
            "det_R": float(fit["determinant"]),
            "l_pos": self.l_pos(p),
        }


def global_angle_residuals(
    world: World, signature_for: callable
) -> tuple[list[float], int]:
    residuals: list[float] = []
    measurements = 0
    for group in EMITTER_GROUPS:
        emitters = (0,) + group
        for receiver in PERIPHERAL:
            if receiver in group:
                continue
            signature = signature_for(receiver, emitters)
            batch = world.sensor.batch(receiver, emitters)
            measurements += 1
            for key, target in signature.pair_targets.items():
                residuals.append(batch.angles[key] - target)
    if len(residuals) != 108:
        raise AssertionError(f"global angle scan produced {len(residuals)} residuals")
    return residuals, measurements


def global_l_ang(world: World, signature_for: callable) -> float:
    residuals, _ = global_angle_residuals(world, signature_for)
    return float(math.sqrt(math.fsum(value * value for value in residuals) / 108.0))


@dataclass
class RunRecorder:
    states: list[dict[str, object]] = field(default_factory=list)
    adjustments: list[dict[str, object]] = field(default_factory=list)
    moves: list[dict[str, object]] = field(default_factory=list)
    summaries: list[dict[str, object]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    _move_id: int = 0

    def log(self, message: str) -> None:
        self.logs.append(message)

    def add_move(self, **kwargs: object) -> None:
        self._move_id += 1
        row = {"move_id": self._move_id, **kwargs}
        self.moves.append(row)

    def add_state(self, row: Mapping[str, object]) -> None:
        self.states.append(dict(row))

    def add_summary(self, row: Mapping[str, object]) -> None:
        self.summaries.append(dict(row))

    def add_adjustment(self, row: Mapping[str, object]) -> None:
        self.adjustments.append(dict(row))


def state_row(
    world: World,
    evaluator: Evaluator,
    signature_for: callable,
    phase: str,
    state_step: int,
    event: str,
    extras: Mapping[str, float | int] | None = None,
) -> dict[str, object]:
    residuals, _ = global_angle_residuals(world, signature_for)
    l_ang = float(math.sqrt(math.fsum(value * value for value in residuals) / 108.0))
    positions = world.positions()
    diagnostics = evaluator.diagnostics(positions)
    row: dict[str, object] = {
        "state_step": int(state_step),
        "phase": phase,
        "event": event,
        "l_ang_rad": l_ang,
        **diagnostics,
    }
    for k in range(N_UAV):
        row[f"true_x_FY{k:02d}"] = float(positions[k, 0])
        row[f"true_y_FY{k:02d}"] = float(positions[k, 1])
    if extras:
        row.update(extras)
    return row


def stage_i_key_extras(
    batches: Mapping[int, AngleBatch],
    signatures: Mapping[int, TargetSignature],
) -> dict[str, float]:
    extras: dict[str, float] = {}
    for receiver, signature in signatures.items():
        for key, target in signature.pair_targets.items():
            name = STAGE_I_KEY_NAMES[(receiver, key)]
            observed = batches[receiver].angles[key]
            extras[name] = float(observed - target)
    return extras


def stage_ii_local_extras(
    batches: Mapping[int, AngleBatch],
    signatures: Mapping[int, TargetSignature],
    controller: Controller,
) -> dict[str, float]:
    return {
        f"local_rms_FY{k:02d}_rad": controller.rms(batches[k], signatures[k])
        for k in signatures
    }


def append_state(
    world: World,
    evaluator: Evaluator,
    compiler: SignatureCompiler,
    recorder: RunRecorder,
    phase: str,
    state_step: int,
    event: str,
    extras: Mapping[str, float | int] | None = None,
) -> dict[str, object]:
    row = state_row(world, evaluator, compiler, phase, state_step, event, extras)
    recorder.add_state(row)
    recorder.log(
        f"state={int(state_step):04d} phase={phase} event={event} "
        f"l_ang={float(row['l_ang_rad']):.9e} l_pos={float(row['l_pos']):.9e}"
    )
    return row


def probe_cycle(
    world: World,
    receiver: int,
    signature: TargetSignature,
    controller: Controller,
    runner: ProbeRunner,
    recorder: RunRecorder,
    phase: str,
    state_step: int,
) -> tuple[Pulse | None, AngleBatch, AngleBatch]:
    base = world.sensor.batch(receiver, signature.emitters)
    candidates: dict[Pulse, AngleBatch] = {}
    for candidate_id in range(4):
        pulse = controller.propose(receiver, base)
        candidates[pulse] = runner.probe(
            receiver,
            pulse,
            signature.emitters,
            phase=phase,
            state_step=state_step,
            candidate_id=candidate_id,
        )
    selected = controller.select(receiver, base, candidates)
    if selected is None:
        return None, base, base
    runner.commit(receiver, selected, phase=phase, state_step=state_step)
    after = world.sensor.batch(receiver, signature.emitters)
    return selected, base, after


def adjustment_row(
    phase: str,
    state_step: int,
    receiver: int,
    pulse: Pulse | None,
    before: AngleBatch,
    after: AngleBatch,
    signature: TargetSignature,
    controller: Controller,
) -> dict[str, object]:
    return {
        "state_step": int(state_step),
        "phase": phase,
        "receiver": int(receiver),
        "pulse_primitive": "" if pulse is None else pulse.primitive,
        "pulse_level": -1 if pulse is None else pulse.level,
        "local_rms_before_rad": controller.rms(before, signature),
        "local_rms_after_rad": controller.rms(after, signature),
    }


def run_stage_i(
    world: World,
    evaluator: Evaluator,
    compiler: SignatureCompiler,
    controller: Controller,
    runner: ProbeRunner,
    recorder: RunRecorder,
    tau: float,
    max_cycles: int,
    state_step: int,
) -> tuple[bool, int, dict[str, object]]:
    signatures = stage_i_signatures(compiler)
    phase = "stage_i"
    recorder.log("stage I started")
    for cycle in range(1, max_cycles + 1):
        batches = {
            receiver: world.sensor.batch(receiver, signature.emitters)
            for receiver, signature in signatures.items()
        }
        extras = stage_i_key_extras(batches, signatures)
        if max(abs(value) for value in extras.values()) <= tau:
            final = append_state(
                world,
                evaluator,
                compiler,
                recorder,
                phase,
                state_step,
                "stage_i_complete",
                extras,
            )
            recorder.add_summary(
                {
                    "milestone": "stage_i_complete",
                    "state_step": state_step,
                    "cycle": cycle - 1,
                    "l_ang_rad": final["l_ang_rad"],
                    "l_pos": final["l_pos"],
                    "success": 1,
                }
            )
            recorder.log("stage I passed all six key-angle criteria")
            return True, state_step, final

        any_selected = False
        last_extras = extras
        for receiver in STAGE_I_RECEIVERS:
            selected, before, after = probe_cycle(
                world,
                receiver,
                signatures[receiver],
                controller,
                runner,
                recorder,
                phase,
                state_step,
            )
            recorder.add_adjustment(
                adjustment_row(
                    phase,
                    state_step,
                    receiver,
                    selected,
                    before,
                    after,
                    signatures[receiver],
                    controller,
                )
            )
            if selected is not None:
                state_step += 1
                any_selected = True
                refreshed = {
                    r: world.sensor.batch(r, signatures[r].emitters)
                    for r in STAGE_I_RECEIVERS
                }
                last_extras = stage_i_key_extras(refreshed, signatures)
                row = append_state(
                    world,
                    evaluator,
                    compiler,
                    recorder,
                    phase,
                    state_step,
                    f"accepted_FY{receiver:02d}",
                    last_extras,
                )
                recorder.log(
                    f"stage I FY{receiver:02d} accepted {selected.primitive}"
                    f"@{selected.level}; l_ang={float(row['l_ang_rad']):.9e}"
                )
        all_blocked = all(
            controller.at_minimum(receiver) for receiver in STAGE_I_RECEIVERS
        )
        if not any_selected and all_blocked:
            recorder.log(
                f"stage I stopped: no accepted pulse at minimum command level "
                f"(cycle {cycle})"
            )
            break

    batches = {
        receiver: world.sensor.batch(receiver, signature.emitters)
        for receiver, signature in signatures.items()
    }
    extras = stage_i_key_extras(batches, signatures)
    final = append_state(
        world,
        evaluator,
        compiler,
        recorder,
        phase,
        state_step,
        "stage_i_failed",
        extras,
    )
    recorder.add_summary(
        {
            "milestone": "stage_i_failed",
            "state_step": state_step,
            "l_ang_rad": final["l_ang_rad"],
            "l_pos": final["l_pos"],
            "success": 0,
        }
    )
    return False, state_step, final


def run_stage_ii(
    world: World,
    evaluator: Evaluator,
    compiler: SignatureCompiler,
    controller: Controller,
    runner: ProbeRunner,
    recorder: RunRecorder,
    tau: float,
    max_batches: int,
    state_step: int,
) -> tuple[bool, int, dict[str, object]]:
    signatures = stage_ii_signatures(compiler)
    phase = "stage_ii"
    recorder.log("stage II started")
    for batch_number in range(1, max_batches + 1):
        base_batches = {
            receiver: world.sensor.batch(receiver, signature.emitters)
            for receiver, signature in signatures.items()
        }
        local_rms = {
            receiver: controller.rms(base_batches[receiver], signatures[receiver])
            for receiver in signatures
        }
        if max(local_rms.values()) <= tau:
            extras = stage_ii_local_extras(base_batches, signatures, controller)
            final = append_state(
                world,
                evaluator,
                compiler,
                recorder,
                phase,
                state_step,
                "stage_ii_complete",
                extras,
            )
            recorder.add_summary(
                {
                    "milestone": "stage_ii_complete",
                    "state_step": state_step,
                    "batch": batch_number - 1,
                    "l_ang_rad": final["l_ang_rad"],
                    "l_pos": final["l_pos"],
                    "success": 1,
                }
            )
            recorder.log("stage II passed all local four-anchor signatures")
            return True, state_step, final

        plans: dict[int, Pulse] = {}
        pending_rows: dict[int, dict[str, object]] = {}
        for receiver in STAGE_II_RECEIVERS:
            if local_rms[receiver] <= tau:
                continue
            signature = signatures[receiver]
            candidates: dict[Pulse, AngleBatch] = {}
            for candidate_id in range(4):
                pulse = controller.propose(receiver, base_batches[receiver])
                candidates[pulse] = runner.probe(
                    receiver,
                    pulse,
                    signature.emitters,
                    phase=phase,
                    state_step=state_step,
                    candidate_id=candidate_id,
                )
            selected = controller.select(
                receiver, base_batches[receiver], candidates
            )
            if selected is not None:
                plans[receiver] = selected
                pending_rows[receiver] = adjustment_row(
                    phase,
                    state_step,
                    receiver,
                    selected,
                    base_batches[receiver],
                    base_batches[receiver],
                    signature,
                    controller,
                )
            else:
                recorder.add_adjustment(
                    adjustment_row(
                        phase,
                        state_step,
                        receiver,
                        None,
                        base_batches[receiver],
                        base_batches[receiver],
                        signature,
                        controller,
                    )
                )

        pending_receivers = [
            receiver for receiver in STAGE_II_RECEIVERS if local_rms[receiver] > tau
        ]
        all_blocked = all(controller.at_minimum(receiver) for receiver in pending_receivers)
        if not plans and all_blocked:
            recorder.log(
                f"stage II stopped: no accepted pulse at minimum command level "
                f"(batch {batch_number})"
            )
            break

        runner.commit_many(plans, phase=phase, state_step=state_step)
        state_step += 1
        after_batches = {
            receiver: world.sensor.batch(receiver, signature.emitters)
            for receiver, signature in signatures.items()
        }
        extras = stage_ii_local_extras(after_batches, signatures, controller)
        row = append_state(
            world,
            evaluator,
            compiler,
            recorder,
            phase,
            state_step,
            "parallel_batch_accepted",
            extras,
        )
        for receiver, pulse in plans.items():
            recorder.add_adjustment(
                {
                    "state_step": state_step,
                    "phase": phase,
                    "receiver": receiver,
                    "pulse_primitive": pulse.primitive,
                    "pulse_level": pulse.level,
                    "local_rms_before_rad": local_rms[receiver],
                    "local_rms_after_rad": controller.rms(
                        after_batches[receiver], signatures[receiver]
                    ),
                }
            )
            recorder.log(
                f"stage II FY{receiver:02d} accepted {pulse.primitive}"
                f"@{pulse.level}; l_ang={float(row['l_ang_rad']):.9e}"
            )

    base_batches = {
        receiver: world.sensor.batch(receiver, signature.emitters)
        for receiver, signature in signatures.items()
    }
    extras = stage_ii_local_extras(base_batches, signatures, controller)
    final = append_state(
        world,
        evaluator,
        compiler,
        recorder,
        phase,
        state_step,
        "stage_ii_failed",
        extras,
    )
    recorder.add_summary(
        {
            "milestone": "stage_ii_failed",
            "state_step": state_step,
            "l_ang_rad": final["l_ang_rad"],
            "l_pos": final["l_pos"],
            "success": 0,
        }
    )
    return False, state_step, final


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    recorder: RunRecorder,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "state_history.csv", recorder.states)
    write_csv(output_dir / "adjustments.csv", recorder.adjustments)
    write_csv(output_dir / "moves.csv", recorder.moves)
    write_csv(output_dir / "phase_summary.csv", recorder.summaries)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "run.log").write_text("\n".join(recorder.logs) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_root / "outputs")
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--stage-i-max-cycles", type=int, default=500)
    parser.add_argument("--stage-ii-max-batches", type=int, default=500)
    parser.add_argument("--stage-i-level0", type=int, default=3)
    parser.add_argument("--stage-ii-level0", type=int, default=0)
    parser.add_argument("--level-min", type=int, default=24)
    parser.add_argument("--base-gain", type=float, default=5.0)
    parser.add_argument("--actuator-seed", type=int, default=20260905)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    world = World(initial_positions())
    evaluator = Evaluator()
    compiler = SignatureCompiler()
    recorder = RunRecorder()
    actuator = ActuatorModel.deterministic(args.base_gain, args.level_min)

    def move_listener(**kwargs: object) -> None:
        recorder.add_move(**kwargs)

    runner = ProbeRunner(world, actuator, move_listener=move_listener)
    append_state(
        world,
        evaluator,
        compiler,
        recorder,
        "initial",
        0,
        "initial",
    )
    recorder.add_summary(
        {
            "milestone": "initial",
            "state_step": 0,
            "l_ang_rad": recorder.states[-1]["l_ang_rad"],
            "l_pos": recorder.states[-1]["l_pos"],
            "success": 1,
        }
    )

    stage_i_controller = Controller(
        stage_i_signatures(compiler),
        level0=args.stage_i_level0,
        level_min=args.level_min,
    )
    stage_i_ok, state_step, stage_i_final = run_stage_i(
        world,
        evaluator,
        compiler,
        stage_i_controller,
        runner,
        recorder,
        args.tau,
        args.stage_i_max_cycles,
        state_step=1,
    )

    success = False
    final_row = stage_i_final
    if stage_i_ok:
        stage_ii_controller = Controller(
            stage_ii_signatures(compiler),
            level0=args.stage_ii_level0,
            level_min=args.level_min,
        )
        stage_ii_ok, state_step, stage_ii_final = run_stage_ii(
            world,
            evaluator,
            compiler,
            stage_ii_controller,
            runner,
            recorder,
            args.tau,
            args.stage_ii_max_batches,
            state_step=state_step + 1,
        )
        final_row = stage_ii_final
        final_residuals, final_measurements = global_angle_residuals(world, compiler)
        final_l_ang = float(
            math.sqrt(math.fsum(v * v for v in final_residuals) / 108.0)
        )
        success = bool(stage_ii_ok and final_l_ang < args.tau)
        if success:
            append_state(
                world,
                evaluator,
                compiler,
                recorder,
                "final",
                state_step,
                "final_success",
                stage_ii_local_extras(
                    {
                        k: world.sensor.batch(k, ANCHOR_IDS)
                        for k in STAGE_II_RECEIVERS
                    },
                    stage_ii_signatures(compiler),
                    stage_ii_controller,
                ),
            )

    positions = world.positions()
    final_diagnostics = evaluator.diagnostics(positions)
    summary = {
        "success": success,
        "stop_reason": (
            "all angle criteria passed"
            if success
            else "stage criterion not reached within protection limits"
        ),
        "tau_rad": float(args.tau),
        "final_l_ang_rad": float(final_row["l_ang_rad"]),
        "final_l_pos": float(final_diagnostics["l_pos"]),
        "final_center_offset_m": float(final_diagnostics["center_offset"]),
        "final_err_max_m": float(final_diagnostics["err_max"]),
        "final_gap_err_max_deg": float(final_diagnostics["gap_err_max_deg"]),
        "final_label_order_ok": bool(final_diagnostics["label_order_ok"]),
        "fy00_displacement_m": float(np.linalg.norm(positions[0] - initial_positions()[0])),
        "stage_i_success": bool(stage_i_ok),
        "stage_ii_success": bool(success),
        "probe_count": runner.probe_count,
        "rollback_failures": runner.rollback_failures,
        "accepted_move_count": sum(
            1 for row in recorder.moves if row["role"] == "commit"
        ),
        "total_flight_distance_m": float(
            math.fsum(
                math.hypot(row["dx"], row["dy"])
                for row in recorder.moves
                if row["role"] in {"probe", "probe_inverse", "commit"}
            )
        ),
        "parameters": {
            "stage_i_level0": int(args.stage_i_level0),
            "stage_ii_level0": int(args.stage_ii_level0),
            "level_min": int(args.level_min),
            "base_gain_m": float(args.base_gain),
            "actuator_seed": int(args.actuator_seed),
            "stage_i_max_cycles": int(args.stage_i_max_cycles),
            "stage_ii_max_batches": int(args.stage_ii_max_batches),
        },
    }
    write_outputs(args.output_dir, recorder, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "success": summary["success"],
                "final_l_ang_rad": summary["final_l_ang_rad"],
                "final_l_pos": summary["final_l_pos"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
