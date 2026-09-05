#!/usr/bin/env python3
"""Data-driven plots and the slow first-round animation for v2."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-x2v2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import uav_no_odometry_optimizer as v2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
REPRESENTATIVE_DIR = FIGURE_DIR / "representative_frames"
FIGURE_DATA_DIR = FIGURE_DIR / "figure_data"
TAU_DEG = 0.001 * 180.0 / np.pi

plt.rcParams.update(
    {
        "font.family": ["Songti SC", "Arial Unicode MS", "Hiragino Sans GB", "sans-serif"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = pd.read_csv(OUTPUT_DIR / "state_history.csv")
    moves = pd.read_csv(OUTPUT_DIR / "moves.csv")
    summary = pd.read_csv(OUTPUT_DIR / "phase_summary.csv")
    return states, moves, summary


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIGURE_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def formation_axes(ax: plt.Axes, row: pd.Series, label: str) -> None:
    xs = np.array([row[f"true_x_FY{k:02d}"] for k in range(10)])
    ys = np.array([row[f"true_y_FY{k:02d}"] for k in range(10)])
    peripheral = list(range(1, 10)) + [1]
    ax.plot(
        xs[peripheral], ys[peripheral],
        color="#7f8c8d", linewidth=0.8, linestyle="--", alpha=0.8,
        label="外围标签顺序" if label else None,
    )
    ax.scatter(xs[1:], ys[1:], s=22, c="#1f77b4", zorder=3, label="FY01–FY09")
    ax.scatter([xs[0]], [ys[0]], s=45, c="#d62728", marker="s", zorder=4, label="FY00")
    for k in range(10):
        ax.annotate(
            f"FY{k:02d}", (xs[k], ys[k]), xytext=(3, 3),
            textcoords="offset points", fontsize=6.5,
        )
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.text(
        0.02, 0.98,
        f"{label}\n$L_{{ang}}$={row['l_ang_rad']:.3e} rad\n$L_{{pos}}$={row['l_pos']:.3e}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )


def plot_initial_final(states: pd.DataFrame) -> None:
    initial = states.iloc[0]
    final_rows = states[states["event"] == "final_success"]
    final = final_rows.iloc[-1] if not final_rows.empty else states.iloc[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), constrained_layout=True)
    formation_axes(axes[0], initial, "初始")
    formation_axes(axes[1], final, "最终")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="lower center",
        ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03),
    )
    save_all(fig, "fig_initial_final")


def plot_loss_curves(states: pd.DataFrame) -> None:
    data = states[["state_step", "phase", "event", "l_ang_rad", "l_pos"]].copy()
    data.to_csv(FIGURE_DATA_DIR / "fig_loss_curves.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot(data["state_step"], data["l_ang_rad"], marker="o", markersize=2.5, linewidth=1, color="#d62728")
    axes[0].set_ylabel(r"$L_{\mathrm{ang}}$ / rad")
    axes[0].grid(True, linewidth=0.3, alpha=0.35)
    axes[1].plot(data["state_step"], data["l_pos"], marker="o", markersize=2.5, linewidth=1, color="#1f77b4")
    axes[1].set_ylabel(r"$L_{\mathrm{pos}}$")
    axes[1].set_xlabel("正式调整状态序号")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    for ax in axes:
        ax.axvline(5, color="#555555", linestyle=":", linewidth=0.8)
    axes[0].text(5.1, 0.92, "阶段 I 完成", transform=axes[0].get_xaxis_transform(), fontsize=7)
    save_all(fig, "fig_loss_curves")


def plot_angle_diagnostics(states: pd.DataFrame) -> None:
    key_columns = [
        "key_250_rad", "key_850_rad", "key_258_rad",
        "key_280_rad", "key_580_rad", "key_285_rad",
    ]
    local_columns = [f"local_rms_FY{k:02d}_rad" for k in v2.STAGE_II_RECEIVERS]
    stage_i = states.dropna(subset=key_columns).copy()
    stage_ii = states.dropna(subset=local_columns).copy()
    data = states[["state_step", "phase", "event", *key_columns, *local_columns]].copy()
    data.to_csv(FIGURE_DATA_DIR / "fig_angle_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
    for col in key_columns:
        axes[0].plot(
            stage_i["state_step"], np.degrees(stage_i[col].abs()),
            marker="o", markersize=2.5, linewidth=0.9, label=col.replace("key_", "").replace("_rad", ""),
        )
    axes[0].axhline(TAU_DEG, color="black", linestyle=":", linewidth=0.8, label=r"阈值 $10^{-3}$ rad")
    axes[0].set_ylabel("关键角残差绝对值 / (°)")
    axes[0].set_xlabel("正式调整状态序号")
    axes[0].grid(True, linewidth=0.3, alpha=0.35)
    axes[0].legend(ncol=4)

    for col in local_columns:
        axes[1].plot(
            stage_ii["state_step"], np.degrees(stage_ii[col]),
            marker="o", markersize=2.2, linewidth=0.9,
            label=col.replace("local_rms_FY", "FY").replace("_rad", ""),
        )
    axes[1].axhline(TAU_DEG, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("四锚角签名 RMS / (°)")
    axes[1].set_xlabel("正式调整状态序号")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    axes[1].legend(ncol=6)
    save_all(fig, "fig_angle_diagnostics")


def parse_pulse_column(moves: pd.DataFrame) -> pd.DataFrame:
    parsed = moves["pulse"].astype(str).str.extract(
        r"Pulse\(primitive='(?P<primitive>[A-Z])', level=(?P<level>\d+)\)"
    )
    moves = moves.copy()
    moves["primitive"] = parsed["primitive"]
    moves["level"] = pd.to_numeric(parsed["level"], errors="coerce")
    moves["accepted"] = moves["role"].eq("commit").astype(int)
    moves["flight_step_m"] = np.hypot(moves["dx"], moves["dy"])
    moves["accepted_cumulative"] = moves["accepted"].cumsum()
    moves["flight_cumulative_m"] = moves["flight_step_m"].cumsum()
    return moves


def plot_control_budget(moves: pd.DataFrame) -> None:
    budget = parse_pulse_column(moves)
    columns = [
        "move_id", "phase", "state_step", "candidate_id", "receiver", "role",
        "primitive", "level", "accepted", "accepted_cumulative",
        "flight_step_m", "flight_cumulative_m",
    ]
    budget[columns].to_csv(FIGURE_DATA_DIR / "fig_control_budget.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), constrained_layout=True)
    axes[0].plot(
        budget["move_id"], budget["accepted_cumulative"],
        linewidth=1, color="#2c7fb8", label="累计接受脉冲",
    )
    axes[0].set_ylabel("累计接受数")
    axes[0].set_xlabel("隐藏动作事件序号")
    axes[0].grid(True, linewidth=0.3, alpha=0.35)
    axes[0].legend()

    commits = budget[budget["accepted"].eq(1)]
    for receiver, group in commits.groupby("receiver"):
        axes[1].plot(
            group["accepted_cumulative"], group["level"],
            marker="o", markersize=2.4, linewidth=0.8, label=f"FY{int(receiver):02d}",
        )
    axes[1].set_ylabel("命令级别")
    axes[1].set_xlabel("累计接受脉冲")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    axes[1].legend(ncol=4, fontsize=6)

    axes[2].plot(
        budget["move_id"], budget["flight_cumulative_m"],
        linewidth=1, color="#7f2704", label="累计实际路程",
    )
    axes[2].set_ylabel("累计路程 / m")
    axes[2].set_xlabel("隐藏动作事件序号")
    axes[2].grid(True, linewidth=0.3, alpha=0.35)
    axes[2].legend()
    save_all(fig, "fig_control_budget")


def key_angle_values(world: v2.World, compiler: v2.SignatureCompiler) -> dict[str, float]:
    signatures = v2.stage_i_signatures(compiler)
    batches = {
        receiver: world.sensor.batch(receiver, signature.emitters)
        for receiver, signature in signatures.items()
    }
    return v2.stage_i_key_extras(batches, signatures)


def build_animation_frames(states: pd.DataFrame, moves: pd.DataFrame):
    compiler = v2.SignatureCompiler()
    positions = states.iloc[0][
        [f"true_x_FY{k:02d}" for k in range(10)]
        + [f"true_y_FY{k:02d}" for k in range(10)]
    ].to_numpy(dtype=float).reshape(2, 10).T
    stage_i_moves = moves[moves["phase"].eq("stage_i")].sort_values("move_id")
    frames: list[dict[str, object]] = []
    for row in stage_i_moves.itertuples(index=False):
        if row.role not in {"probe", "commit"}:
            continue
        candidate = positions.copy()
        receiver = int(row.receiver)
        candidate[receiver] += np.array([row.dx, row.dy], dtype=float)
        world = v2.World(candidate)
        values = key_angle_values(world, compiler)
        fy05 = candidate[5]
        fy08 = candidate[8]
        frames.append(
            {
                "frame_id": len(frames) + 1,
                "move_id": int(row.move_id),
                "role": row.role,
                "receiver": receiver,
                "candidate_id": int(row.candidate_id),
                "pulse": row.pulse,
                "positions": candidate,
                "l_ang_rad": v2.global_l_ang(world, compiler),
                "fy05_r": float(np.hypot(fy05[0], fy05[1])),
                "fy05_theta_deg": float(np.degrees(np.arctan2(fy05[1], fy05[0])) % 360.0),
                "fy08_r": float(np.hypot(fy08[0], fy08[1])),
                "fy08_theta_deg": float(np.degrees(np.arctan2(fy08[1], fy08[0])) % 360.0),
                **values,
            }
        )
        if row.role == "commit":
            positions = candidate
    return frames


def draw_animation_frame(frame: dict[str, object]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 7.6), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.15)
    positions = np.asarray(frame["positions"], dtype=float)
    peripheral = list(range(1, 10)) + [1]
    ax.plot(
        positions[peripheral, 0], positions[peripheral, 1],
        color="#9ecae1", linewidth=0.8, linestyle="--", alpha=0.8,
    )
    ax.scatter(positions[1:, 0], positions[1:, 1], s=22, c="#1f77b4", zorder=3, label="FY01–FY09")
    ax.scatter(positions[5, 0], positions[5, 1], s=55, c="#2ca25f", zorder=4, label="FY05")
    ax.scatter(positions[8, 0], positions[8, 1], s=55, c="#756bb1", zorder=4, label="FY08")
    ax.scatter([0], [0], s=50, c="#d62728", marker="s", zorder=5, label="FY00 固定")
    for k in range(10):
        ax.annotate(f"FY{k:02d}", (positions[k, 0], positions[k, 1]), xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    circle = plt.Circle((0, 0), float(frame["fy05_r"]), fill=False, color="#2ca25f", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.add_patch(circle)
    ax.set_xlim(-130, 130)
    ax.set_ylim(-130, 130)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    action = str(frame["pulse"]).replace("Pulse(", "").replace(")", "")
    role = "试探" if frame["role"] == "probe" else "接受提交"
    receiver = f"FY{int(frame['receiver']):02d}"
    ax.set_title(f"阶段 I 微步 {int(frame['frame_id']):02d}/36：{receiver} {role}  {action}", fontsize=9)
    key_text = "\n".join(
        f"{name.replace('key_', '').replace('_rad', '')}: {abs(float(frame[name])) * 180 / np.pi:.4f}°"
        for name in (
            "key_250_rad", "key_850_rad", "key_258_rad",
            "key_280_rad", "key_580_rad", "key_285_rad",
        )
    )
    polar_text = (
        f"FY05: r={float(frame['fy05_r']):.3f} m, θ={float(frame['fy05_theta_deg']):.3f}°\n"
        f"FY08: r={float(frame['fy08_r']):.3f} m, θ={float(frame['fy08_theta_deg']):.3f}°"
    )
    ax.text(
        0.02, 0.98, polar_text, transform=ax.transAxes, va="top", fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82},
    )
    ax.text(
        0.98, 0.98,
        key_text + f"\n$L_{{ang}}$={float(frame['l_ang_rad']):.3e} rad",
        transform=ax.transAxes, va="top", ha="right", fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82},
    )
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), frameon=False)
    return fig


def make_animation(states: pd.DataFrame, moves: pd.DataFrame) -> list[dict[str, object]]:
    frames = build_animation_frames(states, moves)
    if not 30 <= len(frames) <= 40:
        raise AssertionError(f"expected 30-40 first-round frames, got {len(frames)}")
    durations: list[int] = []
    pil_frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for frame in frames:
            fig = draw_animation_frame(frame)
            path = Path(temp_dir) / f"frame_{int(frame['frame_id']):03d}.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            pil_frames.append(Image.open(path).convert("P", palette=Image.ADAPTIVE, colors=128))
            if int(frame["frame_id"]) == 1:
                durations.append(1800)
            elif frame["role"] == "commit":
                durations.append(1000)
            else:
                durations.append(480)
        pil_frames[0].save(
            FIGURE_DIR / "anim_first_round_slow.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
    columns = [
        "frame_id", "move_id", "role", "receiver", "candidate_id", "pulse",
        "l_ang_rad", "fy05_r", "fy05_theta_deg", "fy08_r", "fy08_theta_deg",
        "key_250_rad", "key_850_rad", "key_258_rad",
        "key_280_rad", "key_580_rad", "key_285_rad",
    ]
    pd.DataFrame([{k: frame[k] for k in columns} for frame in frames]).to_csv(
        FIGURE_DATA_DIR / "anim_first_round_frames.csv", index=False
    )
    return frames


def save_representative_state(row: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.6), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.10)
    formation_axes(ax, row, title)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def make_representative_frames(states: pd.DataFrame) -> None:
    REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)
    selections = [
        ("01_initial.png", states.iloc[0], "初始"),
        ("02_fy05_first_accepted.png", states[states["event"].eq("accepted_FY05")].iloc[0], "FY05 首次调整后"),
        ("03_fy08_first_accepted.png", states[states["event"].eq("accepted_FY08")].iloc[0], "FY08 首次调整后"),
        ("04_after_joint_alternation.png", states[states["event"].eq("accepted_FY05")].iloc[1], "一次联合交替后"),
        ("05_stage_i_complete.png", states[states["event"].eq("stage_i_complete")].iloc[-1], "阶段 I 完成"),
    ]
    for filename, row, title in selections:
        save_representative_state(row, REPRESENTATIVE_DIR / filename, title)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    states, moves, _ = read_inputs()
    plot_initial_final(states)
    plot_loss_curves(states)
    plot_angle_diagnostics(states)
    plot_control_budget(moves)
    make_representative_frames(states)
    frames = make_animation(states, moves)
    print(f"figures written to {FIGURE_DIR}; GIF frames={len(frames)}")


if __name__ == "__main__":
    main()
