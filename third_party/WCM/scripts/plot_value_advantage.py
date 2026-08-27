# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WCM value and advantage curves from episode curve CSV output."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-kind", default="success")
    return parser.parse_args()


def load_curves(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["episode_id"])].append(
                (int(row["frame_index"]), float(row["value"]), float(row["return"]))
            )

    curves: dict[int, dict[str, np.ndarray]] = {}
    for episode_id, episode_rows in sorted(rows.items()):
        episode_rows.sort(key=lambda item: item[0])
        data = np.asarray(episode_rows, dtype=np.float64)
        if len(np.unique(data[:, 0])) != len(data):
            raise ValueError(f"Duplicate frame indices in episode {episode_id}.")
        curves[episode_id] = {
            "frame": data[:, 0].astype(np.int64),
            "value": data[:, 1],
            "return": data[:, 2],
            "advantage": data[:, 2] - data[:, 1],
        }
    if not curves:
        raise ValueError(f"No episode curve rows found in {path}.")
    return curves


def write_advantage_csv(curves: dict[int, dict[str, np.ndarray]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "frame_index", "value", "return", "advantage"])
        for episode_id, curve in curves.items():
            for values in zip(
                curve["frame"],
                curve["value"],
                curve["return"],
                curve["advantage"],
                strict=True,
            ):
                writer.writerow([episode_id, *values])


def curve_summary(curves: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    all_values = np.concatenate([curve["value"] for curve in curves.values()])
    all_returns = np.concatenate([curve["return"] for curve in curves.values()])
    all_advantages = np.concatenate([curve["advantage"] for curve in curves.values()])
    per_episode = []
    for episode_id, curve in curves.items():
        per_episode.append(
            {
                "episode_id": episode_id,
                "num_points": int(len(curve["frame"])),
                "frame_range": [int(curve["frame"][0]), int(curve["frame"][-1])],
                "value_range": [float(curve["value"].min()), float(curve["value"].max())],
                "return_range": [float(curve["return"].min()), float(curve["return"].max())],
                "advantage_range": [
                    float(curve["advantage"].min()),
                    float(curve["advantage"].max()),
                ],
                "advantage_mean": float(curve["advantage"].mean()),
                "advantage_mae": float(np.abs(curve["advantage"]).mean()),
                "start": {
                    "value": float(curve["value"][0]),
                    "return": float(curve["return"][0]),
                    "advantage": float(curve["advantage"][0]),
                },
                "end": {
                    "value": float(curve["value"][-1]),
                    "return": float(curve["return"][-1]),
                    "advantage": float(curve["advantage"][-1]),
                },
            }
        )
    return {
        "definition": "advantage = normalized_return - predicted_value",
        "num_episodes": len(curves),
        "num_points": int(all_values.size),
        "aggregate": {
            "value_range": [float(all_values.min()), float(all_values.max())],
            "return_range": [float(all_returns.min()), float(all_returns.max())],
            "advantage_range": [float(all_advantages.min()), float(all_advantages.max())],
            "advantage_mean": float(all_advantages.mean()),
            "advantage_std": float(all_advantages.std()),
            "advantage_mae": float(np.abs(all_advantages).mean()),
            "value_return_pearson": float(np.corrcoef(all_values, all_returns)[0, 1]),
            "positive_advantage_fraction": float((all_advantages > 0).mean()),
        },
        "episodes": per_episode,
    }


def plot_overview(
    curves: dict[int, dict[str, np.ndarray]], output_dir: Path, trajectory_kind: str
) -> list[Path]:
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(12, 6))
    first_curve = next(iter(curves.values()))
    axis.plot(
        first_curve["frame"],
        first_curve["return"],
        color="#111827",
        linestyle="--",
        linewidth=2.4,
        label="Normalized return target",
        zorder=5,
    )
    for index, (episode_id, curve) in enumerate(curves.items()):
        axis.plot(
            curve["frame"],
            curve["value"],
            color=colors[index % len(colors)],
            linewidth=1.6,
            alpha=0.9,
            label=f"Episode {episode_id}",
        )
    axis.set(title=f"WCM value curves: {trajectory_kind} trajectories", xlabel="Frame", ylabel="Value")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    value_path = output_dir / f"{trajectory_kind}_value_curves.png"
    figure.savefig(value_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    for index, (episode_id, curve) in enumerate(curves.items()):
        axis.plot(
            curve["frame"],
            curve["advantage"],
            color=colors[index % len(colors)],
            linewidth=1.5,
            alpha=0.9,
            label=f"Episode {episode_id}",
        )
    axis.axhline(0.0, color="#111827", linestyle="--", linewidth=1.5, label="Zero advantage")
    axis.set(
        title=f"WCM advantage curves: {trajectory_kind} trajectories",
        xlabel="Frame",
        ylabel="Advantage (return - value)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    advantage_path = output_dir / f"{trajectory_kind}_advantage_curves.png"
    figure.savefig(advantage_path, dpi=180)
    plt.close(figure)
    return [value_path, advantage_path]


def plot_episode_panels(
    curves: dict[int, dict[str, np.ndarray]], output_dir: Path, trajectory_kind: str
) -> Path:
    figure, axes = plt.subplots(len(curves), 2, figsize=(15, 3.2 * len(curves)), squeeze=False)
    for row, (episode_id, curve) in enumerate(curves.items()):
        value_axis, advantage_axis = axes[row]
        value_axis.plot(curve["frame"], curve["return"], color="#111827", linestyle="--", label="Return")
        value_axis.plot(curve["frame"], curve["value"], color="#dc2626", label="Value")
        value_axis.set_title(f"Episode {episode_id}: value vs return")
        value_axis.set_ylabel("Value / return")
        value_axis.legend(loc="upper left")

        advantage_axis.plot(curve["frame"], curve["advantage"], color="#2563eb")
        advantage_axis.axhline(0.0, color="#111827", linestyle="--", linewidth=1.2)
        advantage_axis.set_title(f"Episode {episode_id}: advantage")
        advantage_axis.set_ylabel("Return - value")
        for axis in (value_axis, advantage_axis):
            axis.set_xlabel("Frame")
            axis.grid(True, alpha=0.25)
    figure.suptitle(f"WCM value and advantage by {trajectory_kind} episode", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path = output_dir / f"{trajectory_kind}_value_advantage_by_episode.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(args.input_csv)
    advantage_csv = args.output_dir / f"{args.trajectory_kind}_value_advantage.csv"
    write_advantage_csv(curves, advantage_csv)
    plots = plot_overview(curves, args.output_dir, args.trajectory_kind)
    plots.append(plot_episode_panels(curves, args.output_dir, args.trajectory_kind))

    summary = curve_summary(curves)
    summary["input_csv"] = str(args.input_csv.resolve())
    summary["advantage_csv"] = str(advantage_csv.resolve())
    summary["plots"] = [str(path.resolve()) for path in plots]
    summary_path = args.output_dir / f"{args.trajectory_kind}_value_advantage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary["aggregate"], "summary": str(summary_path.resolve()), "plots": summary["plots"]}, indent=2))


if __name__ == "__main__":
    main()
