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

"""Quantify and plot WCM success/failure value discrimination."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

__all__ = ["analyze", "load_curves", "load_episode_labels"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3072)
    parser.add_argument("--curve-points", type=int, default=101)
    return parser.parse_args()


def load_curves(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["episode_id"])].append(
                (int(row["frame_index"]), float(row["value"]), float(row["return"]))
            )
    result = {}
    for episode_id, episode_rows in sorted(rows.items()):
        episode_rows.sort(key=lambda item: item[0])
        array = np.asarray(episode_rows, dtype=np.float64)
        if len(np.unique(array[:, 0])) != len(array):
            raise ValueError(f"Duplicate frame indices in episode {episode_id}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite curve value in episode {episode_id}.")
        result[episode_id] = {
            "frame": array[:, 0].astype(np.int64),
            "value": array[:, 1],
            "return": array[:, 2],
            "advantage": array[:, 2] - array[:, 1],
        }
    if not result:
        raise ValueError(f"No curves found in {path}.")
    return result


def load_episode_labels(path: Path) -> dict[int, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    for episode in payload["episodes"]:
        episode_id = int(episode["output_episode_id"])
        if episode_id in labels:
            raise ValueError(f"Duplicate output_episode_id={episode_id} in {path}.")
        success = episode["success"]
        if type(success) is not bool:
            raise ValueError(f"Episode {episode_id} success label is not a JSON bool.")
        labels[episode_id] = success
    return labels


def _rank_auc(success: np.ndarray, failure: np.ndarray) -> float:
    """Return P(success score > failure score), with ties worth one half."""

    comparisons = success[:, None] - failure[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def _cohens_d(success: np.ndarray, failure: np.ndarray) -> float:
    numerator = (len(success) - 1) * success.var(ddof=1) + (len(failure) - 1) * failure.var(ddof=1)
    denominator = len(success) + len(failure) - 2
    pooled = math.sqrt(float(numerator / denominator))
    return float((success.mean() - failure.mean()) / pooled) if pooled > 0 else math.inf


def _bootstrap_mean_difference(
    success: np.ndarray, failure: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_success = rng.choice(success, len(success), replace=True)
        sampled_failure = rng.choice(failure, len(failure), replace=True)
        differences[index] = sampled_success.mean() - sampled_failure.mean()
    return np.quantile(differences, [0.025, 0.975]).tolist()


def _signal_metrics(
    success: np.ndarray,
    failure: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "success": {
            "count": int(len(success)),
            "mean": float(success.mean()),
            "std": float(success.std(ddof=1)),
            "median": float(np.median(success)),
            "min": float(success.min()),
            "max": float(success.max()),
        },
        "failure": {
            "count": int(len(failure)),
            "mean": float(failure.mean()),
            "std": float(failure.std(ddof=1)),
            "median": float(np.median(failure)),
            "min": float(failure.min()),
            "max": float(failure.max()),
        },
        "success_minus_failure_mean": float(success.mean() - failure.mean()),
        "success_minus_failure_mean_bootstrap_95ci": _bootstrap_mean_difference(
            success,
            failure,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "cohens_d": _cohens_d(success, failure),
        "roc_auc": _rank_auc(success, failure),
    }


def _episode_signals(curve: dict[str, np.ndarray]) -> dict[str, float]:
    tail_count = max(1, math.ceil(len(curve["value"]) * 0.1))
    return {
        "mean_value": float(curve["value"].mean()),
        "last_value": float(curve["value"][-1]),
        "tail_10pct_mean_value": float(curve["value"][-tail_count:].mean()),
        "mean_advantage": float(curve["advantage"].mean()),
        "last_advantage": float(curve["advantage"][-1]),
        "value_mae": float(np.abs(curve["value"] - curve["return"]).mean()),
    }


def _normalized_curves(
    curves: dict[int, dict[str, np.ndarray]], ids: list[int], field: str, points: int
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, points)
    interpolated = []
    for episode_id in ids:
        values = curves[episode_id][field]
        progress = np.linspace(0.0, 1.0, len(values))
        interpolated.append(np.interp(grid, progress, values))
    return np.stack(interpolated)


def _plot_group_curves(
    curves: dict[int, dict[str, np.ndarray]],
    labels: dict[int, bool],
    output_dir: Path,
    *,
    field: str,
    points: int,
) -> Path:
    figure, axis = plt.subplots(figsize=(11, 6))
    grid = np.linspace(0.0, 1.0, points)
    styles = ((True, "Success", "#15803d"), (False, "Failure", "#b91c1c"))
    for outcome, name, color in styles:
        ids = [episode_id for episode_id in curves if labels[episode_id] is outcome]
        values = _normalized_curves(curves, ids, field, points)
        for curve in values:
            axis.plot(grid, curve, color=color, alpha=0.09, linewidth=0.8)
        mean = values.mean(axis=0)
        lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
        axis.fill_between(grid, lower, upper, color=color, alpha=0.16)
        axis.plot(grid, mean, color=color, linewidth=2.8, label=f"{name} mean (n={len(ids)})")
    axis.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ylabel = "Predicted value" if field == "value" else "Advantage (return - value)"
    axis.set(xlabel="Normalized episode progress", ylabel=ylabel, title=f"WCM {field}: success vs failure")
    axis.grid(True, alpha=0.22)
    axis.legend()
    figure.tight_layout()
    path = output_dir / f"success_failure_{field}_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_outcome_value_advantage(
    curves: dict[int, dict[str, np.ndarray]],
    labels: dict[int, bool],
    output_dir: Path,
    *,
    outcome: bool,
    points: int,
) -> Path:
    name = "Success" if outcome else "Failure"
    slug = name.lower()
    color = "#15803d" if outcome else "#b91c1c"
    ids = [episode_id for episode_id in curves if labels[episode_id] is outcome]
    grid = np.linspace(0.0, 1.0, points)
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for axis, field, ylabel in (
        (axes[0], "value", "Predicted value"),
        (axes[1], "advantage", "Advantage (return - value)"),
    ):
        values = _normalized_curves(curves, ids, field, points)
        for curve in values:
            axis.plot(grid, curve, color=color, alpha=0.08, linewidth=0.75)
        mean = values.mean(axis=0)
        lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
        axis.fill_between(
            grid,
            lower,
            upper,
            color=color,
            alpha=0.2,
            label="10th-90th percentile",
        )
        axis.plot(grid, mean, color=color, linewidth=2.8, label="Mean")
        axis.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.22)
        axis.legend(loc="best")
    axes[0].set_title(f"WCM value and advantage: {name.lower()} trajectories (n={len(ids)})")
    axes[1].set_xlabel("Normalized episode progress")
    figure.tight_layout()
    path = output_dir / f"{slug}_value_advantage_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_distributions(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    signals = ("mean_value", "last_value", "tail_10pct_mean_value")
    labels = ("Mean value", "Last evaluated value", "Tail 10% mean value")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    rng = np.random.default_rng(3072)
    for axis, signal, title in zip(axes[0], signals, labels, strict=True):
        success = np.asarray([row[signal] for row in rows if row["success"]])
        failure = np.asarray([row[signal] for row in rows if not row["success"]])
        axis.boxplot([success, failure], tick_labels=["Success", "Failure"], widths=0.5)
        axis.scatter(1 + rng.normal(0, 0.035, len(success)), success, color="#15803d", alpha=0.75, s=22)
        axis.scatter(2 + rng.normal(0, 0.035, len(failure)), failure, color="#b91c1c", alpha=0.75, s=22)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.22)
    axes[0, 0].set_ylabel("Predicted value")
    figure.suptitle("Episode-level WCM value discrimination")
    figure.tight_layout()
    path = output_dir / "success_failure_value_distributions.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def analyze(
    curves: dict[int, dict[str, np.ndarray]],
    labels: dict[int, bool],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    curve_points: int,
) -> dict[str, Any]:
    missing = set(curves) - set(labels)
    if missing:
        raise ValueError(f"Curves have no genuine success labels: {sorted(missing)}")
    rows = [
        {"episode_id": episode_id, "success": labels[episode_id], **_episode_signals(curve)}
        for episode_id, curve in curves.items()
    ]
    success_count = sum(row["success"] for row in rows)
    failure_count = len(rows) - success_count
    if success_count < 2 or failure_count < 2:
        raise ValueError("Discrimination analysis requires at least two success and two failure episodes.")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "episode_discrimination.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {}
    for offset, signal in enumerate(("mean_value", "last_value", "tail_10pct_mean_value")):
        success = np.asarray([row[signal] for row in rows if row["success"]])
        failure = np.asarray([row[signal] for row in rows if not row["success"]])
        metrics[signal] = _signal_metrics(
            success,
            failure,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
    plots = [
        _plot_outcome_value_advantage(
            curves, labels, output_dir, outcome=True, points=curve_points
        ),
        _plot_outcome_value_advantage(
            curves, labels, output_dir, outcome=False, points=curve_points
        ),
        _plot_group_curves(curves, labels, output_dir, field="value", points=curve_points),
        _plot_group_curves(curves, labels, output_dir, field="advantage", points=curve_points),
        _plot_distributions(rows, output_dir),
    ]
    result = {
        "definition": "advantage = normalized return target - predicted value",
        "num_episodes": len(rows),
        "num_success_episodes": success_count,
        "num_failure_episodes": failure_count,
        "signals": metrics,
        "episode_csv": str(row_path.resolve()),
        "plots": [str(path.resolve()) for path in plots],
    }
    summary_path = output_dir / "success_failure_discrimination.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary"] = str(summary_path.resolve())
    return result


def main() -> None:
    args = parse_args()
    result = analyze(
        load_curves(args.input_csv),
        load_episode_labels(args.selection_manifest),
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        curve_points=args.curve_points,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
