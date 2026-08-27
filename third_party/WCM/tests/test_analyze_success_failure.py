from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_success_failure import analyze


def test_analyze_reports_perfect_separation_and_writes_plots(tmp_path: Path) -> None:
    curves = {}
    labels = {0: True, 1: True, 2: False, 3: False}
    for episode_id, offset in enumerate((0.1, 0.2, -0.8, -0.7)):
        values = np.asarray([offset, offset + 0.05, offset + 0.1])
        returns = np.asarray([0.0, 0.0, 0.0])
        curves[episode_id] = {
            "frame": np.arange(3),
            "value": values,
            "return": returns,
            "advantage": returns - values,
        }

    result = analyze(
        curves,
        labels,
        tmp_path,
        bootstrap_samples=100,
        bootstrap_seed=7,
        curve_points=11,
    )

    assert result["num_success_episodes"] == 2
    assert result["num_failure_episodes"] == 2
    for signal in result["signals"].values():
        assert signal["roc_auc"] == 1.0
        assert signal["success_minus_failure_mean"] > 0
    assert Path(result["episode_csv"]).is_file()
    assert all(Path(path).is_file() for path in result["plots"])
    assert (tmp_path / "success_value_advantage_curves.png").is_file()
    assert (tmp_path / "failure_value_advantage_curves.png").is_file()
    assert Path(result["summary"]).is_file()
