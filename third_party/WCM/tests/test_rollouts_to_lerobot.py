from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import numpy as np

from scripts.rollouts_to_lerobot import (
    RolloutEpisode,
    build_stratified_split,
    load_rollout_manifests,
    load_rollouts,
    select_ratio,
    write_dataset,
)
from scripts.add_returns_to_lerobot import copy_metadata_sidecars


def _write_manifest(root: Path, outcomes: list[bool]) -> None:
    episodes = root / "episodes"
    episodes.mkdir(parents=True)
    rows = []
    for episode_id, success in enumerate(outcomes):
        episode_file = episodes / f"episode_{episode_id:03d}.npz"
        episode_file.touch()
        rows.append(
            {
                "schema_version": "wcm_rollout_v1",
                "episode_id": episode_id,
                "seed": 1000 + episode_id,
                "instruction": "handover",
                "success": success,
                "truncated": not success,
                "length": 10,
                "episode_file": f"episodes/{episode_file.name}",
            }
        )
    (root / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_auto_selection_uses_exact_largest_two_to_one_ratio(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [True] * 7 + [False] * 4)
    selected = select_ratio(
        load_rollouts([tmp_path]),
        success_count=None,
        failure_count=None,
        selection_seed=42,
    )
    assert len(selected) == 9
    assert sum(episode.success for episode in selected) == 6


def test_selection_is_deterministic(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [True] * 8 + [False] * 4)
    episodes = load_rollouts([tmp_path])
    first = select_ratio(episodes, success_count=4, failure_count=2, selection_seed=7)
    second = select_ratio(episodes, success_count=4, failure_count=2, selection_seed=7)
    assert [episode.seed for episode in first] == [episode.seed for episode in second]


def test_selection_rejects_non_two_to_one_counts(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [True] * 4 + [False] * 2)
    with pytest.raises(ValueError, match="2:1"):
        select_ratio(
            load_rollouts([tmp_path]),
            success_count=3,
            failure_count=2,
            selection_seed=1,
        )


def test_load_rejects_duplicate_seed_across_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_manifest(first, [True])
    _write_manifest(second, [False])
    with pytest.raises(ValueError, match="Duplicate rollout seed"):
        load_rollouts([first, second])


def test_load_preselected_manifest_resolves_paths_from_explicit_root(tmp_path: Path) -> None:
    batch = tmp_path / "outputs" / "batch000"
    _write_manifest(batch, [True, True, False])
    selected = tmp_path / "selected.jsonl"
    rows = [json.loads(line) for line in (batch / "episode_metrics.jsonl").read_text().splitlines()]
    for row in rows:
        row["episode_file"] = f"outputs/batch000/{row['episode_file']}"
    selected.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    episodes = load_rollout_manifests([selected], episode_root=tmp_path)

    assert [episode.seed for episode in episodes] == [1000, 1001, 1002]
    assert [episode.success for episode in episodes] == [True, True, False]


def test_load_preselected_manifest_rejects_path_outside_explicit_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.npz"
    outside.touch()
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "wcm_rollout_v1",
                "episode_id": 0,
                "seed": 1000,
                "instruction": "handover",
                "success": True,
                "truncated": False,
                "length": 10,
                "episode_file": "../outside.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe episode_file"):
        load_rollout_manifests([manifest], episode_root=tmp_path)


def test_stratified_split_preserves_two_to_one_ratio(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [True] * 10 + [False] * 5)
    selected = load_rollouts([tmp_path])
    split = build_stratified_split(selected, val_fraction=0.2, split_seed=9)

    assert len(split["train"]) == 12
    assert len(split["val"]) == 3
    for ids in (split["train"], split["val"]):
        outcomes = [selected[index].success for index in ids]
        assert sum(outcomes) == 2 * (len(outcomes) - sum(outcomes))
    assert sorted(split["train"] + split["val"]) == list(range(15))


def test_copy_metadata_sidecars_preserves_bytes_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    payload = b'{"train":[0,1],"val":[2]}\n'
    (source / "meta" / "episode_split.json").write_bytes(payload)

    records = copy_metadata_sidecars(source, output)

    assert (output / "meta" / "episode_split.json").read_bytes() == payload
    assert records == [
        {
            "path": "meta/episode_split.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]


def test_completed_staging_dataset_resumes_without_duplicate_episodes(tmp_path: Path) -> None:
    episodes = []
    outcomes = [True, True, True, True, False, False]
    for episode_id, success in enumerate(outcomes):
        episode_file = tmp_path / f"source_{episode_id}.npz"
        images = np.full((3, 16, 16, 3), episode_id, dtype=np.uint8)
        np.savez(
            episode_file,
            schema_version=np.asarray("wcm_rollout_v1"),
            episode_id=np.asarray(episode_id),
            seed=np.asarray(100 + episode_id),
            instruction=np.asarray("test task"),
            action=np.zeros((3, 1, 2), dtype=np.float32),
            success=np.full((3,), success),
            **{
                "observation.state": np.zeros((3, 2), dtype=np.float32),
                "observation.images.cam_high": images,
                "observation.images.cam_left_wrist": images,
                "observation.images.cam_right_wrist": images,
            },
        )
        episodes.append(
            RolloutEpisode(
                source_root=tmp_path,
                episode_file=episode_file,
                episode_id=episode_id,
                seed=100 + episode_id,
                instruction="test task",
                success=success,
                truncated=not success,
                length=3,
            )
        )

    output = tmp_path / "dataset"
    kwargs = dict(
        output_dir=output,
        repo_id="test/resume",
        fps=25,
        selection_seed=1,
        val_fraction=0.2,
        split_seed=1,
        encoder_threads=1,
        device="cpu",
        streaming_encoding=True,
    )
    write_dataset(episodes, **kwargs)
    output.rename(output.with_name(output.name + ".inprogress"))
    write_dataset(episodes, resume=True, **kwargs)

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 6
    assert info["total_frames"] == 18
    manifest = json.loads((output / "meta" / "rollout_selection.json").read_text())
    assert [record["seed"] for record in manifest["episodes"]] == list(range(100, 106))
