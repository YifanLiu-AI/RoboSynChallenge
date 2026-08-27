#!/usr/bin/env python3
"""Benchmark legacy and high-throughput rollout conversion on six real episodes."""

from __future__ import annotations

import argparse
import resource
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from scripts.rollouts_to_lerobot import load_rollout_manifests, write_dataset


def _subset(episodes):
    successes = [episode for episode in episodes if episode.success][:4]
    failures = [episode for episode in episodes if not episode.success][:2]
    if len(successes) != 4 or len(failures) != 2:
        raise ValueError("Benchmark requires at least four successes and two failures")
    return successes + failures


def _gpu_sampler(stop: threading.Event, samples: list[tuple[int, int]]) -> None:
    while not stop.wait(0.5):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.encoder",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            gpu, encoder = result.stdout.splitlines()[0].split(",")
            samples.append((int(gpu.strip()), int(encoder.strip())))
        except (IndexError, ValueError):
            return


def _usage_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _validate(path: Path, repo_id: str, expected_episodes: int) -> dict[str, object]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_reader import DatasetReader

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=path)
    reader = DatasetReader(
        meta=meta,
        root=path,
        episodes=None,
        tolerance_s=1e-4,
        video_backend="pyav",
        delta_timestamps=None,
        image_transforms=None,
    )
    if not reader.try_load():
        raise AssertionError("LeRobot DatasetReader rejected the local dataset")
    if reader.num_episodes != expected_episodes:
        raise AssertionError((reader.num_episodes, expected_episodes))
    first = reader.get_item(0)
    last = reader.get_item(reader.num_frames - 1)
    camera_keys = [f"observation.images.{name}" for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    shapes = {key: tuple(first[key].shape) for key in camera_keys}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values())) != (3, 480, 640):
        raise AssertionError(shapes)
    labels = {int(first["episode_success"].item()), int(last["episode_success"].item())}
    if labels != {0, 1}:
        raise AssertionError(labels)
    return {"frames": reader.num_frames, "shapes": shapes, "endpoint_labels": sorted(labels)}


def _run(name: str, episodes, root: Path, *, optimized: bool) -> dict[str, object]:
    repo_id = f"benchmark/{name}"
    output = root / name
    stop = threading.Event()
    samples: list[tuple[int, int]] = []
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True)
    before_cpu = _usage_seconds()
    started = time.perf_counter()
    sampler.start()
    try:
        write_dataset(
            episodes,
            output_dir=output,
            repo_id=repo_id,
            fps=25,
            selection_seed=3072,
            val_fraction=0.2,
            split_seed=3072,
            encoder_threads=4,
            device="cuda" if optimized else "cpu",
            streaming_encoding=optimized,
            encoder_queue_maxsize=60,
            hash_source=not optimized,
            legacy_layout=not optimized,
        )
    finally:
        stop.set()
        sampler.join()
    wall = time.perf_counter() - started
    cpu = _usage_seconds() - before_cpu
    validation = _validate(output, repo_id, len(episodes))
    return {
        "mode": name,
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_percent_one_core_100": round(cpu / wall * 100, 1),
        "gpu_avg_percent": round(sum(x[0] for x in samples) / len(samples), 1) if samples else None,
        "gpu_max_percent": max((x[0] for x in samples), default=None),
        "encoder_avg_percent": round(sum(x[1] for x in samples) / len(samples), 1) if samples else None,
        "encoder_max_percent": max((x[1] for x in samples), default=None),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("both", "legacy_cpu", "streaming_nvenc"), default="both")
    parser.add_argument("--output-root", type=Path, help="Keep benchmark outputs here instead of using a temporary directory.")
    args = parser.parse_args()
    episodes = _subset(load_rollout_manifests([args.manifest], episode_root=args.episode_root))
    modes = ((False, "legacy_cpu"), (True, "streaming_nvenc"))
    if args.mode != "both":
        modes = tuple(mode for mode in modes if mode[1] == args.mode)
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=False)
        for optimized, name in modes:
            print(_run(name, episodes, args.output_root, optimized=optimized), flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="rollout_conversion_benchmark_") as temp:
            for optimized, name in modes:
                print(_run(name, episodes, Path(temp), optimized=optimized), flush=True)


if __name__ == "__main__":
    main()
