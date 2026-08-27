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

"""Select recorded pi0.5 rollouts and write a canonical LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "RolloutEpisode",
    "build_stratified_split",
    "load_rollout_manifests",
    "load_rollouts",
    "select_ratio",
    "write_dataset",
]

SCHEMA_VERSION = "wcm_rollout_v1"
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class RolloutEpisode:
    source_root: Path
    episode_file: Path
    episode_id: int
    seed: int
    instruction: str
    success: bool
    truncated: bool
    length: int


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a JSON boolean, got {value!r}.")
    return value


def load_rollout_manifests(
    manifests: Iterable[Path], *, episode_root: Path | None = None
) -> list[RolloutEpisode]:
    episodes: list[RolloutEpisode] = []
    seen_sources: set[Path] = set()
    seen_seeds: set[int] = set()
    resolved_episode_root = episode_root.expanduser().resolve() if episode_root is not None else None
    for manifest_value in manifests:
        manifest = manifest_value.expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing rollout manifest: {manifest}")
        root = resolved_episode_root or manifest.parent
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{manifest}:{line_number} has schema {record.get('schema_version')!r}; "
                    f"expected {SCHEMA_VERSION!r}."
                )
            relative_path = Path(record["episode_file"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe episode_file at {manifest}:{line_number}: {relative_path}")
            episode_file = (root / relative_path).resolve()
            if root not in episode_file.parents or not episode_file.is_file():
                raise FileNotFoundError(f"Missing rollout episode: {episode_file}")
            if episode_file in seen_sources:
                raise ValueError(f"Duplicate rollout source: {episode_file}")
            seed = int(record["seed"])
            if seed in seen_seeds:
                raise ValueError(f"Duplicate rollout seed {seed}; repeated initial states are not allowed.")
            seen_sources.add(episode_file)
            seen_seeds.add(seed)
            length = int(record["length"])
            if length < 1:
                raise ValueError(f"Rollout {episode_file} has non-positive length {length}.")
            episodes.append(
                RolloutEpisode(
                    source_root=root,
                    episode_file=episode_file,
                    episode_id=int(record["episode_id"]),
                    seed=seed,
                    instruction=str(record.get("instruction", "")),
                    success=_strict_bool(record["success"], field="success"),
                    truncated=_strict_bool(record["truncated"], field="truncated"),
                    length=length,
                )
            )
    if not episodes:
        raise ValueError("No rollout episodes were found.")
    return episodes


def load_rollouts(roots: Iterable[Path]) -> list[RolloutEpisode]:
    """Load canonical per-batch manifests with paths relative to each batch root."""

    manifests = [root.expanduser().resolve() / "episode_metrics.jsonl" for root in roots]
    return load_rollout_manifests(manifests)


def select_ratio(
    episodes: Iterable[RolloutEpisode],
    *,
    success_count: int | None,
    failure_count: int | None,
    selection_seed: int,
) -> list[RolloutEpisode]:
    successes = [episode for episode in episodes if episode.success]
    failures = [episode for episode in episodes if not episode.success]
    if failure_count is None:
        failure_count = min(len(failures), len(successes) // 2)
    if success_count is None:
        success_count = failure_count * 2
    if success_count != failure_count * 2:
        raise ValueError(
            f"Exact success:failure=2:1 is required, got {success_count}:{failure_count}."
        )
    if failure_count < 1:
        raise ValueError("At least one genuine failed rollout is required.")
    if success_count > len(successes) or failure_count > len(failures):
        raise ValueError(
            "Requested selection exceeds available rollouts: "
            f"requested={success_count} success/{failure_count} failure, "
            f"available={len(successes)} success/{len(failures)} failure."
        )
    rng = random.Random(selection_seed)
    selected = rng.sample(successes, success_count) + rng.sample(failures, failure_count)
    rng.shuffle(selected)
    return selected


def build_stratified_split(
    selected: list[RolloutEpisode], *, val_fraction: float, split_seed: int
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between zero and one.")
    success_ids = [index for index, episode in enumerate(selected) if episode.success]
    failure_ids = [index for index, episode in enumerate(selected) if not episode.success]
    if len(success_ids) != 2 * len(failure_ids) or len(failure_ids) < 2:
        raise ValueError("A stratified split requires an exact 2:1 dataset with at least two failures.")
    rng = random.Random(split_seed)
    rng.shuffle(success_ids)
    rng.shuffle(failure_ids)
    val_failure_count = max(1, min(len(failure_ids) - 1, round(len(failure_ids) * val_fraction)))
    val_success_count = 2 * val_failure_count
    val = sorted(success_ids[:val_success_count] + failure_ids[:val_failure_count])
    train = sorted(success_ids[val_success_count:] + failure_ids[val_failure_count:])
    return {
        "schema_version": 1,
        "seed": split_seed,
        "val_fraction": val_fraction,
        "stratified_by": "episode_success",
        "ratio": "2:1",
        "train": train,
        "val": val,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: Path, *, full_hash: bool = False) -> str:
    """Cheap, stable checkpoint identity; full hashing is opt-in because NPZs are huge."""
    if full_hash:
        return _sha256(path)
    stat = path.stat()
    return f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"


def _validate_episode(episode: RolloutEpisode, arrays: Any) -> tuple[int, int, int]:
    required = {
        "schema_version",
        "episode_id",
        "seed",
        "instruction",
        "action",
        "success",
        "observation.state",
        *(f"observation.images.{camera}" for camera in CAMERA_KEYS),
    }
    missing = sorted(required - set(arrays.files))
    if missing:
        raise KeyError(f"{episode.episode_file} is missing fields: {missing}")
    if str(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"NPZ schema mismatch in {episode.episode_file}.")
    actions = np.asarray(arrays["action"])
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions[:, 0]
    states = np.asarray(arrays["observation.state"])
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError(f"Action/state ranks must be two in {episode.episode_file}.")
    if len(actions) != episode.length or len(states) != episode.length:
        raise ValueError(f"Manifest/array length mismatch in {episode.episode_file}.")
    if not np.all(np.asarray(arrays["success"], dtype=bool) == episode.success):
        raise ValueError(f"Manifest/NPZ success mismatch in {episode.episode_file}.")
    image_shape: tuple[int, int, int] | None = None
    for camera in CAMERA_KEYS:
        images = np.asarray(arrays[f"observation.images.{camera}"])
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Invalid RGB array for {camera} in {episode.episode_file}: {images.shape}")
        if len(images) != episode.length:
            raise ValueError(f"Image length mismatch for {camera} in {episode.episode_file}.")
        current_shape = tuple(map(int, images.shape[1:]))
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(f"Camera image shapes differ in {episode.episode_file}.")
    assert image_shape is not None
    return int(actions.shape[1]), int(states.shape[1]), image_shape[0]


def write_dataset(
    selected: list[RolloutEpisode],
    *,
    output_dir: Path,
    repo_id: str,
    fps: int,
    selection_seed: int,
    val_fraction: float,
    split_seed: int,
    encoder_threads: int,
    device: str = "cpu",
    vcodec: str | None = None,
    streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 60,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    resume: bool = False,
    hash_source: bool = False,
    legacy_layout: bool = False,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = output_dir.expanduser().resolve()
    if fps < 1 or encoder_threads < 1 or encoder_queue_maxsize < 1:
        raise ValueError("fps and encoder_threads must be positive.")

    # Keep an incomplete conversion out of the requested final directory.  A
    # completed staging directory is renamed into place atomically.
    staging_dir = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if output_dir.exists():
        staging_dir = output_dir
    elif staging_dir.exists():
        resume = True
    staging_dir.parent.mkdir(parents=True, exist_ok=True)

    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cpu" and vcodec in {"h264_nvenc", "auto"}:
        raise ValueError(f"{vcodec} conflicts with --device cpu; use --device cuda or auto")
    if device == "cuda" and vcodec not in {None, "h264_nvenc", "auto"}:
        raise ValueError(f"{vcodec} does not use CUDA/NVENC; use --device cpu")
    if device in {"cuda", "auto"}:
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        try:
            import av
            av.codec.Codec("h264_nvenc", "w")
            nvenc_ok = True
        except Exception:
            nvenc_ok = False
        if device == "cuda" and not (cuda_ok and nvenc_ok):
            raise RuntimeError("--device cuda requested, but CUDA or PyAV h264_nvenc is unavailable")
        if cuda_ok and nvenc_ok:
            vcodec = vcodec or "h264_nvenc"
        elif device == "auto":
            vcodec = "h264"
            print("[convert] CUDA/NVENC unavailable; falling back to CPU h264", flush=True)
    vcodec = vcodec or "h264"

    first = selected[0]
    with np.load(first.episode_file, allow_pickle=False) as arrays:
        action_dim, state_dim, _ = _validate_episode(first, arrays)
        image_shape = tuple(map(int, arrays["observation.images.cam_high"].shape[1:]))
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "episode_success": {"dtype": "int8", "shape": (1,), "names": None},
    }
    for camera in CAMERA_KEYS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }

    create_kwargs = dict(
        repo_id=repo_id,
        root=staging_dir,
        fps=fps,
        robot_type="cobotmagic",
        features=features,
        use_videos=True,
        video_backend="pyav",
        vcodec=vcodec,
        batch_encoding_size=1,
        encoder_threads=encoder_threads,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=max(encoder_queue_maxsize, max(ep.length for ep in selected)),
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        metadata_buffer_size=1,
    )
    if resume:
        dataset = LeRobotDataset.resume(**{k: v for k, v in create_kwargs.items() if k not in {"fps", "features", "robot_type", "use_videos", "metadata_buffer_size"}})
        dataset.meta._metadata_buffer_size = 1
    else:
        dataset = LeRobotDataset.create(**create_kwargs)
    # One episode per parquet/video file makes the episode commit boundary
    # recoverable and avoids repeated O(N^2) video concatenation.
    if not legacy_layout:
        dataset.meta.update_chunk_settings(data_files_size_in_mb=0.001, video_files_size_in_mb=0.001)
    completed = int(dataset.meta.total_episodes) if resume else 0
    if completed > len(selected):
        raise RuntimeError("Existing dataset has more episodes than the selected manifest")
    records: list[dict[str, Any]] = []
    checkpoint_path = staging_dir / "meta" / "rollout_selection.progress.json"
    if checkpoint_path.exists():
        try:
            records = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("episodes", [])
        except (ValueError, OSError):
            records = []
    if len(records) > completed:
        records = records[:completed]
    for index, record in enumerate(records):
        episode = selected[index]
        expected = _source_fingerprint(episode.episode_file, full_hash=hash_source)
        if record.get("source_fingerprint") != expected or record.get("seed") != episode.seed:
            raise RuntimeError(f"Resume source mismatch at output episode {index}")
    # LeRobot info.json is committed before this sidecar.  A crash in that tiny
    # window is recoverable by rebuilding the deterministic source records.
    for index in range(len(records), completed):
        episode = selected[index]
        record = asdict(episode)
        record.update(
            {
                "source_root": str(episode.source_root),
                "episode_file": str(episode.episode_file),
                "output_episode_id": index,
                "source_fingerprint": _source_fingerprint(
                    episode.episode_file, full_hash=hash_source
                ),
            }
        )
        records.append(record)
    try:
        for output_episode_id, episode in enumerate(selected):
            if output_episode_id < completed:
                continue
            with np.load(episode.episode_file, allow_pickle=False) as arrays:
                current_action_dim, current_state_dim, _ = _validate_episode(episode, arrays)
                if (current_action_dim, current_state_dim) != (action_dim, state_dim):
                    raise ValueError(f"Feature dimensions changed in {episode.episode_file}.")
                actions = np.asarray(arrays["action"], dtype=np.float32)
                if actions.ndim == 3:
                    actions = actions[:, 0]
                states = np.asarray(arrays["observation.state"], dtype=np.float32)
                images = {
                    camera: np.asarray(arrays[f"observation.images.{camera}"])
                    for camera in CAMERA_KEYS
                }
                for frame_index in range(episode.length):
                    frame: dict[str, Any] = {
                        "observation.state": states[frame_index],
                        "action": actions[frame_index],
                        "episode_success": np.asarray([episode.success], dtype=np.int8),
                        "task": episode.instruction,
                    }
                    frame.update(
                        {
                            f"observation.images.{camera}": images[camera][frame_index]
                            for camera in CAMERA_KEYS
                        }
                    )
                    dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=True)
            # Close a valid parquet footer at every commit boundary.  The next
            # episode opens fresh files because of the small shard limit above.
            if not legacy_layout:
                dataset.writer.close_writer()
                dataset.meta._close_writer()
            record = asdict(episode)
            record.update(
                {
                    "source_root": str(episode.source_root),
                    "episode_file": str(episode.episode_file),
                    "output_episode_id": output_episode_id,
                    "source_fingerprint": _source_fingerprint(episode.episode_file, full_hash=hash_source),
                }
            )
            records.append(record)
            checkpoint = {"schema_version": 1, "episodes": records}
            tmp_checkpoint = checkpoint_path.with_suffix(".tmp")
            tmp_checkpoint.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
            os.replace(tmp_checkpoint, checkpoint_path)
            print(
                f"[convert] {output_episode_id + 1}/{len(selected)} "
                f"success={episode.success} seed={episode.seed} frames={episode.length}",
                flush=True,
            )
        dataset.finalize()
    except BaseException:
        writer = getattr(dataset, "writer", None)
        if writer is not None:
            writer.stop_image_writer()
        raise

    success_count = sum(episode.success for episode in selected)
    manifest = {
        "schema_version": 1,
        "source_schema": SCHEMA_VERSION,
        "selection_seed": selection_seed,
        "selection": {
            "success": success_count,
            "failure": len(selected) - success_count,
            "ratio": "2:1",
        },
        "repo_id": repo_id,
        "fps": fps,
        "episodes": records,
    }
    (staging_dir / "meta" / "rollout_selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    split = build_stratified_split(
        selected, val_fraction=val_fraction, split_seed=split_seed
    )
    (staging_dir / "meta" / "episode_split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    if staging_dir != output_dir:
        checkpoint_path.unlink(missing_ok=True)
        os.replace(staging_dir, output_dir)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--rollout-root", action="append", type=Path)
    source.add_argument(
        "--manifest",
        type=Path,
        help="Exact preselected JSONL manifest; its episode paths are resolved below --episode-root.",
    )
    result.add_argument(
        "--episode-root",
        type=Path,
        help="Trusted path root for --manifest episode_file entries (default: manifest directory).",
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--repo-id", required=True)
    result.add_argument("--success-count", type=int)
    result.add_argument("--failure-count", type=int)
    result.add_argument("--selection-seed", type=int, default=3072)
    result.add_argument("--val-fraction", type=float, default=0.2)
    result.add_argument("--split-seed", type=int, default=3072)
    result.add_argument("--fps", type=int, default=25)
    result.add_argument("--encoder-threads", type=int, default=4)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    result.add_argument("--vcodec", choices=("h264", "h264_nvenc", "auto"), default=None)
    result.add_argument("--streaming-encoding", action="store_true", help="Feed RGB arrays directly to PyAV; avoids PNG round-trip.")
    result.add_argument("--encoder-queue-maxsize", type=int, default=60)
    result.add_argument("--image-writer-threads", type=int, default=4)
    result.add_argument("--image-writer-processes", type=int, default=0)
    result.add_argument("--resume", action="store_true", help="Resume output or its .inprogress staging directory.")
    result.add_argument("--hash-source", action="store_true", help="Use full SHA-256 source hashes (slower; default uses size+mtime).")
    result.add_argument("--legacy-layout", action="store_true", help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.manifest is not None:
        episodes = load_rollout_manifests([args.manifest], episode_root=args.episode_root)
        successes = sum(episode.success for episode in episodes)
        failures = len(episodes) - successes
        expected_successes = args.success_count if args.success_count is not None else successes
        expected_failures = args.failure_count if args.failure_count is not None else failures
        if successes != expected_successes or failures != expected_failures:
            raise ValueError(
                "Preselected manifest counts do not match the requested counts: "
                f"manifest={successes} success/{failures} failure, "
                f"requested={expected_successes} success/{expected_failures} failure."
            )
        if failures < 1 or successes != 2 * failures:
            raise ValueError(
                f"Preselected manifest must have exact success:failure=2:1, got {successes}:{failures}."
            )
        selected = episodes
    else:
        if args.episode_root is not None:
            raise ValueError("--episode-root is only valid with --manifest.")
        episodes = load_rollouts(args.rollout_root)
        selected = select_ratio(
            episodes,
            success_count=args.success_count,
            failure_count=args.failure_count,
            selection_seed=args.selection_seed,
        )
    successes = sum(episode.success for episode in episodes)
    print(
        f"[source] episodes={len(episodes)} success={successes} "
        f"failure={len(episodes) - successes}",
        flush=True,
    )
    write_dataset(
        selected,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        selection_seed=args.selection_seed,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        encoder_threads=args.encoder_threads,
        device=args.device,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        resume=args.resume,
        hash_source=args.hash_source,
        legacy_layout=args.legacy_layout,
    )


if __name__ == "__main__":
    main()
