from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import (
    collectively_validate,
    inspect_checkpoint_config,
    load_training_checkpoint,
    save_deploy_bundle,
    save_training_checkpoint,
)
from .config import (
    apply_runtime_overrides,
    config_argument_parser,
    load_config,
    save_resolved_config,
    validate_train_config,
)
from .data import (
    WorldCriticCollator,
    build_datasets,
    build_episode_split,
    build_processor,
    episode_ids_from_dataset,
    fit_action_normalization,
    infer_feature_dim,
    load_episode_split,
    load_lerobot_dataset,
    save_episode_split,
)
from .distributed import (
    DistributedEvalSampler,
    barrier,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
)
from .model import SIGReg
from .training import (
    autocast_context,
    build_model,
    compute_losses,
    create_optimizer,
    create_scheduler,
    evaluate_loader,
    move_batch_to_device,
    seed_everything,
    wrap_ddp,
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _tracking_task(repo_id: str) -> str:
    configured = os.environ.get("WCM_TASK")
    if configured:
        return configured
    name = repo_id.rsplit("/", 1)[-1]
    marker = "_wcm_"
    if marker not in name:
        raise RuntimeError(
            "cannot derive SwanLab task from data.repo_id; set WCM_TASK explicitly"
        )
    return name.split(marker, 1)[0]


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def run() -> None:
    args = config_argument_parser(
        "Train the language-conditioned World Critic Model."
    ).parse_args()
    config = apply_runtime_overrides(load_config(args.config))
    validate_train_config(config)
    ctx = initialize_distributed(config.expected_world_size, config.ddp_timeout_minutes)
    swanlab_run = None
    swanlab_metadata_path = None
    swanlab_metadata = None
    try:
        if config.resume:
            saved_config = inspect_checkpoint_config(config.resume, ctx)
            config.data.action_mean = saved_config["data"].get("action_mean")
            config.data.action_std = saved_config["data"].get("action_std")
            validate_train_config(config)
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError(
                "gradient_accumulation_steps>1 is intentionally disabled: return masks and global "
                "SIGReg require accumulation-window normalization for exact DDP semantics."
            )
        seed_everything(config.seed, config.deterministic)
        output_dir = Path(config.output_dir).resolve()

        # Any rank-0-only filesystem operation must be wrapped in a
        # collective validation.  A plain ``barrier`` after a failing mkdir or
        # config write would leave the other seven ranks waiting forever (the
        # most common failure mode when a shared filesystem is read-only or
        # temporarily unavailable).
        def prepare_output_directory() -> None:
            if ctx.is_main:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_resolved_config(config, output_dir / "resolved_config.json")

        collectively_validate(
            ctx, "Output directory/config preparation", prepare_output_directory
        )
        barrier(ctx)

        manifest_path = output_dir / "episode_split.json"

        def prepare_episode_manifest() -> None:
            if (
                config.data.split_manifest is None
                and not manifest_path.exists()
                and ctx.is_main
            ):
                manifest_dataset = load_lerobot_dataset(config.data)
                try:
                    manifest_split = build_episode_split(
                        episode_ids_from_dataset(manifest_dataset),
                        config.data.val_fraction,
                        config.data.split_seed,
                    )
                    save_episode_split(
                        manifest_split,
                        manifest_path,
                        config.data.split_seed,
                        config.data.val_fraction,
                    )
                finally:
                    del manifest_dataset
                    import gc

                    gc.collect()

        collectively_validate(
            ctx, "Episode split preparation", prepare_episode_manifest
        )
        barrier(ctx)

        def prepare_action_statistics() -> None:
            if (
                config.data.normalize_action
                and config.data.action_mean is None
                and ctx.is_main
            ):
                stats_dataset = load_lerobot_dataset(config.data)
                try:
                    stats_split_path = (
                        Path(config.data.split_manifest)
                        if config.data.split_manifest
                        else manifest_path
                    )
                    stats_split = load_episode_split(stats_split_path)
                    available_stats_episodes = set(
                        map(int, episode_ids_from_dataset(stats_dataset).tolist())
                    )
                    if not set(stats_split.train).issubset(available_stats_episodes):
                        raise ValueError(
                            "Split manifest references training episodes absent from the dataset."
                        )
                    fit_action_normalization(
                        stats_dataset, config.data, stats_split.train
                    )
                finally:
                    del stats_dataset
                    import gc

                    gc.collect()

        collectively_validate(
            ctx, "Training action-statistics preparation", prepare_action_statistics
        )
        if ctx.distributed:
            config.data.action_mean = broadcast_object(
                config.data.action_mean if ctx.is_main else None,
                ctx,
            )
            config.data.action_std = broadcast_object(
                config.data.action_std if ctx.is_main else None,
                ctx,
            )
        barrier(ctx)
        base_dataset = train_dataset = val_dataset = split = None

        def prepare_datasets_and_schema() -> None:
            nonlocal base_dataset, train_dataset, val_dataset, split
            base_dataset, train_dataset, val_dataset, split = build_datasets(
                config.data,
                manifest_path=manifest_path,
            )
            if len(train_dataset) == 0:
                raise ValueError(
                    "Training split contains no valid temporal windows. Reduce history_size or add longer episodes."
                )
            required_windows = (
                ctx.world_size * config.per_device_batch_size
                if ctx.distributed
                else config.per_device_batch_size
            )
            if len(train_dataset) < required_windows:
                raise ValueError(
                    "Training split is too small for one complete training batch: "
                    f"windows={len(train_dataset)}, required={required_windows}."
                )
            inferred_action_dim = infer_feature_dim(
                base_dataset, config.data.action_key
            )
            if config.model.action_dim is None:
                config.model.action_dim = inferred_action_dim
            elif config.model.action_dim != inferred_action_dim:
                raise ValueError(
                    f"Configured action_dim={config.model.action_dim}, dataset has {inferred_action_dim}."
                )
            if config.model.predict_state_vector:
                if config.data.state_key is None:
                    raise ValueError("predict_state_vector requires data.state_key.")
                if config.data.state_key not in base_dataset.features:
                    raise KeyError(
                        "predict_state_vector=true requires the configured state feature "
                        f"{config.data.state_key!r} to be present in the dataset."
                    )
                inferred_state_dim = infer_feature_dim(
                    base_dataset, config.data.state_key
                )
                if config.model.state_dim is None:
                    config.model.state_dim = inferred_state_dim
                elif config.model.state_dim != inferred_state_dim:
                    raise ValueError(
                        f"Configured state_dim={config.model.state_dim}, dataset has {inferred_state_dim}."
                    )

        collectively_validate(
            ctx, "Dataset/schema preparation", prepare_datasets_and_schema
        )

        def save_resolved_config_after_schema_inference() -> None:
            if ctx.is_main:
                save_resolved_config(config, output_dir / "resolved_config.json")

        collectively_validate(
            ctx,
            "Resolved config write after schema inference",
            save_resolved_config_after_schema_inference,
        )

        def initialize_swanlab() -> None:
            nonlocal swanlab_run, swanlab_metadata_path, swanlab_metadata
            if not ctx.is_main:
                return
            task = _tracking_task(config.data.repo_id)
            project = os.environ.get("WCM_SWANLAB_PROJECT", "robosynchallenge-wcm")
            experiment_name = os.environ.get(
                "WCM_SWANLAB_EXPERIMENT",
                f"wcm-{task}-seed{config.seed}",
            )
            mode = os.environ.get("WCM_SWANLAB_MODE", "online").lower()
            if mode not in {"online", "disabled"}:
                raise RuntimeError(
                    f"WCM_SWANLAB_MODE={mode!r} cannot produce the required remote URL; "
                    "use online or disabled"
                )
            swanlab_metadata_path = output_dir / "swanlab_run.json"
            previous_metadata = None
            if swanlab_metadata_path.is_file():
                previous_metadata = json.loads(
                    swanlab_metadata_path.read_text(encoding="utf-8")
                )
                for key, expected in (("task", task), ("project", project)):
                    if previous_metadata.get(key) != expected:
                        raise RuntimeError(
                            f"existing SwanLab metadata {key} mismatch: "
                            f"{previous_metadata.get(key)!r} != {expected!r}"
                        )
                if (
                    previous_metadata.get("run_id")
                    and previous_metadata.get("mode") != mode
                ):
                    raise RuntimeError(
                        "refusing to change SwanLab mode after a remote run was created"
                    )
            swanlab_metadata = {
                "schema_version": 1,
                "task": task,
                "project": project,
                "name": experiment_name,
                "mode": mode,
                "status": "skipped" if mode == "disabled" else "initializing",
                "run_id": (
                    None
                    if previous_metadata is None
                    else previous_metadata.get("run_id")
                ),
                "url": (
                    None if previous_metadata is None else previous_metadata.get("url")
                ),
                "live_metrics_logged": mode == "online",
            }
            _atomic_json(swanlab_metadata_path, swanlab_metadata)
            if mode == "disabled":
                print("SWANLAB SKIP: WCM_SWANLAB_MODE=disabled", flush=True)
                return
            try:
                import swanlab
            except ImportError as exc:
                raise RuntimeError(
                    "SwanLab SDK is missing; install the WCM 'tracking' dependency or set "
                    "WCM_SWANLAB_MODE=disabled"
                ) from exc

            tags = ["wcm", task, f"batch-{config.per_device_batch_size}"]
            if config.resume:
                tags.append("resumed")
            run_config = asdict(config)
            run_config["tracking"] = {
                "task": task,
                "dataset_repo_id": config.data.repo_id,
                "train_episodes": len(split.train),
                "val_episodes": len(split.val),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                "world_size": ctx.world_size,
            }
            init_kwargs = {
                "project": project,
                "name": experiment_name,
                "config": run_config,
                "log_dir": str(output_dir / "swanlog"),
                "mode": "online",
                "tags": tags,
            }
            if swanlab_metadata["run_id"]:
                init_kwargs.update(id=swanlab_metadata["run_id"], resume="must")
            swanlab_run = swanlab.init(
                **init_kwargs,
            )
            swanlab_metadata.update(
                status="training",
                run_id=str(swanlab_run.id),
                url=getattr(swanlab_run, "url", None),
            )
            _atomic_json(swanlab_metadata_path, swanlab_metadata)
            print(
                json.dumps(
                    {
                        "swanlab_project": project,
                        "swanlab_experiment": experiment_name,
                        "swanlab_url": getattr(swanlab_run, "url", None),
                    }
                )
            )

        collectively_validate(ctx, "SwanLab initialization", initialize_swanlab)
        barrier(ctx)
        if ctx.is_main:
            print(
                json.dumps(
                    {
                        "train_windows": len(train_dataset),
                        "val_windows": 0 if val_dataset is None else len(val_dataset),
                        "train_episodes": len(split.train),
                        "val_episodes": len(split.val),
                        "world_size": ctx.world_size,
                        "effective_global_batch": config.per_device_batch_size
                        * ctx.world_size
                        * config.gradient_accumulation_steps,
                    },
                    indent=2,
                )
            )

        processor = None
        collator = None

        def prepare_processor() -> None:
            nonlocal processor, collator
            processor = build_processor(config.model)
            collator = WorldCriticCollator(
                processor,
                config.model.vision.image_size,
                config.model.language.max_length,
            )

        collectively_validate(ctx, "Processor/tokenizer preparation", prepare_processor)
        assert collator is not None
        train_sampler = None
        train_loader = None
        val_loader = None

        def prepare_loaders() -> None:
            nonlocal train_sampler, train_loader, val_loader
            if ctx.distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=ctx.world_size,
                    rank=ctx.rank,
                    shuffle=True,
                    seed=config.seed,
                    drop_last=True,
                )
            generator = torch.Generator().manual_seed(config.seed + ctx.rank)
            common_loader = {
                "num_workers": config.num_workers,
                "pin_memory": ctx.device.type == "cuda",
                "persistent_workers": config.num_workers > 0,
                "collate_fn": collator,
                "worker_init_fn": worker_init_fn,
            }
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.per_device_batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                drop_last=True,
                generator=generator,
                **common_loader,
            )
            if len(train_loader) == 0:
                raise RuntimeError(
                    "Training DataLoader has zero batches; increase the dataset size or reduce "
                    "per_device_batch_size."
                )
            if val_dataset is not None:
                val_sampler = (
                    DistributedEvalSampler(val_dataset, ctx.rank, ctx.world_size)
                    if ctx.distributed
                    else None
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.eval_batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    drop_last=False,
                    **common_loader,
                )

        collectively_validate(ctx, "DataLoader/sampler preparation", prepare_loaders)
        assert train_loader is not None

        model = None

        def prepare_model() -> None:
            nonlocal model
            model = build_model(config).to(ctx.device)
            if config.compile:
                model = torch.compile(model)

        collectively_validate(ctx, "Model construction/compilation", prepare_model)
        assert model is not None
        model = wrap_ddp(model, ctx)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if ctx.is_main:
            print(
                json.dumps(
                    {
                        "model_parameters": total_parameters,
                        "trainable_parameters": trainable_parameters,
                    }
                )
            )
        optimizer = None
        scheduler = None
        sigreg = None

        def prepare_optimization() -> None:
            nonlocal optimizer, scheduler, sigreg
            optimizer = create_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config, len(train_loader))
            sigreg = SIGReg(
                knots=config.loss.sigreg_knots,
                num_projections=config.loss.sigreg_num_projections,
            ).to(ctx.device)

        collectively_validate(
            ctx, "Optimizer/scheduler preparation", prepare_optimization
        )
        assert optimizer is not None and scheduler is not None and sigreg is not None

        start_epoch = 0
        global_step = 0
        best_metric = math.inf
        if config.resume:
            checkpoint = load_training_checkpoint(
                config.resume,
                model,
                optimizer,
                scheduler,
                ctx,
                expected_config=config,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_metric"])
        if ctx.is_main and swanlab_run is not None:
            swanlab_run.log(
                {
                    "run/initialized": 1,
                    "run/resumed": int(config.resume is not None),
                    "run/start_epoch": start_epoch,
                    "run/train_windows": len(train_dataset),
                    "run/val_windows": 0 if val_dataset is None else len(val_dataset),
                },
                step=global_step,
            )

        optimizer.zero_grad(set_to_none=True)
        last_log_time = time.monotonic()
        last_log_step = global_step
        effective_global_batch = (
            config.per_device_batch_size
            * ctx.world_size
            * config.gradient_accumulation_steps
        )
        for epoch in range(start_epoch, config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_started = time.monotonic()
            for batch_index, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, ctx.device)
                accumulation_index = batch_index % config.gradient_accumulation_steps
                sync_step = (
                    accumulation_index == config.gradient_accumulation_steps - 1
                    or batch_index + 1 == len(train_loader)
                )
                sync_context = (
                    model.no_sync()
                    if ctx.distributed and not sync_step
                    else torch.enable_grad()
                )
                with sync_context:
                    with autocast_context(ctx.device, config.precision):
                        output = model(
                            images=batch["images"],
                            actions=batch["actions"],
                            instruction_input_ids=batch["instruction_input_ids"],
                            instruction_attention_mask=batch[
                                "instruction_attention_mask"
                            ],
                            valid_mask=batch["valid_mask"],
                        )
                        loss, loss_parts = compute_losses(
                            output,
                            batch,
                            config.loss,
                            sigreg,
                            ctx,
                            global_step,
                        )
                        loss = loss / config.gradient_accumulation_steps
                    loss.backward()

                if sync_step:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if ctx.is_main and (
                        global_step % config.log_every == 0
                        or batch_index + 1 == len(train_loader)
                    ):
                        now = time.monotonic()
                        elapsed = max(now - last_log_time, 1e-12)
                        steps_per_second = (global_step - last_log_step) / elapsed
                        printable = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": float(loss_parts["loss"]),
                            "lr": scheduler.get_last_lr()[0],
                            "steps_per_second": steps_per_second,
                            "samples_per_second": steps_per_second
                            * effective_global_batch,
                            **{
                                key: float(value)
                                for key, value in loss_parts.items()
                                if key.endswith("loss") and key != "loss"
                            },
                        }
                        if ctx.device.type == "cuda":
                            printable.update(
                                cuda_memory_allocated_mib=torch.cuda.memory_allocated(
                                    ctx.device
                                )
                                / (1024**2),
                                cuda_memory_reserved_mib=torch.cuda.memory_reserved(
                                    ctx.device
                                )
                                / (1024**2),
                            )
                        print(json.dumps(printable))
                        with (output_dir / "training_metrics.jsonl").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write(json.dumps(printable) + "\n")
                        if swanlab_run is not None:
                            swanlab_run.log(
                                {
                                    "train/loss": printable["loss"],
                                    "train/lr": printable["lr"],
                                    "train/epoch": epoch,
                                    "train/global_step": global_step,
                                    "system/steps_per_second": printable[
                                        "steps_per_second"
                                    ],
                                    "system/samples_per_second": printable[
                                        "samples_per_second"
                                    ],
                                    **{
                                        f"train/{key}": value
                                        for key, value in printable.items()
                                        if key.endswith("loss") and key != "loss"
                                    },
                                    **{
                                        f"system/{key}": value
                                        for key, value in printable.items()
                                        if key.startswith("cuda_memory_")
                                    },
                                },
                                step=global_step,
                            )
                        last_log_time = now
                        last_log_step = global_step

            metrics = None
            is_best = False
            if val_loader is not None and (epoch + 1) % config.eval_every_epochs == 0:
                metrics = evaluate_loader(model, val_loader, config, ctx)

                def validate_validation_metrics() -> None:
                    if metrics is None or not math.isfinite(metrics["value_mse"]):
                        raise FloatingPointError(
                            f"Validation MSE is non-finite: {metrics}"
                        )

                collectively_validate(
                    ctx, "Validation metric sanity check", validate_validation_metrics
                )
                # All ranks normally receive bit-identical reduced metrics,
                # but make the checkpoint branch explicitly collective.  A
                # one-bit numerical discrepancy must never make rank 0 enter
                # ``save_training_checkpoint`` while another rank skips it.
                if ctx.distributed:
                    decision = None
                    if ctx.is_main:
                        is_best = metrics["value_mse"] < best_metric
                        decision = {
                            "is_best": bool(is_best),
                            "best_metric": min(best_metric, metrics["value_mse"]),
                        }
                    decision = broadcast_object(decision, ctx)
                    is_best = bool(decision["is_best"])
                    best_metric = float(decision["best_metric"])
                else:
                    is_best = metrics["value_mse"] < best_metric
                    best_metric = min(best_metric, metrics["value_mse"])

                def write_validation_metrics() -> None:
                    if ctx.is_main:
                        print(
                            json.dumps(
                                {"epoch": epoch, "validation": metrics}, indent=2
                            )
                        )
                        with (output_dir / "metrics.jsonl").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

                collectively_validate(
                    ctx, "Validation metrics write", write_validation_metrics
                )
                if ctx.is_main and swanlab_run is not None:
                    swanlab_run.log(
                        {
                            "validation/epoch": epoch,
                            "val_mse": float(metrics["value_mse"]),
                            "val_pearson": float(metrics["value_pearson"]),
                            **{
                                f"validation/{key}": float(value)
                                for key, value in metrics.items()
                            },
                        },
                        step=global_step,
                    )

            if is_best:
                save_training_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_metric,
                    ctx,
                )

            if (
                epoch + 1
            ) % config.save_every_epochs == 0 or epoch + 1 == config.epochs:
                save_training_checkpoint(
                    output_dir / "checkpoints" / f"epoch-{epoch:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_metric,
                    ctx,
                )
            save_training_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                global_step,
                best_metric,
                ctx,
            )
            barrier(ctx)
            if ctx.is_main:
                print(f"epoch={epoch} elapsed_s={time.monotonic() - epoch_started:.1f}")

        save_deploy_bundle(output_dir / "deploy.pt", model, config, ctx)
        if ctx.is_main and ctx.device.type == "cuda":
            peak_memory = {
                "cuda_peak_memory_allocated_mib": torch.cuda.max_memory_allocated(
                    ctx.device
                )
                / (1024**2),
                "cuda_peak_memory_reserved_mib": torch.cuda.max_memory_reserved(
                    ctx.device
                )
                / (1024**2),
            }
            print(json.dumps(peak_memory))
            if swanlab_run is not None:
                swanlab_run.log(
                    {f"system/{key}": value for key, value in peak_memory.items()},
                    step=global_step,
                )
    finally:
        if swanlab_run is not None:
            error = sys.exc_info()[1]
            if error is None:
                swanlab_run.finish()
            elif isinstance(error, KeyboardInterrupt):
                swanlab_run.finish(state="aborted")
            else:
                swanlab_run.finish(state="crashed", error=str(error))
            if swanlab_metadata_path is not None and swanlab_metadata is not None:
                swanlab_metadata["status"] = (
                    "training_complete" if error is None else "training_failed"
                )
                _atomic_json(swanlab_metadata_path, swanlab_metadata)
        cleanup_distributed(ctx)


if __name__ == "__main__":
    run()
