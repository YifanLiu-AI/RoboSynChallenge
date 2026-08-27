# Items Handover WCM Big-Batch Benchmark

Date: 2026-08-14 (Asia/Shanghai)

## Test conditions

- GPU: NVIDIA GeForce RTX 4090, 49,140 MiB reported by `nvidia-smi`.
- Concurrent processes were preserved throughout the test: WCM batch-2 training
  PID 3079485 used 4,332 MiB and Sunshine used 410 MiB.
- Model and objective matched the production run: 170,601,985 parameters,
  bf16, three 224x224 views, history size 3, full WCM losses, gradient clipping,
  and AdamW.
- Data were isolated to episodes 0 and 2 (1,194 temporal windows). The full-run
  action normalization statistics were reused.
- Each probe performed two forward/backward/clip/optimizer steps. This includes
  the first AdamW step, where optimizer states are allocated, and a second
  steady-state step.
- `nvidia-smi` memory below is the probe process alone. Headroom subtracts the
  probe, the concurrent batch-2 process, and Sunshine from 49,140 MiB.

## Results

| Batch | Result | CUDA allocated peak | CUDA reserved peak | Probe process | Total headroom | Steady step |
|---:|:---:|---:|---:|---:|---:|---:|
| 4 | pass | 5,331 MiB | 5,502 MiB | 5,982 MiB | 38,416 MiB | 0.154 s |
| 8 | pass | 8,844 MiB | 9,364 MiB | 9,844 MiB | 34,554 MiB | 0.251 s |
| 16 | pass | 15,801 MiB | 16,574 MiB | 17,054 MiB | 27,344 MiB | 0.477 s |
| 32 | pass | 29,625 MiB | 30,568 MiB | 31,048 MiB | 13,350 MiB | 0.906 s |
| 40 | pass | 36,578 MiB | 37,402 MiB | 37,882 MiB | 6,516 MiB | 1.143 s |
| 48 | OOM | 42,400 MiB | 42,738 MiB | n/a | less than 0.7 GiB at failure | n/a |
| 64 | OOM | 42,817 MiB | 42,842 MiB | n/a | less than 0.2 GiB at failure | n/a |

The empirical concurrent pass limit is batch 40. The stable limit under the
requested 8-10 GiB reserve is batch 32, which retains about 13.0 GiB even while
the existing batch-2 training process remains resident. Batch 40 is a boundary
test, not a recommended long run. Batch 48 may behave differently after the
current process exits, but it would still leave too little stability margin and
was not demonstrated to pass under the measured conditions.

## Epoch estimate

The full training split has 82,139 windows and uses `drop_last=true`:

| Batch | Optimizer steps/epoch | Reduction vs batch 2 | Compute estimate from steady step |
|---:|---:|---:|---:|
| 2 | 41,069 | 1x | production epoch measured at 3,074.7 s total |
| 16 | 5,133 | 8x | about 40.8 min for train steps |
| 32 | 2,566 | 16x | about 38.7 min for train steps |
| 40 | 2,053 | 20x | about 39.1 min for train steps |

The production batch-2 epoch took 51.2 minutes, including 9.8 minutes of
validation at eval batch 4 and checkpoint work. A batch-32 epoch is expected to
take roughly 46-50 minutes including validation/checkpointing. This is not a
16x wall-clock speedup: there are 16x fewer steps, but each measured batch-32
step takes about 15x as long as the current batch-2 average. The estimate is
necessarily approximate because the short probe reuses a prepared batch while
the full loader decodes videos concurrently.

## Convergence interpretation

All passing probes completed two finite optimizer steps without numerical
failure. On the repeated probe batch, batch 32 loss changed from 3.4287 to
0.3399; this is only an optimizer/overfit sanity check, not evidence of full-run
validation convergence. No competing multi-epoch experiment was started while
the requested batch-2 run was active.

Batch 32 is recommended for the next full run. It gives SIGReg and value losses
a much less noisy batch while preserving a substantial memory margin. With five
epochs it performs about 12,830 optimizer updates instead of about 205,345 at
batch 2. Larger batches do not inherently guarantee a better Pearson score, so
the epoch-by-epoch validation Pearson remains the decision metric. Keep the
current conservative learning rate for the first batch-32 run; tune learning
rate or epoch count only from its validation curve rather than applying a 16x
linear learning-rate increase.

## Launcher

`scripts/run_items_handover_big_batch.sh` defaults to batch 32. It waits for the
current batch-2 training and its post-training pipeline, trains each requested
batch sequentially, then runs full validation evaluation and the
success/failure plotting analysis before starting the next batch.

```bash
scripts/run_items_handover_big_batch.sh --batch 32
scripts/run_items_handover_big_batch.sh --batch 16 --batch 32
```

Use `--no-wait-current` only after manually verifying that no other production
training should be protected. The launcher refuses to overwrite an existing
batch output directory and uses a lock to prevent two queues sharing the same
output root.
