# WCM Integration Handoff

This document is the shortest path for the next engineer to inspect, verify, and reproduce the
RoboSynChallenge WCM demo.

## Published artifacts

- GitHub branch: https://github.com/oplisty/RoboSynchallenge/tree/wcm-integration
- ModelScope: https://modelscope.cn/models/scutyfan/NeurIPS26-WCM-items_handover
- Code: `third_party/WCM/`
- Ready-made demo: `examples/wcm/episode-000072.mp4`
- Full Quickstart: `third_party/WCM/QUICKSTART_ROBOSYN.md`

## 30-second smoke check (no GPU, checkpoint, or dataset)

```bash
git clone --branch wcm-integration https://github.com/oplisty/RoboSynchallenge.git
cd RoboSynchallenge
bash quickstart_wcm.sh demo
```

The command validates the checked-in rollout video and its alignment report, then prints the video
path. Open it with your desktop video player. The expected report is:

```text
mapping_verified=true
source_frame_count=600
curve_point_count=597
```

## Verify the real checkpoint

This downloads the compact `deploy.pt` (about 683 MB) from ModelScope and checks its WCM metadata:

```bash
bash quickstart_wcm.sh verify
```

## Reproduce evaluation and video rendering

A local LeRobot v3 `items_handover` dataset is required because the repository intentionally does
not commit the rollout dataset. Follow:

```text
third_party/WCM/QUICKSTART_ROBOSYN.md
```

Core commands:

```bash
cd third_party/WCM
python -m world_critic.evaluate --help
python -m episode_value_video render --help
python -m world_critic.train --help
```

## Model files

- `wcm/deploy.pt`: compact inference/offline-evaluation bundle.
- `wcm/best.pt`: full training/resume checkpoint.
- `wcm/resolved_config.json`: exact model/data/training configuration.
- `wcm/evaluation_summary.json`: existing validation metrics.
- `wcm/episode_curves.json`: 51 validation episode curves.

## Validation completed before handoff

- WCM tests: `10 passed`.
- Train/evaluate/video CLIs: smoke-tested.
- Demo alignment: `mapping_verified=true`.
- GitHub implementation commit: `6dc7fca3ca19b87b12d7fdaeeb3caca3120914c1`.
- ModelScope upload commit: `9bb9bb451a75a88e0633ff67697783726c561255`.
- ModelScope remote sizes and SHA256 values were checked through its repository API.

## Reproducible fresh environment

WCM includes `.python-version` and `uv.lock`. Use:

```bash
cd third_party/WCM
uv python install 3.12
uv sync --frozen --extra all --all-packages
uv run --frozen pytest -q tests
```

Read `third_party/WCM/TROUBLESHOOTING.md` before changing torch, torchcodec, Transformers, LeRobot,
FFmpeg, CUDA, or dataset paths.
