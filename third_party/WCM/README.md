# WCM for RoboSynChallenge

This branch integrates **WCM (World Critic Model)** training, offline evaluation, and rollout
value-curve video rendering into RoboSynChallenge.

- **Start here:** [RoboSynChallenge Quickstart](QUICKSTART_ROBOSYN.md)
- **ModelScope weights:** https://modelscope.cn/models/scutyfan/NeurIPS26-WCM-items_handover
- **Ready-made video:** [`../../examples/wcm/episode-000072.mp4`](../../examples/wcm/episode-000072.mp4)
- **Original WCM documentation:** [README_UPSTREAM.md](README_UPSTREAM.md)

## Layout

- `world_critic/`: model, dataset, training, checkpointing, and evaluation.
- `episode_value_video/`: frame-aligned rollout/value video rendering.
- `scripts/`: rollout conversion, return construction, and analysis.
- `configs/`: smoke, single-GPU, and multi-GPU configurations.
- `tests/`: conversion and pipeline regression tests.
