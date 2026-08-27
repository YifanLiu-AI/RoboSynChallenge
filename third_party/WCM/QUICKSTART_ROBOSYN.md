# RoboSynChallenge WCM Quickstart

This integration contains WCM training/evaluation code and a rollout value-curve renderer.
The weights are hosted on ModelScope instead of GitHub.

## 0. See the expected result immediately

Open [`../../examples/wcm/episode-000072.mp4`](../../examples/wcm/episode-000072.mp4).
It is a real `items_handover` rollout with the WCM value estimate revealed frame by frame.
Its alignment report has `mapping_verified=true` (600 video frames and 597 predicted points).

## 1. Install

Requirements: Ubuntu, Python 3.12, CUDA-capable PyTorch, and ffmpeg.

```bash
git clone --branch wcm-integration https://github.com/oplisty/RoboSynchallenge.git
cd RoboSynchallenge/third_party/WCM
curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if uv is already installed
uv python install 3.12
uv sync --frozen --extra all --all-packages
sudo apt-get install -y ffmpeg
```

## 2. Download the items_handover checkpoint

```bash
uv run --frozen python - <<'PY'
from modelscope import model_file_download
print(model_file_download(
    'scutyfan/NeurIPS26-WCM-items_handover',
    file_path='wcm/deploy.pt',
    local_dir='modelscope_wcm',
))
PY
```

This downloads only `modelscope_wcm/wcm/deploy.pt` (about 683 MB). Download
`wcm/best.pt` separately only when full training/resume state is required; a full snapshot is about 2.1 GB.

## 3. Verify the checkpoint in seconds

```bash
uv run --frozen python - <<'PY'
import torch
p = 'modelscope_wcm/wcm/deploy.pt'
x = torch.load(p, map_location='cpu', weights_only=False)
assert x['artifact_type'] == 'deploy'
assert x['model'] and x['config']
print('WCM checkpoint OK; tensors =', len(x['model']))
print('action_dim =', x['config']['model']['action_dim'])
print('history_size =', x['config']['data']['history_size'])
PY
```

## 4. Evaluate WCM on a local LeRobot v3 rollout dataset

```bash
uv run --frozen python -m world_critic.evaluate \
  --checkpoint modelscope_wcm/wcm/deploy.pt \
  --output-dir outputs/items_handover_eval \
  --split val \
  --batch-size 32 \
  --num-workers 4 \
  --episode-curves
```

The command writes `summary.json` plus JSON/CSV/PNG episode value curves.
If the checkpoint contains an old machine-specific `data.root`, update that value to your local
LeRobot v3 dataset location before evaluation.

## 5. Render value changing with rollout

```bash
uv run --frozen python -m episode_value_video render \
  --curves outputs/items_handover_eval/episode_curves/episode_curves.json \
  --checkpoint modelscope_wcm/wcm/deploy.pt \
  --dataset-root /absolute/path/to/items_handover_lerobot_v3 \
  --camera-key observation.images.cam_high \
  --episode-id 72 \
  --output-dir outputs/items_handover_value_video \
  --speed 2.0 \
  --overwrite
```

For a combined evaluate-then-render workflow:

```bash
uv run --frozen python -m episode_value_video pipeline --help
```

## 6. Train

Edit the dataset root/repo in `configs/items_handover_single_gpu.yaml`, then run:

```bash
uv run --frozen python -m world_critic.train --config configs/items_handover_single_gpu.yaml
```

Training produces full checkpoints (`best.pt`, `last.pt`, `epoch-*.pt`) and compact `deploy.pt`.

## Fresh-environment problems

Before changing dependencies, read [TROUBLESHOOTING.md](TROUBLESHOOTING.md). It documents the locked Python/PyTorch/torchcodec/Transformers/LeRobot matrix, CUDA and FFmpeg requirements, machine-specific dataset paths, ModelScope download sizes, and offline/proxy setup.
