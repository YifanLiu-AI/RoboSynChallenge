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
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[all]'
pip install -e './episode_value_video[all]'
sudo apt-get install -y ffmpeg
```

## 2. Download the items_handover checkpoint

```bash
pip install -U modelscope
python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download('scutyfan/NeurIPS26-WCM-items_handover', local_dir='modelscope_wcm'))
PY
```

Use `modelscope_wcm/wcm/deploy.pt` for compact inference, or
`modelscope_wcm/wcm/best.pt` when full training/resume state is required.

## 3. Verify the checkpoint in seconds

```bash
python - <<'PY'
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
python -m world_critic.evaluate \
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
python -m episode_value_video render \
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
python -m episode_value_video pipeline --help
```

## 6. Train

Edit the dataset root/repo in `configs/items_handover_single_gpu.yaml`, then run:

```bash
python -m world_critic.train --config configs/items_handover_single_gpu.yaml
```

Training produces full checkpoints (`best.pt`, `last.pt`, `epoch-*.pt`) and compact `deploy.pt`.
