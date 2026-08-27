# WCM Fresh-environment notes and troubleshooting

The checked-in `uv.lock` is the source of truth. The integration was validated with Python 3.12;
`.python-version` and project metadata intentionally reject Python 3.11 and 3.13.

## Known compatibility constraints

| Component | Locked/tested | Why it matters |
| --- | --- | --- |
| Python | 3.12.x | The WCM environment has only been validated on 3.12. |
| PyTorch | 2.7.1 | `torchcodec 0.5` targets this PyTorch ABI generation. |
| torchvision | 0.22.1 | Must match PyTorch 2.7.x. |
| torchcodec | 0.5 | LeRobot permits newer versions, but newer wheels target newer PyTorch ABIs. |
| Transformers | 5.16.1 | Transformers 4 requires `huggingface-hub<1`, conflicting with LeRobot 0.5.1. |
| LeRobot | 0.5.1 | The dataset and evaluator use LeRobot v3 metadata/layout. |
| huggingface-hub | 1.29.0 | Required by LeRobot 0.5.1. |
| ModelScope | 1.39.1 | Used for checkpoint download. |

Do not casually run `pip install -U torch transformers torchcodec`. Use `uv sync --frozen` so these
versions stay coherent.

## Reproducible install

```bash
cd third_party/WCM
uv python install 3.12
uv sync --frozen --extra all --all-packages
uv run --frozen pytest -q tests
```

`--all-packages` installs both workspace members: `world-critic-model` and
`wcm-episode-value-video`.

## CUDA and PyTorch wheels

The lock resolves PyTorch 2.7.1 from PyPI. On Linux, its CUDA runtime dependencies are installed by
the wheel, but the host still needs a sufficiently recent NVIDIA driver. Check:

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print('torch:', torch.__version__)
print('built CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
PY
```

If `torch.cuda.is_available()` is false, verify the NVIDIA driver/container GPU passthrough first.
CPU-only checkpoint structure verification still works, but actual evaluation/training is intended
for CUDA and will be slow or impractical on CPU.

If your platform requires a different PyTorch wheel index (for example CPU-only or a site-specific
CUDA mirror), regenerate a platform-specific lock deliberately; do not mix an arbitrary torch wheel
with the checked-in torchcodec wheel.

## FFmpeg / torchcodec

Video rendering needs an `ffmpeg` executable. Dataset video decoding through torchcodec also needs
loadable FFmpeg shared libraries.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
ldconfig -p | grep -E 'libav(codec|format|util)'
```

Typical symptoms are `Could not load libtorchcodec`, missing `libavcodec`, or an ABI mismatch. Keep
`torchcodec==0.5` and install system FFmpeg before changing Python packages. The pure video renderer
can use PyAV/ffmpeg, but WCM evaluation of LeRobot video rows still depends on the configured dataset
video backend.

## Transformers / huggingface-hub conflict

Do not downgrade to Transformers 4. LeRobot 0.5.1 requires `huggingface-hub>=1`, while Transformers 4
requires `<1`; that combination is unsatisfiable. The lock uses Transformers 5.

## Dataset root is machine-specific

The uploaded checkpoint preserves the original absolute `config.data.root`. That path does not exist
on a new machine. Pass your local LeRobot v3 root to the video renderer with `--dataset-root`; for
offline evaluation, update/override the checkpoint dataset root according to your local deployment.
The dataset must contain the expected keys:

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
action
observation.state
return
episode_index / frame_index / task metadata
```

A checkpoint can load successfully while evaluation still fails because the dataset root, camera
keys, episode split, or LeRobot schema differs.

## ModelScope download size

- `wcm/deploy.pt`: about 683 MB; recommended for inference/evaluation.
- `wcm/best.pt`: about 1.54 GB; full resume/training state.
- Entire repository: about 2.1 GB plus small artifacts.

Use `model_file_download(..., file_path='wcm/deploy.pt')` for the quick verification. Do not use
`snapshot_download` unless you intentionally want the whole repository. Ensure enough free disk for
the download, uv environment, and uv cache.

## Offline and proxy environments

The first `uv sync` and ModelScope download require network access. Configure `HTTPS_PROXY` if needed.
After artifacts and the uv cache are populated, use:

```bash
uv sync --frozen --offline --extra all --all-packages
```

This only succeeds if every required wheel is already cached.

## Lock maintenance

Check that metadata and lock agree:

```bash
uv lock --check
```

Only update dependencies intentionally:

```bash
uv lock --upgrade-package <package>
uv sync --frozen --extra all --all-packages
uv run pytest -q tests
```
