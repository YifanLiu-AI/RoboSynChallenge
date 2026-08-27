#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-demo}"
MODEL_ID="scutyfan/NeurIPS26-WCM-items_handover"
MODEL_DIR="$ROOT/third_party/WCM/modelscope_wcm"
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" ]]; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -x "$candidate" ]]; then UV_BIN="$candidate"; break; fi
  done
fi

case "$MODE" in
  demo)
    VIDEO="$ROOT/examples/wcm/episode-000072.mp4"
    ALIGN="$ROOT/examples/wcm/episode-000072-alignment.json"
    test -s "$VIDEO" || { echo "Missing demo video: $VIDEO" >&2; exit 1; }
    test -s "$ALIGN" || { echo "Missing alignment report: $ALIGN" >&2; exit 1; }
    python3 - "$ALIGN" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x["mapping_verified"] is True, x
print("WCM rollout demo OK")
print(f"mapping_verified={str(x['mapping_verified']).lower()}")
print(f"source_frame_count={x['source_frame_count']}")
print(f"curve_point_count={x['curve_point_count']}")
PY
    if command -v ffprobe >/dev/null; then
      ffprobe -v error -show_entries format=duration,size \
        -of default=noprint_wrappers=1 "$VIDEO"
    fi
    echo "Open this video: $VIDEO"
    ;;
  verify)
    [[ -n "$UV_BIN" ]] || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
    cd "$ROOT/third_party/WCM"
    "$UV_BIN" sync --frozen --extra download --all-packages
    mkdir -p "$MODEL_DIR"
    "$UV_BIN" run --frozen python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
from modelscope import model_file_download
import sys
print(model_file_download(sys.argv[1], file_path="wcm/deploy.pt", local_dir=sys.argv[2]))
PY
    "$UV_BIN" run --frozen python - "$MODEL_DIR/wcm/deploy.pt" <<'PY'
import sys, torch
x = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert x["artifact_type"] == "deploy"
assert x["model"] and x["config"]
print("WCM checkpoint OK")
print("tensors=", len(x["model"]))
print("action_dim=", x["config"]["model"]["action_dim"])
print("history_size=", x["config"]["data"]["history_size"])
PY
    ;;
  test)
    [[ -n "$UV_BIN" ]] || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
    cd "$ROOT/third_party/WCM"
    "$UV_BIN" sync --frozen --extra test --all-packages
    "$UV_BIN" run --frozen pytest -q tests
    ;;
  *)
    echo "Usage: bash quickstart_wcm.sh {demo|verify|test}" >&2
    exit 2
    ;;
esac
