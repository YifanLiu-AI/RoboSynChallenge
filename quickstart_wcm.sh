#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-demo}"
MODEL_ID="scutyfan/NeurIPS26-WCM-items_handover"
MODEL_DIR="$ROOT/third_party/WCM/modelscope_wcm"

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
    command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
    python3 - <<'PY'
try:
    import modelscope  # noqa: F401
except ImportError:
    raise SystemExit("Install ModelScope first: pip install -U modelscope")
PY
    mkdir -p "$MODEL_DIR"
    python3 - "$MODEL_ID" "$MODEL_DIR" <<'PY'
from modelscope import snapshot_download
import sys
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2]))
PY
    python3 - "$MODEL_DIR/wcm/deploy.pt" <<'PY'
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
    cd "$ROOT/third_party/WCM"
    python3 -m pytest -q tests
    ;;
  *)
    echo "Usage: bash quickstart_wcm.sh {demo|verify|test}" >&2
    exit 2
    ;;
esac
