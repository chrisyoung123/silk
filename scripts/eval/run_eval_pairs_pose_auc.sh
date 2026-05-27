#!/usr/bin/env bash
# 用法：设置 PAIRS_JSON、MATCHES_NPZ、CAMERA_JSON、EXTRINSIC_JSON 后执行；其余参数透传给 Python。
# 示例：
#   export PAIRS_JSON=/path/to/pairs.json
#   export MATCHES_NPZ=/path/to/all_pairs_merged.npz
#   export CAMERA_JSON=/path/to/camera.json
#   export EXTRINSIC_JSON=/path/to/extrinsic.json
#   ./scripts/eval/run_eval_pairs_pose_auc.sh --theta-degrees --json-out /tmp/metrics.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

: "${PAIRS_JSON:?请设置环境变量 PAIRS_JSON（pairs JSON 路径）}"
: "${MATCHES_NPZ:?请设置环境变量 MATCHES_NPZ（all_pairs_merged.npz 路径）}"
: "${CAMERA_JSON:?请设置环境变量 CAMERA_JSON（相机内参 JSON 路径）}"
: "${EXTRINSIC_JSON:?请设置环境变量 EXTRINSIC_JSON（相机外参 JSON 路径）}"

exec python "${ROOT}/scripts/eval/eval_pairs_pose_auc.py" \
  --pairs-json "${PAIRS_JSON}" \
  --matches-npz "${MATCHES_NPZ}" \
  --camera-json "${CAMERA_JSON}" \
  --extrinsic-json "${EXTRINSIC_JSON}" \
  "$@"
