#!/usr/bin/env bash
# LoFTR-style ScanNet-1500 relative pose AUC with SiLK (URR-aligned defaults in yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCANNET_ROOT="${SCANNET_ROOT:-${ROOT}/assets/datasets/my_scannet1500}"
TEST_NPZ="${TEST_NPZ:-${SCANNET_ROOT}/scannet_test_1500/test.npz}"
CKPT="${CKPT:-${ROOT}/checkpoints/pvgg-4.ckpt}"
DEVICE="${DEVICE:-cuda:0}"
MAX_PAIRS="${MAX_PAIRS:--1}"
# urr_center_crop | fixed_hw | max_edge
PREPROCESS="${PREPROCESS:-urr_center_crop}"

if [[ ! -f "$TEST_NPZ" ]]; then
  echo "Missing test.npz. Download with:"
  echo "  curl -L -o \"$TEST_NPZ\" \\"
  echo "    https://github.com/zju3dv/LoFTR/raw/refs/heads/master/assets/scannet_test_1500/test.npz"
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT"
  echo "Place pvgg-4.ckpt under checkpoints/ (or set CKPT=...)"
  exit 1
fi

EXTRA=()
if [[ "$PREPROCESS" == "fixed_hw" ]]; then
  EXTRA+=(
    "mode.image_preprocess.mode=fixed_hw"
    "mode.matcher.postprocessing=none"
  )
elif [[ "$PREPROCESS" == "urr_center_crop" ]]; then
  EXTRA+=(
    "mode.image_preprocess.mode=urr_center_crop"
    "mode.image_preprocess.img_dim=146"
  )
elif [[ "$PREPROCESS" == "max_edge" ]]; then
  EXTRA+=(
    "mode.image_preprocess.mode=max_edge"
    "mode.image_preprocess.max_edge=1200"
  )
else
  echo "Unknown PREPROCESS=${PREPROCESS} (use urr_center_crop | fixed_hw | max_edge)"
  exit 1
fi

./bin/silk-cli mode=run-scannet-pose-tests-silk \
  "mode.device=${DEVICE}" \
  "mode.dataset.dataset_root=${SCANNET_ROOT}" \
  "mode.dataset.index_path=${TEST_NPZ}" \
  "mode.dataset.max_pairs=${MAX_PAIRS}" \
  "mode.model.checkpoint_path=${CKPT}" \
  "${EXTRA[@]}"
