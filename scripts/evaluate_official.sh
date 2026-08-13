#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${GPU_IDS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  evaluate_official.py \
  "${OFFICIAL_CHECKPOINT:-${PROJECT_ROOT}/r50_deformable_detr-checkpoint.pth}" \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-dir "${OFFICIAL_EVAL_OUTPUT:-${OUTPUT_ROOT:-/workspace/artifacts}/official_checkpoint_eval}" \
  --torch-cache "${TORCH_CACHE:-${TORCH_HOME:-/workspace/torch-cache}}" \
  --batch-size "${EVAL_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-8}"
