#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${GPU_IDS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-smoke_projected_fp16_gb16}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${OUTPUT_ROOT:-/workspace/artifacts}/smoke}"
SMOKE_CHECKPOINT="${SMOKE_OUTPUT_ROOT}/projected/seed_${SEED:-42}/${SMOKE_RUN_NAME}/checkpoints/latest.pt"

run_smoke() {
  torchrun --standalone --nproc_per_node=2 train.py \
    --method projected \
    --data-root "${COCO_ROOT:-/workspace/data/coco}" \
    --output-root "${SMOKE_OUTPUT_ROOT}" \
    --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
    --epochs 1 \
    --batch-size 4 \
    --batch-recipe coco_gb16_reference \
    --target-global-batch-size 16 \
    --precision fp16 \
    --train-limit "${SMOKE_TRAIN_LIMIT:-32}" \
    --val-limit "${SMOKE_VAL_LIMIT:-16}" \
    --num-workers "${SMOKE_NUM_WORKERS:-2}" \
    --gradient-log-every 1 \
    --performance-log-every 1 \
    --save-every 1 \
    --run-name "${SMOKE_RUN_NAME}" \
    --seed "${SEED:-42}" \
    --resume auto
}

run_smoke
if [[ "${SMOKE_RESUME_CHECK:-1}" == "1" ]]; then
  run_smoke
fi

python scripts/validate_projected_amp_smoke.py \
  "${SMOKE_CHECKPOINT}" \
  --expected-world-size 2 \
  --output "${SMOKE_OUTPUT_ROOT}/projected/seed_${SEED:-42}/${SMOKE_RUN_NAME}/smoke_report.json"
