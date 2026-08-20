#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" train.py \
  --method projected \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --target-global-batch-size "${TARGET_GLOBAL_BATCH_SIZE:-32}" \
  --batch-recipe "${BATCH_RECIPE:-coco_gb32_optimized}" \
  --precision "${PRECISION:-fp32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --performance-log-every "${PERFORMANCE_LOG_EVERY:-0}" \
  --seed "${SEED:-42}" \
  --resume "${RESUME:-auto}"
