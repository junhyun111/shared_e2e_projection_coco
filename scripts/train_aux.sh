#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" train.py \
  --method aux_only \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --target-global-batch-size "${TARGET_GLOBAL_BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --seed "${SEED:-42}" \
  --resume "${RESUME:-auto}"
