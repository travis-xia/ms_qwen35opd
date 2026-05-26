#!/usr/bin/env bash
# PCBA 测试集推理，自动使用所有可见 GPU 做 DDP 并行
set -euo pipefail
cd "$(dirname "$0")"

PCBA_ROOT="${PCBA_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge}"
MODEL="${MODEL:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/ms_qwen35opd/output/Qwen3.5-9B-pcba/v0-20260526-203149/checkpoint-384}"
OUTPUT="${OUTPUT:-submission.csv}"
PREDICT_JSONL="${PREDICT_JSONL:-output/pcba_test_predict.jsonl}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(python3 -c "import torch; print(max(torch.cuda.device_count(), 1))")"
fi

echo "[info] NPROC_PER_NODE=${NPROC_PER_NODE}"

run_test() {
  PCBA_ROOT="${PCBA_ROOT}" \
  MODEL="${MODEL}" \
  OUTPUT="${OUTPUT}" \
  PREDICT_JSONL="${PREDICT_JSONL}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MAX_PIXELS=600000 \
  MIN_PIXELS=3136 \
  VIDEO_MAX_TOKEN_NUM=128 \
  FPS_MAX_FRAMES=12 \
  "$@"
}

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  run_test NPROC_PER_NODE="${NPROC_PER_NODE}" \
    python3 -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" PCBA/test_pcba.py
else
  run_test python3 PCBA/test_pcba.py
fi
