#!/usr/bin/env bash
# Evaluate the untrained/base Qwen3.5-9B model on fixed PCBA validation splits.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-9B"
PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
DATASET="PCBA/ablation/val_all.jsonl"
OUTPUT_DIR="output/pcba_eval"
RUN_TAG="qwen35_9b_base_$(date +%Y%m%d-%H%M%S)"
RESULT_PATH="${OUTPUT_DIR}/${RUN_TAG}_val_all.jsonl"
SUMMARY_PATH="${OUTPUT_DIR}/${RUN_TAG}_summary.json"

if [[ ! -d "${PCBA_ROOT}" && -d "${REPO_ROOT}/PCBA_Standard-to-Real_Challenge" ]]; then
  PCBA_ROOT="${REPO_ROOT}/PCBA_Standard-to-Real_Challenge"
fi

NEED_BUILD_DATASET=1
MANIFEST="PCBA/ablation/manifest.json"
if [[ -f "${DATASET}" && -f "${MANIFEST}" ]]; then
  MANIFEST_PCBA_ROOT="$(python3 -c 'import json; print(json.load(open("PCBA/ablation/manifest.json", encoding="utf-8")).get("pcba_root", ""))' 2>/dev/null || true)"
  if [[ "${MANIFEST_PCBA_ROOT}" == "${PCBA_ROOT}" ]]; then
    NEED_BUILD_DATASET=0
  fi
fi

if [[ "${NEED_BUILD_DATASET}" == "1" ]]; then
  PCBA_ROOT="${PCBA_ROOT}" python3 PCBA/build_pcba_ablation_splits.py
else
  echo "[info] reuse existing ${DATASET}"
fi
mkdir -p "${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=600000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift infer \
  --model "${MODEL}" \
  --infer_backend transformers \
  --val_dataset "${DATASET}" \
  --result_path "${RESULT_PATH}" \
  --max_batch_size 1 \
  --max_new_tokens 16 \
  --temperature 0 \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  --attn_impl sdpa

python3 PCBA/score_pcba_infer_result.py \
  "${RESULT_PATH}" \
  --source-jsonl "${DATASET}" \
  --summary-json "${SUMMARY_PATH}"

echo "[info] result_jsonl: ${RESULT_PATH}"
echo "[info] summary_json: ${SUMMARY_PATH}"
