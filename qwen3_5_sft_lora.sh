#!/usr/bin/env bash
# Qwen3.5-4B PDF-RAG SFT（LoRA）；跳过数据组装: BUILD_DATASET=0 bash qwen3_5_sft.sh
# 不合并、只保留 LoRA checkpoint: MERGE_LORA_AFTER=0 bash qwen3_5_sft.sh

set -euo pipefail
cd "$(dirname "$0")"

DATASET="rag/pdf_rag_gemini_cot_sft.jsonl"
OUTPUT_DIR="output/Qwen3.5-4B-pdf-rag"

# if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
#   (cd rag && OUTPUT_TEST_DIR=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/lava/output_test \
#     OUT_JSONL="../${DATASET}" python3 build_pdf_rag_sft_dataset.py)
# fi
# [[ -f "${DATASET}" ]] || { echo "[error] 缺少 ${DATASET}"; exit 1; }

# 与 pdf_qwen_test: MAX_MODEL_LEN=32000, vLLM dtype=bfloat16, enable_thinking 作答
# 与 utils: CROP_IMAGE_MAX/MIN_PIXELS=360000/3136, INTERLEAVED_PAGE_BLOCK_LIMIT=10
# 与 pdf_qwen_test: LIMIT_MM_IMAGES_PER_PROMPT=36（训练无同等参数，数据侧由 build_interleaved 产生）
# ,4,5,6,7
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=360000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
  --model /inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-4B \
  --tuner_type lora \
  --lora_rank 1024 \
  --lora_alpha 1024 \
  --target_modules all-linear \
  --dataset "${DATASET}" \
  --val_dataset "${DATASET}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-4 \
  --weight_decay 0.00 \
  --gradient_accumulation_steps 8 \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --max_length 32000 \
  --warmup_ratio 0 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --attn_impl sdpa \
  --model_name swift-robot

if [[ "${MERGE_LORA_AFTER:-1}" == "1" ]]; then
  RUN_DIR="$(find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'v*' 2>/dev/null | LC_ALL=C sort | tail -1)"
  LAST_CKPT=""
  if [[ -n "${RUN_DIR}" ]]; then
    LAST_CKPT="$(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' 2>/dev/null | LC_ALL=C sort -V | tail -1)"
  fi
  if [[ -z "${LAST_CKPT}" ]]; then
    echo "[warn] 未找到 checkpoint-*，跳过合并。" >&2
  else
    OUT_FULL="${LAST_CKPT}.__hf__"
    rm -rf "${OUT_FULL}"
    CUDA_VISIBLE_DEVICES=0 swift export \
      --adapters "${LAST_CKPT}" \
      --merge_lora true \
      --output_dir "${OUT_FULL}" \
      --exist_ok true
    rm -rf "${LAST_CKPT}"
    mv "${OUT_FULL}" "${LAST_CKPT}"
  fi
fi
