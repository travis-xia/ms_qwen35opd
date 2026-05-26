#!/usr/bin/env bash
# Qwen3.5-4B PDF-RAG 全参 SFT；跳过数据组装: BUILD_DATASET=0 bash qwen3_5_sft.sh
# 推理时 MODEL_PATH 直接指向 output/.../checkpoint-* 即可，无需 merge LoRA

set -euo pipefail
cd "$(dirname "$0")"

DATASET="rag/pdf_rag_gemini_cot_sft.jsonl"
OUTPUT_DIR="output/Qwen3.5-4B-pdf"

# if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
#   (cd rag && OUTPUT_TEST_DIR=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/lava/output_test \
#     OUT_JSONL="../${DATASET}" python3 build_pdf_rag_sft_dataset.py)
# fi
[[ -f "${DATASET}" ]] || { echo "[error] 缺少 ${DATASET}"; exit 1; }

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
  --tuner_type full \
  --freeze_vit false \
  --freeze_aligner false \
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
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps 39 \
  --save_steps 39 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --max_length 32000 \
  --warmup_ratio 0 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --attn_impl sdpa \
  --model_name swift-robot
