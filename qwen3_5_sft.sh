#!/usr/bin/env bash
# Qwen3.5-4B 全参 SFT：PDF RAG 文档理解（图文交错 + analysis/answer/evidence）
# 参考: docs/source/BestPractices/Qwen3_5-Best-Practice.md
#       rag/pdf_qwen_test.py, rag/build_pdf_rag_sft_dataset.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="${ROOT_DIR}/rag"

MODEL="${MODEL:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Qwen3.5-4B-pdf-rag}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# MinerU 解析结果根目录（与 rag_top_pages.jsonl 中 origin_pdf 一致）
OUTPUT_TEST_DIR="${OUTPUT_TEST_DIR:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/lava/output_test}"
RAG_TOP_PAGES_JSONL="${RAG_TOP_PAGES_JSONL:-${RAG_DIR}/rag_top_pages.jsonl}"
ANSWER_JSON="${ANSWER_JSON:-${RAG_DIR}/answer.json}"
DATASET_JSONL="${DATASET_JSONL:-${RAG_DIR}/pdf_rag_sft_train.jsonl}"
BUILD_DATASET="${BUILD_DATASET:-1}"

if [[ "${BUILD_DATASET}" == "1" ]]; then
  echo "[dataset] 组装 SFT 数据 -> ${DATASET_JSONL}"
  cd "${RAG_DIR}"
  OUTPUT_TEST_DIR="${OUTPUT_TEST_DIR}" \
  RAG_TOP_PAGES_JSONL="${RAG_TOP_PAGES_JSONL}" \
  ANSWER_JSON="${ANSWER_JSON}" \
  OUT_JSONL="${DATASET_JSONL}" \
  python3 build_pdf_rag_sft_dataset.py
  cd "${ROOT_DIR}"
fi

if [[ ! -f "${DATASET_JSONL}" ]]; then
  echo "[error] 训练集不存在: ${DATASET_JSONL}（可设 BUILD_DATASET=1 或先运行 build_pdf_rag_sft_dataset.py）" >&2
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
    --model "${MODEL}" \
    --tuner_type full \
    --dataset "${DATASET_JSONL}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --attn_impl flash_attention_2 \
    --output_dir "${OUTPUT_DIR}" \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length "${MAX_LENGTH}" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --model_author swift \
    --model_name swift-robot
