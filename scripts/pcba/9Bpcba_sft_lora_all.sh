#!/usr/bin/env bash
# Qwen3.5-9B PCBA all-domain LoRA SFT on fixed ablation splits.
# 这个是最后做消融的时候用自己从训练集划分的固定的测试集和训练集来训练测试的

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-9B"
PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
DATASET="PCBA/ablation/train_all.jsonl"
VAL_DATASET="PCBA/ablation/val_all.jsonl"
OUTPUT_DIR="output/Qwen3.5-9B-pcba-ablation-lora-all"

if [[ ! -d "${PCBA_ROOT}" && -d "${REPO_ROOT}/PCBA_Standard-to-Real_Challenge" ]]; then
  PCBA_ROOT="${REPO_ROOT}/PCBA_Standard-to-Real_Challenge"
fi

PCBA_ROOT="${PCBA_ROOT}" python3 PCBA/build_pcba_ablation_splits.py
[[ -s "${DATASET}" ]] || { echo "[error] Missing or empty: ${DATASET}"; exit 1; }
[[ -s "${VAL_DATASET}" ]] || { echo "[error] Missing or empty: ${VAL_DATASET}"; exit 1; }

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=600000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
  --model "${MODEL}" \
  --tuner_type lora \
  --lora_rank 512 \
  --lora_alpha 512 \
  --target_modules all-linear \
  --freeze_vit false \
  --freeze_aligner false \
  --dataset "${DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --enable_thinking false \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --vit_lr 1e-5 \
  --aligner_lr 1e-5 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 100 \
  --save_total_limit 3 \
  --predict_with_generate true \
  --max_new_tokens 16 \
  --temperature 0 \
  --eval_metric acc \
  --acc_strategy seq \
  --metric_for_best_model seq_acc \
  --logging_steps 5 \
  --max_length 32000 \
  --warmup_ratio 0.03 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --attn_impl sdpa \
  --model_name swift-robot

echo "[info] output_dir: ${OUTPUT_DIR}"
