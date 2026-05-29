#!/usr/bin/env bash
# Qwen3.5-27B PCBA LoRA SFT — 仅 Standard 子集
# 用法: bash 27Bpcbasft_lora_standard.sh
# 跳过数据转换: BUILD_DATASET=0 bash 27Bpcbasft_lora_standard.sh
# 不合并 LoRA: MERGE_LORA_AFTER=0 bash 27Bpcbasft_lora_standard.sh

set -euo pipefail
cd "$(dirname "$0")"

PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
DATASET="PCBA/pcba_sft_train_standard.jsonl"
VAL_DATASET="PCBA/pcba_sft_val_standard.jsonl"
OUTPUT_DIR="output/Qwen3.5-27B-pcba-lora-standard"

if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
  (cd PCBA && PCBA_ROOT="${PCBA_ROOT}" \
    TRAIN_SPLITS="standard" \
    OUT_JSONL="pcba_sft_train_standard.jsonl" \
    OUT_VAL_JSONL="pcba_sft_val_standard.jsonl" \
    VAL_RATIO="${VAL_RATIO:-0.02}" \
    VAL_SEED="${VAL_SEED:-42}" \
    python3 build_pcba_task_sft_dataset.py)
fi

[[ -s "${DATASET}" ]] || { echo "[error] 缺少或为空: ${DATASET}"; exit 1; }
[[ -s "${VAL_DATASET}" ]] || { echo "[error] 缺少或为空: ${VAL_DATASET}"; exit 1; }

# Standard 最多 11 张图；单图像素与 PDF 任务保持一致
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=640000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
  --model /inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-27B \
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
  --eval_steps 30 \
  --save_steps 30 \
  --save_total_limit 3 \
  --predict_with_generate true \
  --max_new_tokens 16 \
  --temperature 0 \
  --eval_metric acc \
  --acc_strategy seq \
  --metric_for_best_model seq_acc \
  --logging_steps 5 \
  --max_length 16384 \
  --warmup_ratio 0.03 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --attn_impl sdpa \
  --model_name swift-robot

if [[ "${MERGE_LORA_AFTER:-1}" == "1" ]]; then
  RUN_DIR="$(find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'v*' 2>/dev/null | LC_ALL=C sort | tail -1)"
  LAST_CKPT=""
  if [[ -n "${RUN_DIR}" ]]; then
    LAST_CKPT="$(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' ! -name 'checkpoint-*-merged' 2>/dev/null | LC_ALL=C sort -V | tail -1)"
  fi
  if [[ -z "${LAST_CKPT}" ]]; then
    echo "[warn] 未找到 checkpoint-*，跳过合并。" >&2
  else
    OUT_MERGED="${LAST_CKPT}-merged"
    rm -rf "${OUT_MERGED}"
    CUDA_VISIBLE_DEVICES=0 swift export \
      --adapters "${LAST_CKPT}" \
      --merge_lora true \
      --output_dir "${OUT_MERGED}" \
      --exist_ok true
    echo "[info] merged LoRA -> ${OUT_MERGED}"
  fi
fi

rm -f "${DATASET}" "${VAL_DATASET}"
echo "[info] 已删除临时子集: ${DATASET}, ${VAL_DATASET}"
