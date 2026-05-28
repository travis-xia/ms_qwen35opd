#!/usr/bin/env bash
# Qwen3.5-27B PCBA LoRA SFT — 仅 Standard 子集（从全量 jsonl 过滤，不改 Python）
# 用法: bash 27Bpcbasft_lora_standard.sh
# 跳过全量数据构建（需已有 pcba_sft_train.jsonl）: BUILD_DATASET=0 bash 27Bpcbasft_lora_standard.sh
# 不合并 LoRA: MERGE_LORA_AFTER=0 bash 27Bpcbasft_lora_standard.sh

set -euo pipefail
cd "$(dirname "$0")"

PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
FULL_TRAIN="PCBA/pcba_sft_train.jsonl"
FULL_VAL="PCBA/pcba_sft_val.jsonl"
DATASET="PCBA/pcba_sft_train_standard.jsonl"
VAL_DATASET="PCBA/pcba_sft_val_standard.jsonl"
OUTPUT_DIR="output/Qwen3.5-27B-pcba-lora-standard"

if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
  (cd PCBA && PCBA_ROOT="${PCBA_ROOT}" \
    OUT_JSONL="pcba_sft_train.jsonl" \
    OUT_VAL_JSONL="pcba_sft_val.jsonl" \
    VAL_RATIO="0.02" \
    python3 build_pcba_sft_dataset.py)
fi

[[ -f "${FULL_TRAIN}" ]] || { echo "[error] 缺少 ${FULL_TRAIN}，请先构建全量数据集"; exit 1; }
[[ -f "${FULL_VAL}" ]] || { echo "[error] 缺少 ${FULL_VAL}，请先构建全量数据集"; exit 1; }

grep -F '"standard-' "${FULL_TRAIN}" > "${DATASET}"
grep -F '"standard-' "${FULL_VAL}" > "${VAL_DATASET}"
echo "[info] Standard 子集: $(wc -l < "${DATASET}") train + $(wc -l < "${VAL_DATASET}") val"

[[ -s "${DATASET}" ]] || { echo "[error] ${DATASET} 为空，请检查全量数据是否含 standard 样本"; exit 1; }
[[ -s "${VAL_DATASET}" ]] || { echo "[error] ${VAL_DATASET} 为空，请检查全量 val 是否含 standard 样本"; exit 1; }

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
  --eval_steps 100 \
  --save_steps 100 \
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

rm -f "${DATASET}" "${VAL_DATASET}"
echo "[info] 已删除临时子集: ${DATASET}, ${VAL_DATASET}"
