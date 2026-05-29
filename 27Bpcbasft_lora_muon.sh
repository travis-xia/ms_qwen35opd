#!/usr/bin/env bash
# Qwen3.5-27B PCBA LoRA SFT + MuonClip（MS-SWIFT 内置，无需 pip / 无需 clone Moonlight）
# 基于 27Bpcbasft_lora.sh；去掉 vit_lr/aligner_lr（与 multimodal 分组 LR 互斥）
#
# 跳过数据组装: BUILD_DATASET=0 bash 27Bpcbasft_lora_muon.sh
# 不合并 checkpoint: MERGE_LORA_AFTER=0 bash 27Bpcbasft_lora_muon.sh

set -euo pipefail
cd "$(dirname "$0")"

PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
DATASET="PCBA/pcba_sft_train.jsonl"
VAL_DATASET="PCBA/pcba_sft_val.jsonl"
OUTPUT_DIR="output/Qwen3.5-27B-pcba-lora-muonclip"

if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
  (cd PCBA && PCBA_ROOT="${PCBA_ROOT}" \
    OUT_JSONL="pcba_sft_train.jsonl" \
    OUT_VAL_JSONL="pcba_sft_val.jsonl" \
    VAL_RATIO="0.02" \
    EXTRA_SFT_JSONLS="../PCBA/ipc610g_standard_qa_mineru/ipc610g_standard_qa_sft.jsonl" \
    python3 build_pcba_sft_dataset.py)
fi
[[ -f "${DATASET}" ]] || { echo "[error] 缺少 ${DATASET}"; exit 1; }
[[ -f "${VAL_DATASET}" ]] || { echo "[error] 缺少 ${VAL_DATASET}"; exit 1; }

# Standard 最多 11 张图，RealWorld 单图
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
  --optimizer muonclip \
  --optim_args "qk_clip_tau=10000" \
  --dataset "${DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --enable_thinking false \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 100 \
  --save_total_limit 4 \
  --predict_with_generate true \
  --max_new_tokens 32 \
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
