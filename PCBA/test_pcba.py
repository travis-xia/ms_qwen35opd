#!/usr/bin/env python3
"""PCBA 测试集推理，生成 Codabench 提交文件 submission.csv。"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, List, Tuple

import torch.distributed as dist
from accelerate.utils import gather_object
from tqdm import tqdm

# ============ 按需修改 ============
PCBA_ROOT = (
    '/inspire/qb-ilm/project/traffic-congestion-management/'
    'xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge'
)
MODEL = '/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/ms_qwen35opd/output/Qwen3.5-27B-pcba-lora/v0-20260527-165421/checkpoint-400-merged'
RUN_SUFFIX = '27B0527-1654ckpt400'  # 每次新跑改这里，用于区分输出文件
OUTPUT = f'submission_{RUN_SUFFIX}.csv'
PREDICT_JSONL = f'output/pcba_test_predict_{RUN_SUFFIX}.jsonl'
MAX_NEW_TOKENS = 16
BATCH_SIZE = 1
ATTN_IMPL = 'sdpa'
TORCH_DTYPE = 'bfloat16'
LIMIT = None  # 调试时可设为 10
# =================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import iter_test_rows, normalize_answer


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _shard_items(items: List[Any], rank: int, world_size: int) -> List[Any]:
    if rank < 0 or world_size <= 1:
        return items
    if len(items) < world_size:
        return items[rank:] if rank < len(items) else []
    shard_size = (len(items) + world_size - 1) // world_size
    start = rank * shard_size
    end = min(start + shard_size, len(items))
    return items[start:end]


def _merge_gathered_predictions(gathered: List[Any]) -> List[Tuple[Any, str, str]]:
    """兼容 accelerate 新旧版本：新版已 flatten，旧版返回各 rank 的 list。"""
    if not gathered:
        return []
    if isinstance(gathered[0], tuple):
        return list(gathered)
    merged: List[Tuple[Any, str, str]] = []
    for shard in gathered:
        merged.extend(shard)
    return merged


def main() -> None:
    from swift.arguments import InferArguments
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.pipelines.utils import prepare_model_template
    from swift.utils import get_dist_setting, is_dist, is_master

    rank, _, world_size, _ = get_dist_setting()

    pcba_root = _resolve_path(PCBA_ROOT)
    model = _resolve_path(MODEL)
    output = _resolve_path(OUTPUT)
    predict_jsonl = _resolve_path(PREDICT_JSONL) if PREDICT_JSONL else None

    items = list(iter_test_rows(pcba_root))
    if LIMIT is not None:
        items = items[:LIMIT]
    if not items:
        raise RuntimeError(f'No test samples found under {pcba_root}')

    total = len(items)
    items = _shard_items(items, rank, world_size)
    if is_master():
        print(f'Loaded {total} test samples, world_size={world_size}, rank0 shard={len(items)}')

    args = InferArguments(
        model=model,
        load_args=True,
        torch_dtype=TORCH_DTYPE,
        attn_impl=ATTN_IMPL,
        stream=False,
        val_dataset=[pcba_root],
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    loaded_model, template = prepare_model_template(args)
    engine = TransformersEngine(loaded_model, template=template, max_batch_size=BATCH_SIZE)

    infer_requests = []
    for _, sample in items:
        kwargs = {'messages': sample['messages']}
        if sample.get('images'):
            kwargs['images'] = sample['images']
        infer_requests.append(InferRequest(**kwargs))

    request_config = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=0)
    predictions: List[Tuple[Any, str, str]] = []
    for start in tqdm(
            range(0, len(infer_requests), BATCH_SIZE),
            desc=f'infer[rank{max(rank, 0)}]',
            disable=False,
    ):
        batch_requests = infer_requests[start:start + BATCH_SIZE]
        batch_items = items[start:start + BATCH_SIZE]
        resp_list = engine.infer(batch_requests, request_config, use_tqdm=False)
        for (row, _), resp in zip(batch_items, resp_list):
            raw = resp.choices[0].message.content
            answer = normalize_answer(raw, row)
            predictions.append((row['qid'], answer, raw))

    if is_dist() and dist.is_initialized():
        predictions = _merge_gathered_predictions(gather_object(predictions))

    if not is_master():
        return

    predictions.sort(key=lambda x: x[0])

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'answer'])
        for qid, answer, _ in predictions:
            writer.writerow([qid, answer])

    if predict_jsonl:
        pred_dir = os.path.dirname(predict_jsonl)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        with open(predict_jsonl, 'w', encoding='utf-8') as f:
            for qid, answer, raw in predictions:
                f.write(json.dumps({'qid': qid, 'answer': answer, 'response': raw}, ensure_ascii=False) + '\n')

    print(f'Wrote {len(predictions)} rows to {output}')


if __name__ == '__main__':
    main()
