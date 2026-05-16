#!/usr/bin/env python3
"""
从 rag_top_pages.jsonl（RAG 选页）+ answer.json（analysis/answer/evidence）
组装 ms-swift 多模态 SFT 数据集。用户侧 prompt 与 pdf_qwen_test.py 一致（图文交错）；
assistant 目标为 <think> + <answer> + <evidence>。

用法:
  python3 build_pdf_rag_sft_dataset.py

环境变量:
  RAG_TOP_PAGES_JSONL  默认 rag/rag_top_pages.jsonl
  ANSWER_JSON          默认 rag/answer.json
  OUTPUT_TEST_DIR      MinerU 解析根目录，默认集群路径（见下）
  OUT_JSONL            默认 rag/pdf_rag_sft_train.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# 与 pdf_qwen_test / pdf_qwen_rag_top_pages 共用工具
from utils import (
    _content_list_dir,
    build_interleaved_content,
    content_items_for_pages,
    interleaved_preamble_ja,
    interleaved_preamble_vi,
    load_content_list_raw,
    prompt_answer_ja_interleaved,
    prompt_answer_vi_interleaved,
    system_msg_ja,
    system_msg_vi,
)

_DEFAULT_OUTPUT_TEST = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/lava/output_test"
)

RAG_TOP_PAGES_JSONL = os.environ.get("RAG_TOP_PAGES_JSONL", "rag_top_pages.jsonl")
ANSWER_JSON = os.environ.get("ANSWER_JSON", "answer.json")
OUTPUT_TEST_DIR = os.environ.get("OUTPUT_TEST_DIR", _DEFAULT_OUTPUT_TEST)
OUT_JSONL = os.environ.get("OUT_JSONL", "pdf_rag_sft_train.jsonl")


def _abspath_under_script(path: str, script_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def interleaved_parts_to_user_content(
    preamble: str,
    interleaved_parts: List[Dict[str, Any]],
    question_block: str,
) -> Tuple[str, List[str]]:
    """将图文交错 parts 转为 ms-swift user 字符串（<image> 占位）与 images 路径列表。"""
    chunks: List[str] = [preamble]
    images: List[str] = []
    for part in interleaved_parts:
        tp = part.get("type")
        if tp == "text":
            chunks.append(part.get("text", ""))
        elif tp == "image":
            img_path = part.get("image", "")
            if not img_path or not os.path.isfile(img_path):
                raise FileNotFoundError(f"图片不存在: {img_path}")
            chunks.append("<image>")
            images.append(os.path.abspath(img_path))
    chunks.append(question_block)
    return "".join(chunks), images


def format_assistant_target(rec: Dict[str, Any]) -> str:
    """拼接训练目标：redacted_thinking(analysis) + answer + evidence。"""
    analysis = (rec.get("analysis") or "").strip()
    answer = rec.get("answer", "")
    if isinstance(answer, (list, dict)):
        answer_str = json.dumps(answer, ensure_ascii=False)
    else:
        answer_str = str(answer).strip()

    evidence = rec.get("evidence", [])
    if isinstance(evidence, list):
        evidence_str = json.dumps(evidence, ensure_ascii=False)
    else:
        evidence_str = str(evidence).strip()

    return (
        f"<think>\n{analysis}\n</think>\n"
        f"<answer>{answer_str}</answer>\n"
        f"<evidence>{evidence_str}</evidence>"
    )


def build_one_sample(
    rag_row: Dict[str, Any],
    answer_rec: Dict[str, Any],
    md_root: str,
    cl_cache: Dict[str, List[Dict[str, Any]]],
    cl_dir_cache: Dict[str, str],
) -> Dict[str, Any]:
    fid = rag_row["file_id"]
    lang = (rag_row.get("language") or "ja").strip().lower()
    question = rag_row["question"]
    answer_format = (rag_row.get("answer_format") or "string").strip()
    page_indices = [int(p["page_idx"]) for p in rag_row["selected_pages"]]
    origin_pdf = (rag_row.get("origin_pdf") or "").strip() or None

    if fid not in cl_cache:
        cl_cache[fid] = load_content_list_raw(md_root, fid)
        cl_dir_cache[fid] = _content_list_dir(md_root, fid)

    items = content_items_for_pages(cl_cache[fid], page_indices)
    interleaved_parts, _image_paths, temp_pngs = build_interleaved_content(
        items,
        cl_dir_cache[fid],
        lang,
        origin_pdf=origin_pdf,
    )
    for p in temp_pngs:
        try:
            os.unlink(p)
        except OSError:
            pass

    if lang == "vi":
        preamble = interleaved_preamble_vi(page_indices)
        question_block = prompt_answer_vi_interleaved(question, answer_format, "")
        system = system_msg_vi()
    else:
        preamble = interleaved_preamble_ja(page_indices)
        question_block = prompt_answer_ja_interleaved(question, answer_format, "")
        system = system_msg_ja()

    user_content, images = interleaved_parts_to_user_content(
        preamble, interleaved_parts, question_block
    )
    assistant_content = format_assistant_target(answer_rec)

    return {
        "id": rag_row["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": images,
    }


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rag_path = _abspath_under_script(RAG_TOP_PAGES_JSONL, script_dir)
    answer_path = _abspath_under_script(ANSWER_JSON, script_dir)
    md_root = (
        OUTPUT_TEST_DIR
        if os.path.isabs(OUTPUT_TEST_DIR)
        else os.path.join(script_dir, OUTPUT_TEST_DIR)
    )
    out_path = _abspath_under_script(OUT_JSONL, script_dir)

    if not os.path.isdir(md_root):
        print(f"[error] OUTPUT_TEST_DIR 不存在: {md_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(rag_path):
        print(f"[error] 缺少 RAG 选页文件: {rag_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(answer_path):
        print(f"[error] 缺少答案文件: {answer_path}", file=sys.stderr)
        sys.exit(1)

    with open(answer_path, encoding="utf-8") as f:
        answers_list = json.load(f)
    answers_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in answers_list}

    cl_cache: Dict[str, List[Dict[str, Any]]] = {}
    cl_dir_cache: Dict[str, str] = {}
    n_ok = 0
    n_skip = 0
    errors: List[str] = []

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(rag_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rag_row = json.loads(line)
            qid = rag_row.get("id", "")
            answer_rec = answers_by_id.get(qid)
            if answer_rec is None:
                errors.append(f"行 {line_no} id={qid}: answer.json 中无对应记录")
                n_skip += 1
                continue
            if not (answer_rec.get("analysis") or "").strip():
                errors.append(f"行 {line_no} id={qid}: analysis 为空")
                n_skip += 1
                continue
            try:
                sample = build_one_sample(
                    rag_row, answer_rec, md_root, cl_cache, cl_dir_cache
                )
            except Exception as e:
                errors.append(f"行 {line_no} id={qid}: {e}")
                n_skip += 1
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"已写入 {n_ok} 条…")

    print(f"完成: 成功 {n_ok} 条, 跳过 {n_skip} 条 -> {out_path}")
    print(f"content_list 缓存文档数: {len(cl_cache)}")
    if errors:
        print(f"前 10 条错误:", file=sys.stderr)
        for msg in errors[:10]:
            print(f"  - {msg}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  … 共 {len(errors)} 条", file=sys.stderr)
        if n_ok == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
