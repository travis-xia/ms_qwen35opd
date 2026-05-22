#!/usr/bin/env python3
"""
从预计算的 rag_top_pages.jsonl 读取选页，跳过 Embedding/Reranker，
按 pdf_qwen_test.py 相同逻辑组装图文交错 prompt 并用 vLLM 推理。

与 pdf_qwen_rag_top_pages.py + 本脚本 的分工：
  pdf_qwen_rag_top_pages.py  -> rag_top_pages.jsonl
  pdf_qwen_infer_from_rag.py -> submission / 详细 JSON（本文件）

不修改 pdf_qwen_test.py / build_pdf_rag_sft_dataset.py 等既有文件。

依赖：transformers、torch、vllm、PyMuPDF（fitz）

运行示例:
  cd rag
  MODEL_PATH=/path/to/Qwen3.5-4B \\
  OUTPUT_TEST_DIR=/path/to/output_test \\
  RAG_PAGES_JSONL=rag_top_pages.jsonl \\
  python3 pdf_qwen_infer_from_rag.py

可选:
  QUESTIONS_CSV=test.csv   # 按 CSV 顺序跑题（需与 jsonl id 对齐）
  MAX_SAMPLES=10           # 只跑前 N 题（调试）
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from utils import (
    _content_list_dir,
    apply_generation_prompt_with_brief_thinking,
    apply_generation_prompt_without_thinking,
    build_interleaved_content,
    build_interleaved_messages,
    build_messages,
    content_items_for_pages,
    interleaved_preamble_ja,
    interleaved_preamble_vi,
    load_content_list_raw,
    parse_answer_tag,
    parse_evidence,
    parse_system_msg,
    parse_user_msg,
    prepare_mm_data,
    prompt_answer_ja_interleaved,
    prompt_answer_vi_interleaved,
    release_torch_memory,
    set_random_seed,
    system_msg_ja,
    system_msg_vi,
)

_DEFAULT_OUTPUT_TEST = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/lava/output_test"
)
_DEFAULT_MODEL = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/hf_download/Qwen3.5-4B"
)
_DEFAULT_QUESTIONS_CSV = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/lava/test.csv"
)

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
OUTPUT_TEST_DIR = os.environ.get("OUTPUT_TEST_DIR", _DEFAULT_OUTPUT_TEST)
RAG_PAGES_JSONL = os.environ.get("RAG_PAGES_JSONL", "rag_top_pages.jsonl")
QUESTIONS_CSV = os.environ.get(
    "QUESTIONS_CSV",
    os.environ.get("TEST_CSV", _DEFAULT_QUESTIONS_CSV),
)
SUBMISSION_CSV = os.environ.get("SUBMISSION_CSV", "submission_from_rag.csv")
OUT_JSON = os.environ.get("OUT_JSON", "pdf_rag_infer_pred.json")
SEED = int(os.environ.get("SEED", "42"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32000"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
LIMIT_MM_IMAGES_PER_PROMPT = int(os.environ.get("LIMIT_MM_IMAGES_PER_PROMPT", "36"))
LIMIT_MM_PER_PROMPT = {"image": LIMIT_MM_IMAGES_PER_PROMPT, "video": 0}

SAMPLING_ANS = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.15,
    presence_penalty=0.0,
    max_tokens=12000,
    stop_token_ids=[],
    seed=SEED,
)
SAMPLING_PARSE = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    max_tokens=128,
    stop_token_ids=[],
    seed=SEED,
)
SAMPLING_LIST_FIX = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    max_tokens=int(os.environ.get("LIST_FIX_MAX_TOKENS", "512")),
    stop_token_ids=[],
    seed=SEED,
)


# ---------- 与 pdf_qwen_test.py 一致的后处理（复制，避免 import 该模块）----------


def format_evidence_column(pages: List[int]) -> str:
    if not pages:
        return "[]"
    return "[" + ",".join(str(p) for p in pages) + "]"


def try_parse_list(s: str) -> Optional[List[Any]]:
    text = (s or "").strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return None


def no_answer_text(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang == "ja":
        return "回答なし"
    if lang == "vi":
        return "Không có câu trả lời"
    if lang.startswith("zh"):
        return "没有答案"
    if lang == "en":
        return "No answer"
    return "No answer"


def is_none_placeholder(s: str) -> bool:
    return (s or "").strip().lower() == "none"


def normalize_string_items(value: List[Any], language: str) -> List[str]:
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if is_none_placeholder(text):
            continue
        if text:
            out.append(text)
    return out or [no_answer_text(language)]


def dump_list(value: List[Any], language: str) -> str:
    return json.dumps(normalize_string_items(value, language), ensure_ascii=False)


def list_fix_system_msg() -> str:
    return (
        "你是一个 CSV 提交答案格式规范化助手。"
        "你的任务不是重新答题，而是把已有 raw_answer 解析成符合 answer_format 的列表。"
        "必须只输出一个 <answer>...</answer> 标签。"
        "<answer> 内必须是严格 JSON array，可被 Python json.loads 解析。"
        "数组元素必须是字符串；不要输出 Markdown、解释、代码块或其他标签。"
    )


def list_fix_user_msg(question: str, answer_format: str, raw_answer: str) -> str:
    order_rule = (
        "ordered_list 要保持 raw_answer 中能推断出的顺序。"
        if answer_format == "ordered_list"
        else "unordered_list 不要求顺序，但每个独立答案项应单独成为数组元素。"
    )
    return (
        "请把 raw_answer 规范成提交 CSV 的 answer 字段。\n"
        "要求：\n"
        "1) 只抽取最终答案项，不要保留解释性句子。\n"
        f"2) {order_rule}\n"
        "3) 如果 raw_answer 明显只有一个答案项，输出单元素数组。\n"
        "4) 如果 raw_answer 用顿号、逗号、换行编号等列出多个答案项，请拆成多个字符串元素。\n"
        "5) 不确定时不要编造新答案，只根据 raw_answer 保守整理。\n"
        "6) 输出格式必须严格如下：<answer>[\"項目1\", \"項目2\"]</answer>\n\n"
        f"<answer_format>{answer_format}</answer_format>\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<raw_answer>\n{raw_answer}\n</raw_answer>"
    )


def extract_answer_text(model_output: str) -> str:
    text = (model_output or "").strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    if match:
        return match.group(1).strip()
    direct = text.strip()
    if direct.startswith("[") and direct.endswith("]"):
        return direct
    match = re.search(r"(\[[\s\S]*\])", direct)
    if match:
        return match.group(1).strip()
    return direct


def parse_model_list(model_output: str) -> Optional[List[str]]:
    payload = extract_answer_text(model_output)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        parsed = try_parse_list(payload)
        if parsed is None:
            return None
        value = parsed
    if not isinstance(value, list):
        return None
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if is_none_placeholder(s):
            continue
        if s:
            out.append(s)
    return out


def fallback_list(raw_answer: str, language: str) -> List[str]:
    text = (raw_answer or "").strip()
    if is_none_placeholder(text):
        return [no_answer_text(language)]
    return [text] if text else [no_answer_text(language)]


def sanitize_scalar_answer(answer: str, answer_format: str, language: str) -> str:
    text = re.sub(r"</?answer>", "", answer or "", flags=re.I).strip()
    text = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    fallback = no_answer_text(language)
    if answer_format in ("unordered_list", "ordered_list"):
        return text
    s = text.strip()
    if not s:
        return fallback
    if is_none_placeholder(s):
        return fallback
    if s == "[]":
        return fallback
    if s.startswith("[") and s.endswith("]"):
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            inner = s[1:-1].strip()
            return inner or fallback
        if isinstance(value, (list, tuple)):
            items = normalize_string_items(list(value), language)
            if len(items) == 1 and items[0] == fallback:
                return fallback
            return "、".join(items)
    return str(s)


# ---------- 数据加载 ----------


def _resolve(path: str, script_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def load_rag_by_id(rag_path: str) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(rag_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["id"]] = rec
    return by_id


def load_rows(script_dir: str) -> List[Dict[str, Any]]:
    rag_path = _resolve(RAG_PAGES_JSONL, script_dir)
    rag_by_id = load_rag_by_id(rag_path)
    csv_path = _resolve(QUESTIONS_CSV, script_dir)

    if os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        rows: List[Dict[str, Any]] = []
        for cr in csv_rows:
            qid = cr["id"]
            if qid not in rag_by_id:
                raise KeyError(f"rag_top_pages.jsonl 缺少 id={qid}")
            rag = rag_by_id[qid]
            rows.append(
                {
                    "id": qid,
                    "file_id": cr.get("file_id") or rag["file_id"],
                    "question": cr.get("question") or rag["question"],
                    "language": (cr.get("language") or rag.get("language") or "ja").strip(),
                    "answer_format": (
                        cr.get("answer_format") or rag.get("answer_format") or "string"
                    ).strip(),
                    "selected_pages": rag["selected_pages"],
                    "origin_pdf": rag.get("origin_pdf") or "",
                }
            )
    else:
        print(f"[warn] 未找到 {csv_path}，按 jsonl 文件顺序跑题。")
        rows = []
        with open(rag_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rag = json.loads(line)
                rows.append(
                    {
                        "id": rag["id"],
                        "file_id": rag["file_id"],
                        "question": rag["question"],
                        "language": (rag.get("language") or "ja").strip(),
                        "answer_format": (rag.get("answer_format") or "string").strip(),
                        "selected_pages": rag["selected_pages"],
                        "origin_pdf": rag.get("origin_pdf") or "",
                    }
                )

    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    return rows


def truncate_interleaved_images(
    interleaved_parts: List[Dict[str, Any]],
    image_paths: List[str],
    limit: int,
) -> tuple[List[Dict[str, Any]], List[str]]:
    if len(image_paths) <= limit:
        return interleaved_parts, image_paths
    keep_imgs = set(image_paths[:limit])
    parts = [
        p
        for p in interleaved_parts
        if p["type"] != "image" or p.get("image") in keep_imgs
    ]
    return parts, image_paths[:limit]


def build_llm_input_for_row(
    row: Dict[str, Any],
    cl_cache: Dict[str, List[Dict[str, Any]]],
    cl_dir_cache: Dict[str, str],
    md_root: str,
) -> tuple[Dict[str, Any], str, List[Dict[str, Any]], List[str]]:
    """返回 (vllm_input, prompt_str, selected_pages_meta, temp_pngs)。"""
    lang = (row.get("language") or "ja").strip().lower()
    q = row["question"]
    fid = row["file_id"]
    afmt = row.get("answer_format", "string")
    selected_pages = row["selected_pages"]
    if not selected_pages:
        raise ValueError(f"无选页: id={row.get('id')} file_id={fid}")

    page_indices = [int(p["page_idx"]) for p in selected_pages]
    selected_pages_meta = [
        {
            "page_idx": int(p["page_idx"]),
            "page_num": int(p.get("page_num", int(p["page_idx"]) + 1)),
            "rag_score": float(p.get("rag_score", 0.0)),
        }
        for p in selected_pages
    ]
    origin_pdf = (row.get("origin_pdf") or "").strip() or None

    if fid not in cl_cache:
        cl_cache[fid] = load_content_list_raw(md_root, fid)
        cl_dir_cache[fid] = _content_list_dir(md_root, fid)

    items = content_items_for_pages(cl_cache[fid], page_indices)
    interleaved_parts, image_paths, temp_pngs = build_interleaved_content(
        items,
        cl_dir_cache[fid],
        lang,
        origin_pdf=origin_pdf,
    )

    if len(image_paths) > LIMIT_MM_IMAGES_PER_PROMPT:
        print(
            f"[warn] id={row.get('id')} 图文交错 {len(image_paths)} 张图，"
            f"截断为 LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT}"
        )
        interleaved_parts, image_paths = truncate_interleaved_images(
            interleaved_parts,
            image_paths,
            LIMIT_MM_IMAGES_PER_PROMPT,
        )

    if lang == "vi":
        preamble = interleaved_preamble_vi(page_indices)
        question_block = prompt_answer_vi_interleaved(q, afmt, "")
        sys_msg = system_msg_vi()
    else:
        preamble = interleaved_preamble_ja(page_indices)
        question_block = prompt_answer_ja_interleaved(q, afmt, "")
        sys_msg = system_msg_ja()

    msgs = build_interleaved_messages(sys_msg, preamble, interleaved_parts, question_block)
    return msgs, image_paths, selected_pages_meta, temp_pngs


def main() -> None:
    t0 = time.perf_counter()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_root = _resolve(OUTPUT_TEST_DIR, script_dir)

    print(f"[开始] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    set_random_seed(SEED)
    print(f"SEED={SEED} | MODEL_PATH={MODEL_PATH}")
    print(f"OUTPUT_TEST_DIR={md_root}")
    print(f"RAG_PAGES_JSONL={_resolve(RAG_PAGES_JSONL, script_dir)}")

    if not os.path.isdir(md_root):
        raise FileNotFoundError(f"OUTPUT_TEST_DIR 不存在: {md_root}")

    rows = load_rows(script_dir)
    print(f"待推理 {len(rows)} 题（MAX_SAMPLES={MAX_SAMPLES or '全部'}）")

    unique_fids = sorted({r["file_id"] for r in rows})
    cl_cache: Dict[str, List[Dict[str, Any]]] = {}
    cl_dir_cache: Dict[str, str] = {}
    t_cl = time.perf_counter()
    for fid in unique_fids:
        cl_cache[fid] = load_content_list_raw(md_root, fid)
        cl_dir_cache[fid] = _content_list_dir(md_root, fid)
    print(f"[timing] 加载 content_list {len(unique_fids)} 篇: {time.perf_counter() - t_cl:.3f}s")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    t_llm = time.perf_counter()
    print(
        f"加载 VLM: {MODEL_PATH} "
        f"(LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT})"
    )
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        seed=SEED,
    )
    print(f"[timing] LLM 加载: {time.perf_counter() - t_llm:.3f}s")

    ans_inputs: List[Dict[str, Any]] = []
    gen_prompts: List[str] = []
    selected_pages_list: List[List[Dict[str, Any]]] = []
    all_temp_png: List[str] = []

    t_prep = time.perf_counter()
    for row in rows:
        msgs, image_paths, selected_pages_meta, temp_pngs = build_llm_input_for_row(
            row, cl_cache, cl_dir_cache, md_root
        )
        all_temp_png.extend(temp_pngs)
        selected_pages_list.append(selected_pages_meta)
        prompt = apply_generation_prompt_with_brief_thinking(processor, msgs)
        gen_prompts.append(prompt)
        llm_in: Dict[str, Any] = {"prompt": prompt}
        mm = prepare_mm_data(msgs, image_paths)
        if mm:
            llm_in["multi_modal_data"] = mm
        ans_inputs.append(llm_in)

    print(
        f"[timing] 组装 prompt {len(rows)} 条: {time.perf_counter() - t_prep:.3f}s"
    )

    print(f"回答生成 {len(ans_inputs)} 条…")
    t_ans = time.perf_counter()
    ans_outs = llm.generate(ans_inputs, sampling_params=SAMPLING_ANS)
    print(f"[timing] llm.generate 回答: {time.perf_counter() - t_ans:.3f}s")

    parse_inputs: List[Dict[str, Any]] = []
    for row, o in zip(rows, ans_outs):
        parse_msgs = build_messages(
            parse_system_msg(),
            parse_user_msg(row["question"], o.outputs[0].text),
            image_paths=None,
        )
        parse_prompt = apply_generation_prompt_without_thinking(processor, parse_msgs)
        parse_inputs.append({"prompt": parse_prompt})

    print(f"解析生成 {len(parse_inputs)} 条…")
    t_parse = time.perf_counter()
    parse_outs = llm.generate(parse_inputs, sampling_params=SAMPLING_PARSE)
    print(f"[timing] llm.generate 解析: {time.perf_counter() - t_parse:.3f}s")

    list_fix_by_row: Dict[int, str] = {}
    list_fix_raw_by_row: Dict[int, str] = {}
    list_fix_jobs: List[Dict[str, Any]] = []
    for row_idx, (row, parsed_out) in enumerate(zip(rows, parse_outs)):
        answer_format = (row.get("answer_format") or "string").strip()
        language = (row.get("language") or "").strip()
        if answer_format not in ("unordered_list", "ordered_list"):
            continue
        raw_answer = parse_answer_tag(parsed_out.outputs[0].text)
        if is_none_placeholder(raw_answer):
            list_fix_by_row[row_idx] = dump_list([], language)
            continue
        parsed_list = try_parse_list(raw_answer)
        if parsed_list is not None:
            list_fix_by_row[row_idx] = dump_list(parsed_list, language)
            continue
        list_fix_jobs.append(
            {
                "row_idx": row_idx,
                "question": row["question"],
                "answer_format": answer_format,
                "language": language,
                "raw_answer": raw_answer,
            }
        )

    if list_fix_jobs:
        print(f"列表题格式修复 {len(list_fix_jobs)} 条…")
        list_fix_inputs: List[Dict[str, Any]] = []
        for job in list_fix_jobs:
            list_fix_msgs = build_messages(
                list_fix_system_msg(),
                list_fix_user_msg(
                    job["question"], job["answer_format"], job["raw_answer"]
                ),
                image_paths=None,
            )
            list_fix_prompt = apply_generation_prompt_without_thinking(
                processor, list_fix_msgs
            )
            list_fix_inputs.append({"prompt": list_fix_prompt})
        t_lf = time.perf_counter()
        list_fix_outs = llm.generate(list_fix_inputs, sampling_params=SAMPLING_LIST_FIX)
        print(f"[timing] llm.generate 列表修复: {time.perf_counter() - t_lf:.3f}s")
        for job, out in zip(list_fix_jobs, list_fix_outs):
            raw_model_output = out.outputs[0].text
            list_fix_raw_by_row[job["row_idx"]] = raw_model_output
            items = parse_model_list(raw_model_output)
            if items is None:
                items = fallback_list(job["raw_answer"], job["language"])
            list_fix_by_row[job["row_idx"]] = dump_list(items, job["language"])

    del llm
    del processor
    del cl_cache, cl_dir_cache
    release_torch_memory()

    for p in all_temp_png:
        try:
            os.unlink(p)
        except OSError:
            pass

    results: List[Dict[str, Any]] = []
    submission_rows: List[Dict[str, str]] = []

    for row_idx, (row, o, p, model_input, voted_pages) in enumerate(
        zip(rows, ans_outs, parse_outs, gen_prompts, selected_pages_list)
    ):
        raw = o.outputs[0].text
        parsed = p.outputs[0].text
        ans = parse_answer_tag(parsed)
        row_lang = (row.get("language") or "").strip()
        row_answer_format = (row.get("answer_format") or "string").strip()
        if row_answer_format in ("unordered_list", "ordered_list"):
            ans = list_fix_by_row.get(
                row_idx,
                dump_list(fallback_list(ans, row_lang), row_lang),
            )
        else:
            ans = sanitize_scalar_answer(ans, row_answer_format, row_lang)
        evidence = parse_evidence(parsed)
        if not evidence and voted_pages:
            evidence = [int(x["page_num"]) for x in voted_pages]

        results.append(
            {
                "id": row["id"],
                "file_id": row["file_id"],
                "question": row["question"],
                "language": row_lang,
                "answer_format": row_answer_format,
                "input": model_input,
                "voted_pages": voted_pages,
                "answer": ans,
                "evidence": evidence,
                "raw_answer": raw,
                "raw_parse": parsed,
                "raw_list_fix": list_fix_raw_by_row.get(row_idx, ""),
            }
        )
        submission_rows.append(
            {
                "id": row["id"],
                "answer": ans,
                "evidence_page_number": format_evidence_column(evidence),
            }
        )

    out_path = _resolve(OUT_JSON, script_dir)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    print(f"已写入: {out_path}")

    sub_path = _resolve(SUBMISSION_CSV, script_dir)
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidence_page_number"])
        w.writeheader()
        w.writerows(submission_rows)
    print(f"已写入提交表: {sub_path}")

    elapsed = time.perf_counter() - t0
    print(f"[结束] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"[总耗时] {elapsed:.3f}s（{elapsed / 60.0:.2f} 分钟）")


if __name__ == "__main__":
    main()
