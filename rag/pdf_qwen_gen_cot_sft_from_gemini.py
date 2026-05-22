#!/usr/bin/env python3
"""
用 vLLM 根据「RAG Top-K 页图文交错 + Gemini 参考答案/证据」生成 CoT，并组装 ms-swift SFT jsonl。

数据规则（与线上一致）：
  - user：与 pdf_qwen_infer / pdf_qwen_test 相同（不含教师参考答案）。
  - assistant：<think> 由模型生成；<answer>/<evidence> 来自 Gemini CSV。
  - answer 经 normalize_submission_answer 与推理提交格式对齐。
  - evidence：ref∩RAG 可见页；交集为空则用 Gemini 标签中的完整证据页（与答案一并记忆）。

依赖：transformers、torch、vllm；在 rag 目录下执行。

示例:
  cd rag
  MODEL_PATH=... OUTPUT_TEST_DIR=... GEMINI_CSV=gemini1.csv \\
  python3 pdf_qwen_gen_cot_sft_from_gemini.py
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from build_pdf_rag_sft_dataset import (
    format_assistant_target,
    interleaved_parts_to_user_content,
    persist_fullpage_images,
)
from utils import (
    _content_list_dir,
    apply_generation_prompt_without_thinking,
    build_interleaved_content,
    build_interleaved_messages,
    content_items_for_pages,
    interleaved_preamble_ja,
    interleaved_preamble_vi,
    load_content_list_raw,
    normalize_submission_answer,
    prepare_mm_data,
    prompt_answer_ja_interleaved,
    prompt_answer_vi_interleaved,
    release_torch_memory,
    resolve_training_evidence_pages,
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
GEMINI_CSV = os.environ.get("GEMINI_CSV", "gemini1.csv")
OUT_JSONL = os.environ.get("OUT_JSONL", "pdf_rag_gemini_cot_sft.jsonl")
OUT_RAW_JSON = os.environ.get("OUT_RAW_JSON", "pdf_rag_gemini_cot_raw.json")
OUT_SKIPPED_JSONL = os.environ.get("OUT_SKIPPED_JSONL", "pdf_rag_gemini_cot_skipped.jsonl")
OUT_MISMATCH_REPORT = os.environ.get(
    "OUT_MISMATCH_REPORT", "evidence_mismatch_report.json"
)
PAGE_CACHE_DIR = os.environ.get("PAGE_CACHE_DIR", "pdf_rag_gemini_cot_page_cache")
SEED = int(os.environ.get("SEED", "42"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
MIN_ANALYSIS_CHARS = int(os.environ.get("MIN_ANALYSIS_CHARS", "80"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32000"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
LIMIT_MM_IMAGES_PER_PROMPT = int(os.environ.get("LIMIT_MM_IMAGES_PER_PROMPT", "36"))
LIMIT_MM_PER_PROMPT = {"image": LIMIT_MM_IMAGES_PER_PROMPT, "video": 0}

SAMPLING_COT = SamplingParams(
    temperature=float(os.environ.get("COT_TEMPERATURE", "0.2")),
    top_p=float(os.environ.get("COT_TOP_P", "0.9")),
    top_k=int(os.environ.get("COT_TOP_K", "10")),
    repetition_penalty=float(os.environ.get("COT_REPETITION_PENALTY", "1.1")),
    presence_penalty=0.0,
    max_tokens=int(os.environ.get("COT_MAX_TOKENS", "8192")),
    stop_token_ids=[],
    seed=SEED,
)


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


def load_gemini_by_id(csv_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = (row.get("id") or "").strip()
            if not qid:
                continue
            out[qid] = {
                "answer": (row.get("answer") or "").strip(),
                "evidence_page_number": (row.get("evidence_page_number") or "").strip(),
            }
    return out


def parse_evidence_column(s: str) -> List[int]:
    t = (s or "").strip()
    if not t:
        return []
    try:
        v = ast.literal_eval(t)
    except (ValueError, SyntaxError):
        nums = re.findall(r"\d+", t)
        return [int(x) for x in nums if int(x) > 0]
    if isinstance(v, int):
        return [v] if v > 0 else []
    if isinstance(v, list):
        seen: set[int] = set()
        out: List[int] = []
        for x in v:
            try:
                n = int(x)
            except (TypeError, ValueError):
                continue
            if n > 0 and n not in seen:
                seen.add(n)
                out.append(n)
        return out
    return []


def ctx_page_nums_from_selected(selected_pages: List[Dict[str, Any]]) -> List[int]:
    return sorted(
        {
            int(p.get("page_num", int(p["page_idx"]) + 1))
            for p in selected_pages
        }
    )


def load_merged_rows(script_dir: str) -> List[Dict[str, Any]]:
    rag_path = _resolve(RAG_PAGES_JSONL, script_dir)
    csv_path = _resolve(QUESTIONS_CSV, script_dir)
    gem_path = _resolve(GEMINI_CSV, script_dir)
    rag_by_id = load_rag_by_id(rag_path)
    gem_by_id = load_gemini_by_id(gem_path)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到题目 CSV: {csv_path}")
    if not os.path.isfile(gem_path):
        raise FileNotFoundError(f"未找到 Gemini 答案 CSV: {gem_path}")

    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    for cr in csv_rows:
        qid = cr["id"]
        if qid not in rag_by_id:
            raise KeyError(f"rag_top_pages.jsonl 缺少 id={qid}")
        if qid not in gem_by_id:
            raise KeyError(f"{GEMINI_CSV} 缺少 id={qid}")
        rag = rag_by_id[qid]
        gem = gem_by_id[qid]
        language = (cr.get("language") or rag.get("language") or "ja").strip()
        answer_format = (
            cr.get("answer_format") or rag.get("answer_format") or "string"
        ).strip()
        ref_evidence_full = parse_evidence_column(gem["evidence_page_number"])
        ctx_pages = ctx_page_nums_from_selected(rag["selected_pages"])
        train_evidence, ev_meta = resolve_training_evidence_pages(
            ref_evidence_full, ctx_pages
        )
        norm_answer = normalize_submission_answer(
            gem["answer"], answer_format, language
        )
        rows.append(
            {
                "id": qid,
                "file_id": cr.get("file_id") or rag["file_id"],
                "question": cr.get("question") or rag["question"],
                "language": language,
                "answer_format": answer_format,
                "selected_pages": rag["selected_pages"],
                "origin_pdf": rag.get("origin_pdf") or "",
                "ref_answer_raw": gem["answer"],
                "ref_answer": norm_answer,
                "ref_evidence_full": ref_evidence_full,
                "train_evidence_pages": train_evidence,
                "ctx_page_nums": ctx_pages,
                "evidence_meta": ev_meta,
            }
        )
    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    return rows


def truncate_interleaved_images(
    interleaved_parts: List[Dict[str, Any]],
    image_paths: List[str],
    limit: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if len(image_paths) <= limit:
        return interleaved_parts, image_paths
    keep_imgs = set(image_paths[:limit])
    parts = [
        p
        for p in interleaved_parts
        if p["type"] != "image" or p.get("image") in keep_imgs
    ]
    return parts, image_paths[:limit]


def _answer_format_thinking_hint(answer_format: str, lang: str) -> str:
    afmt = (answer_format or "string").strip()
    if lang == "vi":
        hints = {
            "number": (
                "Định dạng number: nêu vị trí bảng/đồ thị → đơn vị → các bước tính → "
                "kết quả khớp đáp án tham chiếu."
            ),
            "ordered_list": (
                "Định dạng ordered_list: giải thích thứ tự (thời gian/thứ tự đề bài) "
                "rồi liệt kê từng mục tương ứng tài liệu."
            ),
            "unordered_list": (
                "Định dạng unordered_list: với mỗi mục, chỉ rõ trang và đoạn/bảng; "
                "không cần thứ tự."
            ),
            "string": (
                "Định dạng string: trích dẫn ngắn nguồn (trang + nội dung) rồi "
                "suy ra câu trả lời một dòng."
            ),
        }
    else:
        hints = {
            "number": (
                "answer_format=number：表・図の位置→単位→計算過程→参考解答と一致する数値。"
            ),
            "ordered_list": (
                "answer_format=ordered_list：順序の根拠（日付・設問順）を述べ、"
                "各要素を資料のどのページ・表に対応づける。"
            ),
            "unordered_list": (
                "answer_format=unordered_list：各要素ごとにページと根拠箇所を示す（順不同）。"
            ),
            "string": (
                "answer_format=string：根拠文（ページ番号付き）→一行の結論、の順で簡潔に。"
            ),
        }
    return hints.get(afmt, hints["string"])


def cot_supervisor_system_ja() -> str:
    return (
        "あなたは固定設問向けの教師用ラベラーです。"
        "提示ページの内容を手がかりに、参考解答・訓練用根拠ページへ至る"
        "再現可能な中間思考を日本語で書きます（同じ設問に毎回同じ結論へ辿れる手順）。"
        "外部知識は禁止。思考では必ず「ページ N の〜」（PDF 1 始まり）を明示してください。"
        "訓練用根拠ページの集合は最終的な <evidence> と一致させる意識で書いてください。"
        "最終出力は <think>...</think> の1ブロックのみ。"
        "<answer> や <evidence> は出力しないでください。"
    )


def cot_supervisor_system_vi() -> str:
    return (
        "Bạn gán nhãn suy luận (chế độ giáo viên). Chỉ dùng các trang được cung cấp; "
        "viết suy luận trung gian (tiếng Việt) thống nhất với đáp án và trang minh chứng tham chiếu. "
        "Luôn ghi rõ «trang N» (đếm từ 1). "
        "Chỉ xuất một khối <think>...</think>, "
        "không xuất <answer> hay <evidence>."
    )


def cot_supervisor_user_block(
    *,
    question: str,
    answer_format: str,
    ref_answer: str,
    train_evidence: List[int],
    ref_evidence_full: List[int],
    missing_from_ctx: List[int],
    used_ref_label_fallback: bool,
    used_ctx_fallback: bool,
    ctx_page_nums: List[int],
    language: str,
) -> str:
    pages_ctx = ", ".join(str(p) for p in ctx_page_nums)
    train_ev_str = json.dumps(train_evidence, ensure_ascii=False)
    full_ev_str = json.dumps(ref_evidence_full, ensure_ascii=False)
    missing_str = json.dumps(missing_from_ctx, ensure_ascii=False)
    lang = (language or "ja").strip().lower()
    fmt_hint = _answer_format_thinking_hint(answer_format, lang)

    if lang == "vi":
        ref_fb_note = (
            "Lưu ý: giao tham chiếu–ngữ cảnh rỗng; trang minh chứng huấn luyện = "
            f"nhãn đáp án đầy đủ (cần ghi nhớ cho câu cố định): {train_ev_str}.\n"
            if used_ref_label_fallback
            else ""
        )
        ctx_fb_note = (
            "Lưu ý: không có nhãn trang; dùng toàn bộ trang RAG: "
            f"{train_ev_str}.\n"
            if used_ctx_fallback and not used_ref_label_fallback
            else ""
        )
        fallback_note = ref_fb_note + ctx_fb_note
        missing_note = (
            f"Các trang trong nhãn gốc nhưng không có trong ngữ cảnh: {missing_str}. "
            "Không bịa nội dung các trang đó; có thể nêu là ngoài phạm vi RAG.\n"
            if missing_from_ctx
            else ""
        )
        return (
            f"Câu hỏi:\n{question}\n\n"
            f"answer_format: {answer_format}\n"
            f"{fmt_hint}\n\n"
            f"Đáp án tham chiếu (chuỗi nộp CSV sau chuẩn hóa):\n{ref_answer}\n\n"
            f"Trang minh chứng huấn luyện (giao với ngữ cảnh, không rỗng): {train_ev_str}\n"
            f"Nhãn gốc đầy đủ: {full_ev_str}\n"
            f"Chỉ các trang PDF sau có trong ngữ cảnh: {pages_ctx}.\n"
            f"{missing_note}{fallback_note}"
            "Viết suy luận trung gian: đọc bảng/hình → lập luận → khớp đáp án; "
            "không lặp lại đáp án một dòng mà không giải thích nguồn."
        )

    ref_fb_note = (
        "注意: 参考根拠とコンテキストの交差が空のため、"
        f"訓練用根拠ページは答案ラベル（Gemini）の一覧そのままです: {train_ev_str}。"
        "固定設問の提出形式を記憶するため、思考でもこのページ集合に整合させてください。\n"
        if used_ref_label_fallback
        else ""
    )
    ctx_fb_note = (
        "注意: 答案に根拠ページラベルが無いため、"
        f"訓練用根拠は RAG 提示ページ全体です: {train_ev_str}。\n"
        if used_ctx_fallback and not used_ref_label_fallback
        else ""
    )
    fallback_note = ref_fb_note + ctx_fb_note
    missing_note = (
        f"ラベル上の根拠のうち本コンテキストに無いページ: {missing_str}。"
        "これらのページの内容は推測で書かず、「RAG に含まれない」とだけ触れてよい。\n"
        if missing_from_ctx
        else ""
    )
    return (
        f"設問:\n{question}\n\n"
        f"answer_format: {answer_format}\n"
        f"{fmt_hint}\n\n"
        f"参考解答（提出 CSV と同形式に正規化済み）:\n{ref_answer}\n\n"
        f"訓練用根拠ページ（SFT の <evidence> にそのまま入る・非空）: {train_ev_str}\n"
        f"Gemini 元ラベル（全ページ）: {full_ev_str}\n"
        f"本プロンプトに含まれる PDF ページのみ: {pages_ctx}。\n"
        f"{missing_note}{fallback_note}"
        "表・図・本文をページ番号付きで参照し、必要なら計算過程を示したうえで、"
        "参考解答と訓練用根拠ページに至る中間思考を書いてください。"
        "根拠の当たり付け→読取→結論の順を守り、答えの一行転記だけは避けてください。"
    )


def extract_redacted_thinking(raw: str) -> str:
    text = (raw or "").strip()
    m = re.search(
        r"<think>\s*(.*?)\s*</think>",
        text,
        flags=re.S | re.I,
    )
    if m:
        return m.group(1).strip()
    _open, _close = "".join(["<", "think", ">"]), "".join(["</", "think", ">"])
    m = re.search(
        re.escape(_open) + r"\s*(.*?)\s*" + re.escape(_close),
        text,
        flags=re.S,
    )
    if m:
        return m.group(1).strip()
    text = re.split(r"<\s*answer\s*>", text, maxsplit=1, flags=re.I)[0].strip()
    return text


def build_interleaved_for_row(
    row: Dict[str, Any],
    cl_cache: Dict[str, List[Dict[str, Any]]],
    cl_dir_cache: Dict[str, str],
    md_root: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    fid = row["file_id"]
    lang = (row.get("language") or "ja").strip().lower()
    page_indices = [int(p["page_idx"]) for p in row["selected_pages"]]
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
        interleaved_parts, image_paths = truncate_interleaved_images(
            interleaved_parts,
            image_paths,
            LIMIT_MM_IMAGES_PER_PROMPT,
        )
    return interleaved_parts, image_paths, temp_pngs


def build_mismatch_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    n_missing = 0
    n_ref_fallback = 0
    n_ctx_fallback = 0
    for row in rows:
        em = row.get("evidence_meta") or {}
        missing = em.get("missing_from_ctx") or []
        used_ref_fb = bool(em.get("used_ref_label_fallback"))
        used_ctx_fb = bool(em.get("used_ctx_fallback"))
        if missing:
            n_missing += 1
        if used_ref_fb:
            n_ref_fallback += 1
        if used_ctx_fb:
            n_ctx_fallback += 1
        if missing or used_ref_fb or used_ctx_fb:
            items.append(
                {
                    "id": row["id"],
                    "ref_evidence_full": row["ref_evidence_full"],
                    "train_evidence_pages": row["train_evidence_pages"],
                    "ctx_page_nums": row["ctx_page_nums"],
                    "missing_from_ctx": missing,
                    "used_ref_label_fallback": used_ref_fb,
                    "used_ctx_fallback": used_ctx_fb,
                }
            )
    return {
        "total_questions": len(rows),
        "with_missing_evidence_pages": n_missing,
        "with_ref_label_fallback_evidence": n_ref_fallback,
        "with_ctx_fallback_evidence": n_ctx_fallback,
        "items": items,
    }


def main() -> None:
    t0 = time.perf_counter()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_root = _resolve(OUTPUT_TEST_DIR, script_dir)
    page_cache_dir = _resolve(PAGE_CACHE_DIR, script_dir)
    os.makedirs(page_cache_dir, exist_ok=True)

    print(f"[开始] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    set_random_seed(SEED)
    print(f"SEED={SEED} MODEL_PATH={MODEL_PATH}")
    print(f"OUTPUT_TEST_DIR={md_root}")
    print(f"GEMINI_CSV={_resolve(GEMINI_CSV, script_dir)}")
    print(f"MIN_ANALYSIS_CHARS={MIN_ANALYSIS_CHARS}")

    if not os.path.isdir(md_root):
        raise FileNotFoundError(f"OUTPUT_TEST_DIR 不存在: {md_root}")

    rows = load_merged_rows(script_dir)
    print(f"待生成 {len(rows)} 条（MAX_SAMPLES={MAX_SAMPLES or '全部'}）")

    mismatch_report = build_mismatch_report(rows)
    mismatch_path = _resolve(OUT_MISMATCH_REPORT, script_dir)
    with open(mismatch_path, "w", encoding="utf-8") as mf:
        json.dump(mismatch_report, mf, ensure_ascii=False, indent=2)
    print(
        f"证据对齐报告: {mismatch_path} "
        f"(缺页={mismatch_report['with_missing_evidence_pages']}, "
        f"交集空→用标签={mismatch_report['with_ref_label_fallback_evidence']}, "
        f"无标签→RAG页={mismatch_report['with_ctx_fallback_evidence']})"
    )

    cl_cache: Dict[str, List[Dict[str, Any]]] = {}
    cl_dir_cache: Dict[str, str] = {}
    gen_inputs: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []

    for row in rows:
        if not (row.get("ref_answer") or "").strip():
            print(f"[warn] id={row['id']} ref_answer 为空，仍将尝试生成 CoT")

        interleaved_parts, image_paths, temp_pngs = build_interleaved_for_row(
            row, cl_cache, cl_dir_cache, md_root
        )
        persist_fullpage_images(
            interleaved_parts,
            temp_pngs,
            page_cache_dir,
            row["id"],
        )

        lang = (row.get("language") or "ja").strip().lower()
        page_indices = [int(p["page_idx"]) for p in row["selected_pages"]]
        preamble = (
            interleaved_preamble_vi(page_indices)
            if lang == "vi"
            else interleaved_preamble_ja(page_indices)
        )
        em = row.get("evidence_meta") or {}
        sup_block = cot_supervisor_user_block(
            question=row["question"],
            answer_format=row.get("answer_format", "string"),
            ref_answer=row["ref_answer"],
            train_evidence=row["train_evidence_pages"],
            ref_evidence_full=row["ref_evidence_full"],
            missing_from_ctx=em.get("missing_from_ctx") or [],
            used_ref_label_fallback=bool(em.get("used_ref_label_fallback")),
            used_ctx_fallback=bool(em.get("used_ctx_fallback")),
            ctx_page_nums=row["ctx_page_nums"],
            language=lang,
        )
        sys_sup = cot_supervisor_system_vi() if lang == "vi" else cot_supervisor_system_ja()
        sup_msgs = build_interleaved_messages(
            sys_sup, preamble, interleaved_parts, sup_block
        )
        meta.append(
            {
                "row": row,
                "interleaved_parts": interleaved_parts,
                "preamble": preamble,
                "sup_msgs": sup_msgs,
                "image_paths": image_paths,
            }
        )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    for m in meta:
        llm_in: Dict[str, Any] = {
            "prompt": apply_generation_prompt_without_thinking(
                processor, m["sup_msgs"]
            )
        }
        mm = prepare_mm_data(m["sup_msgs"], m["image_paths"])
        if mm:
            llm_in["multi_modal_data"] = mm
        gen_inputs.append(llm_in)

    print(
        f"加载 VLM: {MODEL_PATH} "
        f"(LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT})"
    )
    t_llm = time.perf_counter()
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

    print(f"生成 CoT {len(gen_inputs)} 条…")
    t_gen = time.perf_counter()
    outs = llm.generate(gen_inputs, sampling_params=SAMPLING_COT)
    print(f"[timing] llm.generate: {time.perf_counter() - t_gen:.3f}s")

    del llm
    del processor
    release_torch_memory()

    out_jsonl = _resolve(OUT_JSONL, script_dir)
    skipped_path = _resolve(OUT_SKIPPED_JSONL, script_dir)
    raw_path = _resolve(OUT_RAW_JSON, script_dir)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    raw_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout, open(
        skipped_path, "w", encoding="utf-8"
    ) as fskip:
        for m, out in zip(meta, outs):
            row = m["row"]
            raw_text = out.outputs[0].text
            analysis = extract_redacted_thinking(raw_text)
            em = row.get("evidence_meta") or {}

            raw_rec = {
                "id": row["id"],
                "file_id": row["file_id"],
                "ref_answer_raw": row.get("ref_answer_raw"),
                "ref_answer_norm": row["ref_answer"],
                "ref_evidence_full": row["ref_evidence_full"],
                "train_evidence_pages": row["train_evidence_pages"],
                "ctx_page_nums": row["ctx_page_nums"],
                "missing_from_ctx": em.get("missing_from_ctx"),
                "used_ref_label_fallback": em.get("used_ref_label_fallback"),
                "used_ctx_fallback": em.get("used_ctx_fallback"),
                "raw_model_output": raw_text,
                "extracted_analysis_len": len(analysis),
            }
            raw_records.append(raw_rec)

            skip_reason = ""
            if not (row.get("ref_answer") or "").strip():
                skip_reason = "empty_ref_answer"
            elif len(analysis.strip()) < MIN_ANALYSIS_CHARS:
                skip_reason = f"short_thinking(<{MIN_ANALYSIS_CHARS} chars)"

            if skip_reason:
                n_skip += 1
                rec = {**raw_rec, "skip_reason": skip_reason}
                skipped_records.append(rec)
                fskip.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[skip] id={row['id']} reason={skip_reason}")
                continue

            answer_rec = {
                "analysis": analysis,
                "answer": row["ref_answer"],
                "evidence": row["train_evidence_pages"],
            }
            assistant_content = format_assistant_target(answer_rec)

            lang = (row.get("language") or "ja").strip().lower()
            system = system_msg_vi() if lang == "vi" else system_msg_ja()
            q_block = (
                prompt_answer_vi_interleaved(
                    row["question"], row.get("answer_format", "string"), ""
                )
                if lang == "vi"
                else prompt_answer_ja_interleaved(
                    row["question"], row.get("answer_format", "string"), ""
                )
            )
            user_content, images = interleaved_parts_to_user_content(
                m["preamble"],
                m["interleaved_parts"],
                q_block,
            )

            sample = {
                "id": row["id"],
                "file_id": row["file_id"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": images,
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"已写入 SFT {n_ok} 条…")

    with open(raw_path, "w", encoding="utf-8") as rf:
        json.dump(raw_records, rf, ensure_ascii=False, indent=2)

    print(f"已写入 SFT: {out_jsonl}（{n_ok} 条）")
    print(f"已跳过: {skipped_path}（{n_skip} 条）")
    print(f"已写入 raw: {raw_path}（{len(raw_records)} 条）")
    print(f"整页渲染缓存: {page_cache_dir}")
    print(f"[总耗时] {time.perf_counter() - t0:.3f}s")

    if n_ok == 0:
        raise RuntimeError("无有效 SFT 样本写入，请检查 MIN_ANALYSIS_CHARS 或模型输出。")


if __name__ == "__main__":
    main()
