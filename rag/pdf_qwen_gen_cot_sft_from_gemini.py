#!/usr/bin/env python3
"""
用 vLLM 根据「RAG Top-K 页图文交错 + Gemini 参考答案/证据」生成 CoT，并组装 ms-swift SFT jsonl。

数据规则（与线上一致）：
  - user：与 pdf_qwen_infer / pdf_qwen_test 相同（不含教师参考答案）。
  - assistant：<think> 由模型生成；<answer>/<evidence> 来自 Gemini CSV。
  - answer 经 normalize_submission_answer 与推理提交格式对齐。
  - evidence：ref∩RAG 可见页；交集为空则用 Gemini 标签中的完整证据页（与答案一并记忆）。
  - OUT_RAW_JSON：含 question、gemini_answer（CSV 原文）、gemini_answer_norm、gemini_evidence_pages 等。

依赖：transformers、torch、vllm；在 rag 目录下执行。

示例:
  cd rag
  MODEL_PATH=... OUTPUT_TEST_DIR=... GEMINI_CSV=gemini1.csv \\
  python3 pdf_qwen_gen_cot_sft_from_gemini.py

  # 四卡张量并行（未设置时默认使用全部可见 GPU）
  TENSOR_PARALLEL_SIZE=4 python3 pdf_qwen_gen_cot_sft_from_gemini.py

  # 关闭超长处理（重试 + 提炼）
  COT_REFINE_ENABLED=0 python3 pdf_qwen_gen_cot_sft_from_gemini.py

  # 过短/过长：各重生成 3 次（过短取最长且保留，过长取最短合格稿），仍超长再提炼
  COT_RETRY_SAMPLES=3 COT_REFINE_TOKEN_THRESHOLD=2048 python3 pdf_qwen_gen_cot_sft_from_gemini.py
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import torch
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
    "xiacheng-240108120111/hf_download/Qwen3.5-27B"
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
OUT_LENGTH_STATS = os.environ.get(
    "OUT_LENGTH_STATS", "pdf_rag_gemini_cot_length_stats.json"
)
PAGE_CACHE_DIR = os.environ.get("PAGE_CACHE_DIR", "pdf_rag_gemini_cot_page_cache")
SEED = int(os.environ.get("SEED", "42"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
MIN_ANALYSIS_CHARS = int(os.environ.get("MIN_ANALYSIS_CHARS", "80"))
COT_REFINE_TOKEN_THRESHOLD = int(os.environ.get("COT_REFINE_TOKEN_THRESHOLD", "2048"))
COT_REFINE_ENABLED = os.environ.get("COT_REFINE_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
COT_RETRY_SAMPLES = max(1, int(os.environ.get("COT_RETRY_SAMPLES", "3")))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32000"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
_TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "0"))
LIMIT_MM_IMAGES_PER_PROMPT = int(os.environ.get("LIMIT_MM_IMAGES_PER_PROMPT", "36"))
LIMIT_MM_PER_PROMPT = {"image": LIMIT_MM_IMAGES_PER_PROMPT, "video": 0}

SAMPLING_COT = SamplingParams(
    temperature=float(os.environ.get("COT_TEMPERATURE", "0.15")),
    top_p=float(os.environ.get("COT_TOP_P", "0.9")),
    top_k=int(os.environ.get("COT_TOP_K", "10")),
    repetition_penalty=float(os.environ.get("COT_REPETITION_PENALTY", "1.1")),
    presence_penalty=0.0,
    max_tokens=int(os.environ.get("COT_MAX_TOKENS", "8192")),
    stop_token_ids=[],
    seed=SEED,
)
SAMPLING_RETRY = SamplingParams(
    temperature=float(os.environ.get("COT_TEMPERATURE", "0.15")),
    top_p=float(os.environ.get("COT_TOP_P", "0.9")),
    top_k=int(os.environ.get("COT_TOP_K", "10")),
    repetition_penalty=float(os.environ.get("COT_REPETITION_PENALTY", "1.1")),
    presence_penalty=0.0,
    max_tokens=int(os.environ.get("COT_MAX_TOKENS", "8192")),
    stop_token_ids=[],
    seed=SEED,
    n=COT_RETRY_SAMPLES,
)
SAMPLING_REFINE = SamplingParams(
    temperature=float(os.environ.get("COT_REFINE_TEMPERATURE", "0.1")),
    top_p=float(os.environ.get("COT_REFINE_TOP_P", "0.85")),
    top_k=int(os.environ.get("COT_REFINE_TOP_K", "10")),
    repetition_penalty=float(os.environ.get("COT_REFINE_REPETITION_PENALTY", "1.15")),
    presence_penalty=0.0,
    max_tokens=int(os.environ.get("COT_REFINE_MAX_TOKENS", "4096")),
    stop_token_ids=[],
    seed=SEED,
)


def _resolve(path: str, script_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def resolve_tensor_parallel_size() -> int:
    """0 表示自动使用全部可见 GPU；否则使用指定张量并行度。"""
    if _TENSOR_PARALLEL_SIZE > 0:
        return _TENSOR_PARALLEL_SIZE
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


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


def _cot_efficiency_rules(lang: str) -> str:
    """高效思维：按题定信息需求，定位页与数值，计算比较，收束结论。"""
    if (lang or "").strip().lower() == "vi":
        return (
            "【Suy nghĩ hiệu quả】\n"
            "- Trước khi đọc: xác định thông tin cần để trả lời (số, thứ tự, cụm từ, so sánh…).\n"
            "- Chỉ mở/đọc trang có khả năng chứa thông tin đó; mỗi trang: trang N → thấy gì → "
            "số liệu/câu trích → so sánh/tính một dòng nếu cần.\n"
            "- Không lặp cùng một trang; không giải thích chung chung; không thêm bước thừa.\n"
            "- Kết bằng nhận định ngắn (không nhắc lại đáp án đúng).\n"
        )
    return (
        "【効率的な思考】\n"
        "- 読む前に：この設問に必要な情報だけを頭で決める（数値・順序・文言・比較など）。\n"
        "- 該当しそうなページだけ追い、各ページは「ページ N → 見た内容 → 読んだ数値/文言 → "
        "必要なら一行で比較・計算」に留める。\n"
        "- 同じページの往復説明・一般論・手順宣言・メタ解説は書かない。\n"
        "- 最後は短い所感で終える（正解の再掲はしない）。\n"
    )


def _cot_style_rules(lang: str) -> str:
    """文体约束：像读资料时的内心独白，不要解题教程/讲义。"""
    efficiency = _cot_efficiency_rules(lang)
    if (lang or "").strip().lower() == "vi":
        return (
            f"{efficiency}"
            "【Cách viết】\n"
            "- Viết như độc thoại nội bộ khi đang đọc tài liệu (tôi/thấy/có vẻ), không viết bài hướng dẫn.\n"
            "- Cấm: nhắc lại đề word-by-word; «bước 1/2/3»; tiêu đề Markdown; «đáp án tham chiếu»; "
            "«cần ghi nhớ»; «theo yêu cầu đề bài»; kết luận một dòng trùng đáp án cuối.\n"
            "- Được: nhận xét ngắn khi lật trang; trích số liệu/câu; phép tính gọn; "
            "chỉ rõ «trang N» (đếm từ 1).\n"
        )
    return (
        f"{efficiency}"
        "【文体】\n"
        "- 読みながらの内心独白（見る／気づく／比べる）。解説記事・解法テンプレ・講義調は禁止。\n"
        "- 禁止：設問の言い換え長文；「まず〜を確認します」型の手順宣言；"
        "「1. 2. 3.」番号付き解説；**見出し**や箇条書き教程；"
        "「参考解答」「訓練用」「手順」「チュートリアル」等のメタ語；"
        "最後に答えだけ一行で繰り返す。\n"
        "- 可：ページを開いたときの短い所感；表・図から読んだ数値；"
        "必要なら一行計算；必ず「ページ N」（PDF 1 始まり）。\n"
    )


def _cot_format_nudge(answer_format: str, lang: str) -> str:
    """仅一句格式提醒，不给出流水线步骤。"""
    afmt = (answer_format or "string").strip()
    if (lang or "").strip().lower() == "vi":
        nudges = {
            "number": "Kết quả cuối là một số/đơn vị (không giải thích dài ở cuối).",
            "ordered_list": "Cuối cùng cần thứ tự các mục — suy luận nên lộ trình thời gian/thứ tự khi đọc.",
            "unordered_list": "Cuối cùng là tập mục — gắn từng mục với trang khi gặp trong tài liệu.",
            "string": "Kết quả cuối là một cụm ngắn — trích ý từ đoạn cụ thể trên trang.",
        }
    else:
        nudges = {
            "number": "最終的に数値・単位が一つになる問題（結論の一行提示は不要）。",
            "ordered_list": "最終的に順序付きリストになるので、読み進めながら順序の根拠に触れる。",
            "unordered_list": "最終的に複数項目になるので、見つけた項目をページと結びつける。",
            "string": "最終的に短い文字列になるので、該当フレーズがどのページかを辿る。",
        }
    return nudges.get(afmt, nudges["string"])


def cot_supervisor_system_ja() -> str:
    return (
        "あなたは、提出前に資料だけを読んで考えを整理するアシスタントです。"
        "今は回答タグを出さず、読んだ直後の思考だけを書きます。"
        "設問に必要な情報だけを取り、該当ページと数値を特定し、比較・計算して結論に至る。"
        "冗長な前置き・同じ内容の繰り返し・他人への講義は書かない。"
        "他人への説明や解法講座ではなく、自分用のメモのような独白にしてください。"
        "外部知識は使わず、提示されたページの内容だけに基づいてください。"
        "出力は <think>...</think> の1ブロックのみ。"
        "<answer> や <evidence> は書かないでください。"
    )


def cot_supervisor_system_vi() -> str:
    return (
        "Bạn là trợ lý đang đọc tài liệu trước khi nộp bài; chỉ viết suy nghĩ nội bộ, "
        "không viết bài hướng dẫn cho người khác. "
        "Xác định thông tin cần cho câu hỏi, tìm đúng trang và số liệu, so sánh/tính rồi kết luận; "
        "không thêm bước hay lặp lại thừa. "
        "Chỉ dựa trên các trang được cung cấp. "
        "Chỉ xuất <think>...</think>, không xuất <answer> hay <evidence>."
    )


def cot_supervisor_user_block(
    *,
    question: str,
    answer_format: str,
    ref_answer: str,
    train_evidence: List[int],
    missing_from_ctx: List[int],
    ctx_page_nums: List[int],
    language: str,
) -> str:
    pages_ctx = ", ".join(str(p) for p in ctx_page_nums)
    train_ev_str = json.dumps(train_evidence, ensure_ascii=False)
    missing_str = json.dumps(missing_from_ctx, ensure_ascii=False)
    lang = (language or "ja").strip().lower()
    style = _cot_style_rules(lang)
    nudge = _cot_format_nudge(answer_format, lang)

    if lang == "vi":
        missing_note = (
            f"Một số trang trong nhãn gốc không có trong ngữ cảnh ({missing_str}); "
            "nếu nhắc tới thì chỉ nói là không thấy trong các trang đã gửi, không bịa.\n"
            if missing_from_ctx
            else ""
        )
        ev_note = (
            f"Sau khi đọc, trọng tâm minh chứng sẽ là các trang {train_ev_str} "
            "(hãy để lộ khi bạn «phát hiện» trên các trang đó).\n"
        )
        internal = (
            "【Chỉ để đối chiếu nội bộ — KHÔNG nhắc trong độc thoại, KHÔNG chép nguyên câu trả lời】\n"
            f"Đáp án đúng: {ref_answer}\n"
        )
        return (
            f"{style}\n"
            f"Câu hỏi (chỉ để biết cần tìm gì, không viết lại dài):\n{question}\n\n"
            f"Định dạng nộp: {answer_format} — {nudge}\n"
            f"Các trang PDF bạn đang đọc: {pages_ctx}.\n"
            f"{ev_note}{missing_note}"
            f"{internal}\n"
            "Viết <think> ngắn gọn: đọc các trang trên theo nhu cầu của câu hỏi; "
            "mỗi trang chỉ ghi phát hiện và số liệu liên quan; so sánh/tính một dòng nếu cần; "
            "kết bằng nhận định ngắn, không lặp đáp án đúng."
        )

    missing_note = (
        f"ラベル上は根拠に含まれるが今回の提示に無いページ: {missing_str}。"
        "触れる場合は「この抜粋には無い」とだけ書き、内容は推測しない。\n"
        if missing_from_ctx
        else ""
    )
    ev_note = (
        f"読み終えたとき、根拠として強いのはページ {train_ev_str} 付近だと自然に辿れるように書く。\n"
    )
    internal = (
        "【内部照合用・思考本文に書かない・正解文をそのまま繰り返さない】\n"
        f"正解の答え: {ref_answer}\n"
    )
    return (
        f"{style}\n"
        f"設問（何を探すかのメモ程度。設問文の言い換え長文は不要）:\n{question}\n\n"
        f"提出形式: {answer_format} — {nudge}\n"
        f"今回読める PDF ページ: {pages_ctx}。\n"
        f"{ev_note}{missing_note}"
        f"{internal}\n"
        "上の資料を読み、<think> は必要な情報だけを簡潔に書く。"
        "設問に要る情報を先に決め、該当ページで見つけた数値・文言をページ番号付きで記し、"
        "比較・計算は一行まで。同じ説明の繰り返し・手順宣言・正解の再掲はしない。"
    )


def count_text_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def compute_length_stats(values: List[int]) -> Dict[str, Any]:
    """对一组长度值返回 count/min/max/mean/median。"""
    if not values:
        return {"count": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n % 2:
        median = float(sorted_v[n // 2])
    else:
        median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
    return {
        "count": n,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "mean": round(sum(sorted_v) / n, 1),
        "median": median,
    }


def _format_stats_line(label: str, stats: Dict[str, Any], unit: str) -> str:
    if not stats.get("count"):
        return f"  {label}: (无数据)"
    return (
        f"  {label}: n={stats['count']} "
        f"min={stats['min']} max={stats['max']} "
        f"mean={stats['mean']} median={stats['median']} {unit}"
    )


def build_length_summary(
    gen_results: List[Dict[str, Any]],
    *,
    sft_ids: set[str],
    refine_threshold: int,
) -> Dict[str, Any]:
    all_tokens = [int(it["token_count"]) for it in gen_results]
    initial_tokens = [
        int(it.get("initial_token_count", it["token_count"])) for it in gen_results
    ]
    sft_tokens = [
        int(it["token_count"]) for it in gen_results if it["row"]["id"] in sft_ids
    ]
    over_threshold = sum(1 for t in initial_tokens if t > refine_threshold)
    n_retried = sum(1 for it in gen_results if it.get("retried"))
    n_retry_short = sum(1 for it in gen_results if it.get("retry_mode") == "short")
    n_retry_long = sum(1 for it in gen_results if it.get("retry_mode") == "long")
    n_refined = sum(1 for it in gen_results if it.get("refined"))
    return {
        "refine_threshold_tokens": refine_threshold,
        "min_analysis_chars": MIN_ANALYSIS_CHARS,
        "over_refine_threshold_initial": over_threshold,
        "retry_resolved_count": n_retried,
        "retry_short_count": n_retry_short,
        "retry_long_count": n_retry_long,
        "refined_count": n_refined,
        "all_generated": {
            "tokens_initial": compute_length_stats(initial_tokens),
            "tokens_final": compute_length_stats(all_tokens),
        },
        "sft_written": {
            "tokens": compute_length_stats(sft_tokens),
        },
    }


def print_and_save_length_summary(
    summary: Dict[str, Any],
    stats_path: str,
) -> None:
    ag = summary["all_generated"]
    sw = summary["sft_written"]
    print("\n========== CoT token 统计 ==========")
    print(
        f"  初稿超过提炼阈值({summary['refine_threshold_tokens']} tokens): "
        f"{summary['over_refine_threshold_initial']} 条"
    )
    print(
        f"  重生成: 过短={summary['retry_short_count']} | "
        f"过长={summary['retry_long_count']} | "
        f"提炼={summary['refined_count']}"
    )
    print("【全部生成（含跳过）】")
    print(_format_stats_line("初稿", ag["tokens_initial"], "tokens"))
    print(_format_stats_line("终稿", ag["tokens_final"], "tokens"))
    print("【写入 SFT 的有效样本】")
    print(_format_stats_line("终稿", sw["tokens"], "tokens"))
    print("====================================\n")
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    print(f"token 统计已写入: {stats_path}")


def cot_refine_system_ja() -> str:
    return (
        "あなたは長すぎる読書メモを圧縮するアシスタントです。"
        "事実・ページ番号・数値・比較・計算・発見の順序は残し、"
        "重複・設問の言い換え・手順宣言・メタ解説・正解の再掲は削除します。"
        "文体は内心独白のまま。出力は <think>...</think> のみ。"
        "<answer> や <evidence> は書かないでください。"
    )


def cot_refine_system_vi() -> str:
    return (
        "Bạn rút gọn bản nháp suy nghĩ quá dài. Giữ trang, số liệu, so sánh/tính và thứ tự phát hiện; "
        "bỏ lặp, diễn giải đề, bước thừa, meta và nhắc lại đáp án. "
        "Chỉ xuất <think>...</think>."
    )


def cot_refine_user_block(
    *,
    question: str,
    answer_format: str,
    ref_answer: str,
    train_evidence: List[int],
    ctx_page_nums: List[int],
    language: str,
    long_thinking: str,
    token_count: int,
) -> str:
    pages_ctx = ", ".join(str(p) for p in ctx_page_nums)
    train_ev_str = json.dumps(train_evidence, ensure_ascii=False)
    lang = (language or "ja").strip().lower()
    style = _cot_style_rules(lang)
    nudge = _cot_format_nudge(answer_format, lang)
    if lang == "vi":
        return (
            f"{style}\n"
            f"Câu hỏi (chỉ để đối chiếu, không viết lại trong độc thoại):\n{question}\n\n"
            f"Định dạng: {answer_format} — {nudge}\n"
            f"Trang PDF: {pages_ctx}. Trọng tâm minh chứng: {train_ev_str}.\n"
            f"Bản nháp hiện ~{token_count} token (quá dài). Rút gọn còn khoảng một nửa hoặc ít hơn, "
            "nhưng vẫn giữ mọi trang/số liệu quan trọng.\n"
            "【Bản nháp cần rút gọn】\n"
            f"{long_thinking}\n\n"
            "Xuất <think> đã rút gọn; không lặp đáp án đúng ở cuối."
        )
    return (
        f"{style}\n"
        f"設問（照合用・本文に書かない）:\n{question}\n\n"
        f"提出形式: {answer_format} — {nudge}\n"
        f"PDF ページ: {pages_ctx}。根拠の中心: {train_ev_str}。\n"
        f"下の草稿は約 {token_count} トークンで長すぎます。半分以下を目安に圧縮し、"
        "重要なページ番号・数値・比較は残してください。\n"
        "【圧縮対象の草稿】\n"
        f"{long_thinking}\n\n"
        "圧縮後の <think> のみ出力。正解の再掲はしない。"
    )


def build_cot_gen_input(processor: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    llm_in: Dict[str, Any] = {
        "prompt": apply_generation_prompt_without_thinking(processor, meta["sup_msgs"])
    }
    mm = prepare_mm_data(meta["sup_msgs"], meta["image_paths"])
    if mm:
        llm_in["multi_modal_data"] = mm
    return llm_in


def parse_cot_output(
    raw_text: str,
    ref_answer: str,
    tokenizer: Any,
) -> Tuple[str, int]:
    analysis = strip_echo_answer_from_thinking(
        extract_redacted_thinking(raw_text),
        ref_answer,
    )
    return analysis, count_text_tokens(tokenizer, analysis)


def select_shortest_valid_cot(
    candidates: List[Dict[str, Any]],
    *,
    token_threshold: int,
    min_chars: int,
) -> Dict[str, Any] | None:
    """在 token<=阈值 且字数达标的候选中，取 token 最短的一条。"""
    valid = [
        c
        for c in candidates
        if c["token_count"] <= token_threshold
        and len((c.get("analysis") or "").strip()) >= min_chars
    ]
    if not valid:
        return None
    return min(valid, key=lambda c: c["token_count"])


def select_longest_cot(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """取 token 最长的一条；若无非空正文则仍在全部候选中取最长。"""
    if not candidates:
        return None
    non_empty = [c for c in candidates if (c.get("analysis") or "").strip()]
    pool = non_empty or candidates
    return max(pool, key=lambda c: c["token_count"])


def collect_retry_candidates(
    retry_out: Any,
    row: Dict[str, Any],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    attempt_tokens: List[int] = []
    retry_candidates: List[Dict[str, Any]] = []
    for attempt_idx, output in enumerate(retry_out.outputs):
        analysis, tc = parse_cot_output(output.text, row["ref_answer"], tokenizer)
        attempt_tokens.append(tc)
        retry_candidates.append(
            {
                "analysis": analysis,
                "raw_text": output.text,
                "token_count": tc,
                "attempt_idx": attempt_idx,
            }
        )
    return retry_candidates, attempt_tokens


def needs_short_cot_retry(item: Dict[str, Any]) -> bool:
    return len((item.get("analysis") or "").strip()) < MIN_ANALYSIS_CHARS


def needs_long_cot_retry(item: Dict[str, Any]) -> bool:
    return bool(
        COT_REFINE_ENABLED
        and (item.get("analysis") or "").strip()
        and int(item["token_count"]) > COT_REFINE_TOKEN_THRESHOLD
    )


def build_cot_refine_inputs(
    processor: Any,
    pending: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """纯文本二次生成，无需多模态输入。"""
    refine_inputs: List[Dict[str, Any]] = []
    for item in pending:
        row = item["row"]
        lang = (row.get("language") or "ja").strip().lower()
        sys_msg = cot_refine_system_vi() if lang == "vi" else cot_refine_system_ja()
        user_block = cot_refine_user_block(
            question=row["question"],
            answer_format=row.get("answer_format", "string"),
            ref_answer=row["ref_answer"],
            train_evidence=row["train_evidence_pages"],
            ctx_page_nums=row["ctx_page_nums"],
            language=lang,
            long_thinking=item["analysis"],
            token_count=item["token_count"],
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_block},
        ]
        refine_inputs.append(
            {
                "prompt": apply_generation_prompt_without_thinking(
                    processor, messages
                )
            }
        )
    return refine_inputs


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


def strip_echo_answer_from_thinking(thinking: str, ref_answer: str) -> str:
    """去掉独白末尾对正解的一行复述（正解在 <answer> 里单独训练）。"""
    t = (thinking or "").strip()
    ref = (ref_answer or "").strip()
    if not t or not ref:
        return t
    lines = t.splitlines()
    drop_suffixes = (
        ref,
        f"結論：{ref}",
        f"結論:{ref}",
        f"したがって、{ref}",
        f"よって、{ref}",
    )
    changed = True
    while changed and lines:
        changed = False
        last = lines[-1].strip()
        if last in drop_suffixes or last == ref:
            lines.pop()
            changed = True
    out = "\n".join(lines).strip()
    if out.endswith(ref) and len(out) > len(ref):
        out = out[: -len(ref)].rstrip()
        out = re.sub(r"(結論[：:]?\s*)$", "", out).rstrip()
    return out or t


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
    tp_size = resolve_tensor_parallel_size()
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if tp_size > 1 and visible_gpus < tp_size:
        raise ValueError(
            f"TENSOR_PARALLEL_SIZE={tp_size} 超过可见 GPU 数 {visible_gpus}"
        )

    print(f"SEED={SEED} MODEL_PATH={MODEL_PATH}")
    print(
        f"TENSOR_PARALLEL_SIZE={tp_size} "
        f"(env={_TENSOR_PARALLEL_SIZE or 'auto'}, visible_gpus={visible_gpus})"
    )
    print(f"OUTPUT_TEST_DIR={md_root}")
    print(f"GEMINI_CSV={_resolve(GEMINI_CSV, script_dir)}")
    print(f"MIN_ANALYSIS_CHARS={MIN_ANALYSIS_CHARS}")
    print(
        f"COT_REFINE_ENABLED={COT_REFINE_ENABLED} "
        f"COT_REFINE_TOKEN_THRESHOLD={COT_REFINE_TOKEN_THRESHOLD} "
        f"COT_RETRY_SAMPLES={COT_RETRY_SAMPLES}"
    )

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
            missing_from_ctx=em.get("missing_from_ctx") or [],
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
        gen_inputs.append(build_cot_gen_input(processor, m))

    print(
        f"加载 VLM: {MODEL_PATH} "
        f"(tensor_parallel_size={tp_size}, "
        f"LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT})"
    )
    t_llm = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        seed=SEED,
    )
    print(f"[timing] LLM 加载: {time.perf_counter() - t_llm:.3f}s")

    print(f"生成 CoT {len(gen_inputs)} 条…")
    t_gen = time.perf_counter()
    outs = llm.generate(gen_inputs, sampling_params=SAMPLING_COT)
    print(f"[timing] llm.generate: {time.perf_counter() - t_gen:.3f}s")

    tokenizer = processor.tokenizer
    gen_results: List[Dict[str, Any]] = []
    long_retry_pending: List[Dict[str, Any]] = []
    short_retry_pending: List[Dict[str, Any]] = []
    for m, out in zip(meta, outs):
        row = m["row"]
        raw_text = out.outputs[0].text
        analysis, token_count = parse_cot_output(
            raw_text, row["ref_answer"], tokenizer
        )
        item = {
            "meta": m,
            "row": row,
            "raw_text": raw_text,
            "analysis": analysis,
            "token_count": token_count,
            "initial_token_count": token_count,
            "initial_analysis_chars": len(analysis.strip()),
            "retried": False,
            "retry_mode": None,
            "retry_attempt_tokens": [],
            "refined": False,
            "refine_raw_text": "",
        }
        gen_results.append(item)
        if needs_long_cot_retry(item):
            long_retry_pending.append(item)
        elif needs_short_cot_retry(item):
            short_retry_pending.append(item)

    if short_retry_pending:
        print(
            f"过短重试 {len(short_retry_pending)} 条：每条再生成 {COT_RETRY_SAMPLES} 次，"
            f"取最长稿（仍不足 {MIN_ANALYSIS_CHARS} 字也保留）…"
        )
        retry_inputs = [
            build_cot_gen_input(processor, item["meta"]) for item in short_retry_pending
        ]
        t_retry = time.perf_counter()
        retry_outs = llm.generate(retry_inputs, sampling_params=SAMPLING_RETRY)
        print(f"[timing] llm.generate(retry-short): {time.perf_counter() - t_retry:.3f}s")
        for item, retry_out in zip(short_retry_pending, retry_outs):
            row = item["row"]
            prev_chars = len((item.get("analysis") or "").strip())
            retry_candidates, attempt_tokens = collect_retry_candidates(
                retry_out, row, tokenizer
            )
            item["retry_attempt_tokens"] = attempt_tokens
            picked = select_longest_cot(retry_candidates)
            if picked is None:
                print(f"[warn] id={row['id']} 过短重试无输出，保留初稿")
                continue
            item["analysis"] = picked["analysis"]
            item["raw_text"] = picked["raw_text"]
            item["token_count"] = picked["token_count"]
            item["retried"] = True
            item["retry_mode"] = "short"
            new_chars = len(picked["analysis"].strip())
            meets_min = new_chars >= MIN_ANALYSIS_CHARS
            print(
                f"[retry-short] id={row['id']} chars {prev_chars} -> {new_chars} "
                f"({picked['token_count']} tokens, attempt "
                f"{picked['attempt_idx'] + 1}/{COT_RETRY_SAMPLES}, "
                f"attempts={attempt_tokens}"
                f"{', 仍偏短但保留' if not meets_min else ''})"
            )
            if needs_long_cot_retry(item):
                long_retry_pending.append(item)
                print(
                    f"[retry-short] id={row['id']} 重试后仍超长 "
                    f"({picked['token_count']} tokens)，加入过长重试"
                )

    still_refine_pending: List[Dict[str, Any]] = []
    if long_retry_pending:
        print(
            f"超长重试 {len(long_retry_pending)} 条：每条再生成 {COT_RETRY_SAMPLES} 次，"
            f"取 token<={COT_REFINE_TOKEN_THRESHOLD} 的最短合格稿…"
        )
        retry_inputs = [
            build_cot_gen_input(processor, item["meta"]) for item in long_retry_pending
        ]
        t_retry = time.perf_counter()
        retry_outs = llm.generate(retry_inputs, sampling_params=SAMPLING_RETRY)
        print(f"[timing] llm.generate(retry-long): {time.perf_counter() - t_retry:.3f}s")
        for item, retry_out in zip(long_retry_pending, retry_outs):
            row = item["row"]
            prev_tokens = item["token_count"]
            retry_candidates, attempt_tokens = collect_retry_candidates(
                retry_out, row, tokenizer
            )
            item["retry_attempt_tokens"] = attempt_tokens
            picked = select_shortest_valid_cot(
                retry_candidates,
                token_threshold=COT_REFINE_TOKEN_THRESHOLD,
                min_chars=MIN_ANALYSIS_CHARS,
            )
            if picked is not None:
                item["analysis"] = picked["analysis"]
                item["raw_text"] = picked["raw_text"]
                item["token_count"] = picked["token_count"]
                item["retried"] = True
                item["retry_mode"] = "long"
                print(
                    f"[retry-long] id={row['id']} {prev_tokens} -> {picked['token_count']} "
                    f"tokens (attempt {picked['attempt_idx'] + 1}/{COT_RETRY_SAMPLES}, "
                    f"attempts={attempt_tokens})"
                )
            else:
                still_refine_pending.append(item)
                print(
                    f"[retry-long] id={row['id']} {COT_RETRY_SAMPLES} 次均未达标 "
                    f"(attempts={attempt_tokens})，进入提炼"
                )

    if still_refine_pending:
        print(
            f"二次提炼 {len(still_refine_pending)} 条（>{COT_REFINE_TOKEN_THRESHOLD} tokens）…"
        )
        refine_inputs = build_cot_refine_inputs(processor, still_refine_pending)
        t_refine = time.perf_counter()
        refine_outs = llm.generate(refine_inputs, sampling_params=SAMPLING_REFINE)
        print(f"[timing] llm.generate(refine): {time.perf_counter() - t_refine:.3f}s")
        for item, ref_out in zip(still_refine_pending, refine_outs):
            row = item["row"]
            refine_raw = ref_out.outputs[0].text
            refined, new_tokens = parse_cot_output(
                refine_raw, row["ref_answer"], tokenizer
            )
            if not refined.strip():
                print(f"[warn] id={row['id']} 提炼为空，保留当前稿")
                continue
            prev_tokens = item["token_count"]
            if new_tokens >= prev_tokens:
                print(
                    f"[warn] id={row['id']} 提炼后仍 {new_tokens} tokens "
                    f"(>= {prev_tokens})，保留当前稿"
                )
                continue
            item["analysis"] = refined
            item["token_count"] = new_tokens
            item["refined"] = True
            item["refine_raw_text"] = refine_raw
            print(f"[refine] id={row['id']} {prev_tokens} -> {new_tokens} tokens")
    elif COT_REFINE_ENABLED and not long_retry_pending and not short_retry_pending:
        print(f"无需重试/提炼（阈值 {COT_REFINE_TOKEN_THRESHOLD} tokens）")

    del llm
    del processor
    release_torch_memory()

    out_jsonl = _resolve(OUT_JSONL, script_dir)
    skipped_path = _resolve(OUT_SKIPPED_JSONL, script_dir)
    raw_path = _resolve(OUT_RAW_JSON, script_dir)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    raw_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    sft_ids: set[str] = set()
    n_ok = 0
    n_skip = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout, open(
        skipped_path, "w", encoding="utf-8"
    ) as fskip:
        for item in gen_results:
            m = item["meta"]
            row = item["row"]
            analysis = item["analysis"]
            em = row.get("evidence_meta") or {}

            raw_rec = {
                "id": row["id"],
                "file_id": row["file_id"],
                "question": row.get("question"),
                "gemini_answer": row.get("ref_answer_raw"),
                "gemini_answer_norm": row["ref_answer"],
                "gemini_evidence_pages": row["ref_evidence_full"],
                "ref_answer_raw": row.get("ref_answer_raw"),
                "ref_answer_norm": row["ref_answer"],
                "ref_evidence_full": row["ref_evidence_full"],
                "train_evidence_pages": row["train_evidence_pages"],
                "ctx_page_nums": row["ctx_page_nums"],
                "missing_from_ctx": em.get("missing_from_ctx"),
                "used_ref_label_fallback": em.get("used_ref_label_fallback"),
                "used_ctx_fallback": em.get("used_ctx_fallback"),
                "raw_model_output": item["raw_text"],
                "retried": item.get("retried", False),
                "retry_mode": item.get("retry_mode"),
                "retry_attempt_tokens": item.get("retry_attempt_tokens") or None,
                "initial_analysis_chars": item.get("initial_analysis_chars"),
                "refined": item["refined"],
                "refine_raw_model_output": item.get("refine_raw_text") or None,
                "extracted_analysis_tokens": item["token_count"],
                "initial_analysis_tokens": item.get("initial_token_count"),
            }
            raw_records.append(raw_rec)

            skip_reason = ""
            if not (row.get("ref_answer") or "").strip():
                skip_reason = "empty_ref_answer"
            elif (
                len(analysis.strip()) < MIN_ANALYSIS_CHARS
                and item.get("retry_mode") != "short"
            ):
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
            sft_ids.add(row["id"])
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"已写入 SFT {n_ok} 条…")

    with open(raw_path, "w", encoding="utf-8") as rf:
        json.dump(raw_records, rf, ensure_ascii=False, indent=2)

    length_summary = build_length_summary(
        gen_results,
        sft_ids=sft_ids,
        refine_threshold=COT_REFINE_TOKEN_THRESHOLD,
    )
    stats_path = _resolve(OUT_LENGTH_STATS, script_dir)
    print_and_save_length_summary(length_summary, stats_path)

    print(f"已写入 SFT: {out_jsonl}（{n_ok} 条）")
    print(f"已跳过: {skipped_path}（{n_skip} 条）")
    print(f"已写入 raw: {raw_path}（{len(raw_records)} 条）")
    print(f"整页渲染缓存: {page_cache_dir}")
    print(f"[总耗时] {time.perf_counter() - t0:.3f}s")

    if n_ok == 0:
        raise RuntimeError("无有效 SFT 样本写入，请检查 MIN_ANALYSIS_CHARS 或模型输出。")


if __name__ == "__main__":
    main()
