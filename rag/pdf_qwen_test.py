#!/usr/bin/env python3
"""
test.csv 读题、从 output_test 加载 MinerU（content_list + 必选 origin.pdf）。
RAG 仅用于选页；作答上下文为 **content_list 阅读顺序下的图文交错**（正文文字 + chart/table 的 crop 图；type=image 仅为包裹内的文字说明），与 pdf_qwen_train 一致。
若 origin PDF 页数 ≤ RAG_TARGET_PAGES，则不做文档切块 embedding，直接按顺序纳入全部页，与 RAG 选出的页列表在下游共用同一套 prompt 构建逻辑。
其余文档：content_list 正文按滑窗切块后做 Embedding 检索；先取 top RAG_RERANK_TOP_K 块，若启用 Qwen3-Reranker 则重排后再凑满 RAG_TARGET_PAGES 个互异页。
RERANKER_MODEL_PATH 设为空字符串或 RAG_USE_RERANKER=0 可关闭精排。
本文件**不**跑 eval。结果：SUBMISSION_CSV（默认 submission.csv）与可选 OUT_JSON（默认 pdf_test_pred.json）。

依赖：transformers、torch、vllm、PyMuPDF（fitz）。检索为稠密向量，无 BM25。

uv pip install torch==2.10 torchvision torchaudio transformers==5.6.2 qwen_vl_utils vllm==0.19.1 pymupdf

所有 prompt 都会写个中文注释（与训练脚本一致）。
"""

import ast
import csv
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from utils import (
    EMBED_DOCUMENT_PREFIX,
    EMBED_QUERY_PREFIX,
    QwenEmbedder,
    QwenReranker,
    RAG_CHUNK_CHARS,
    RAG_CHUNK_OVERLAP,
    RAG_RERANK_TOP_K,
    RAG_TARGET_PAGES,
    _content_list_dir,
    apply_generation_prompt_with_brief_thinking,
    apply_generation_prompt_without_thinking,
    build_indexes_for_files,
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
    pages_from_rag_embed_sims,
    pdf_origin_meta,
    prepare_mm_data,
    rag_chunks_all_pages,
    prompt_answer_ja_interleaved,
    prompt_answer_vi_interleaved,
    release_torch_memory,
    set_random_seed,
    similarity_matrix,
    system_msg_ja,
    system_msg_vi,
)

# 作答与结构化解析共用同一模型，仅加载一次 vLLM（避免先大后小再卸再载）。
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/root/autodl-tmp/Qwen3.5-9B",
)
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/root/autodl-tmp/Qwen3-Embedding-0.6B",
)
_DEFAULT_RERANKER_PATH = "/root/autodl-tmp/Qwen3-Reranker-0.6B"
_rerank_path_env = os.environ.get("RERANKER_MODEL_PATH")
if _rerank_path_env is None:
    RERANKER_MODEL_PATH = _DEFAULT_RERANKER_PATH
else:
    RERANKER_MODEL_PATH = _rerank_path_env.strip()
RAG_USE_RERANKER = os.environ.get("RAG_USE_RERANKER", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
OUTPUT_TEST_DIR = os.environ.get("OUTPUT_TEST_DIR", "output_test")
TEST_CSV = os.environ.get("TEST_CSV", "test.csv")
SUBMISSION_CSV = os.environ.get("SUBMISSION_CSV", "submission.csv")
OUT_JSON = os.environ.get("OUT_JSON", "pdf_test_pred.json")
# random / numpy / torch；LLM/SamplingParams 传 seed=SEED；vLLM 与部分 CUDA 算子仍可能不完全可复现
SEED = int(os.environ.get("SEED", "42"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32000"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
# vLLM：单条 prompt 最多允许多少张图（图文交错时每页多图时需调大；过大易 OOM）
LIMIT_MM_IMAGES_PER_PROMPT = int(os.environ.get("LIMIT_MM_IMAGES_PER_PROMPT", "36"))
LIMIT_MM_PER_PROMPT = {"image": LIMIT_MM_IMAGES_PER_PROMPT, "video": 0}
# 文档索引在内存中的批大小：0 表示一次建完全部 file_id（与旧行为一致）；>0 时每批最多该数量篇，批内检索完即释放，降峰值内存
INDEX_BUILD_BATCH = int(os.environ.get("INDEX_BUILD_BATCH", "25"))

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


def format_evidence_column(pages: List[int]) -> str:
    """与 sample_submission.csv 中 evidence_page_number 列风格一致，如 [1]、[1,2]。"""
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


def main():
    t_script0 = time.perf_counter()
    print(f"[脚本开始] 机器本地时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    set_random_seed(SEED)
    print(f"随机种子 SEED={SEED}（可用环境变量 SEED 覆盖）")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = TEST_CSV if os.path.isabs(TEST_CSV) else os.path.join(script_dir, TEST_CSV)
    md_root = OUTPUT_TEST_DIR if os.path.isabs(OUTPUT_TEST_DIR) else os.path.join(
        script_dir, OUTPUT_TEST_DIR
    )

    with open(test_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    t_embedder = time.perf_counter()
    embedder = QwenEmbedder(EMBEDDING_MODEL_PATH)
    print(f"[timing] QwenEmbedder 加载: {time.perf_counter() - t_embedder:.3f}s")
    q_prev = (EMBED_QUERY_PREFIX[:48] + "…") if len(EMBED_QUERY_PREFIX) > 48 else EMBED_QUERY_PREFIX
    d_prev = (EMBED_DOCUMENT_PREFIX[:32] + "…") if len(EMBED_DOCUMENT_PREFIX) > 32 else EMBED_DOCUMENT_PREFIX
    print(
        f"加载 Embedding: {EMBEDDING_MODEL_PATH} "
        f"(向量维=模型全维 hidden_size={getattr(embedder.model.config, 'hidden_size', '?')}, "
        f"正文切块={RAG_CHUNK_CHARS}字重叠{RAG_CHUNK_OVERLAP}, "
        f"目标检索页数={RAG_TARGET_PAGES}, Rerank候选块topK={RAG_RERANK_TOP_K}, "
        f"索引批大小INDEX_BUILD_BATCH={INDEX_BUILD_BATCH or '全量'}; "
        f"查询前缀={q_prev!r} 文档前缀={d_prev!r})"
    )

    reranker: Optional[QwenReranker] = None
    if RAG_USE_RERANKER and RERANKER_MODEL_PATH:
        if os.path.isdir(RERANKER_MODEL_PATH):
            t_rr = time.perf_counter()
            reranker = QwenReranker(RERANKER_MODEL_PATH)
            print(f"[timing] QwenReranker 加载: {time.perf_counter() - t_rr:.3f}s | 路径={RERANKER_MODEL_PATH}")
        else:
            print(f"[warn] RERANKER_MODEL_PATH 非目录，跳过 rerank: {RERANKER_MODEL_PATH}")
    elif not RAG_USE_RERANKER:
        print("RAG_USE_RERANKER=0：仅用 Embedding 选页（不加载 Qwen3-Reranker）。")

    rows_by_fid: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        rows_by_fid[row["file_id"]].append(i)
    unique_fids = sorted(rows_by_fid.keys())
    batch_n = INDEX_BUILD_BATCH if INDEX_BUILD_BATCH > 0 else len(unique_fids)

    fid_pdf_meta: Dict[str, Tuple[str, int]] = {}
    short_fid_set: set[str] = set()
    for fid in unique_fids:
        op, pc = pdf_origin_meta(md_root, fid)
        if not op or pc <= 0:
            raise FileNotFoundError(
                f"缺少原始 PDF（需要 *_origin.pdf 且页数>0），无法继续: file_id={fid}，"
                f"已查 {md_root}/{fid}/vlm 与 hybrid_auto"
            )
        fid_pdf_meta[fid] = (op, pc)
        if pc <= RAG_TARGET_PAGES:
            short_fid_set.add(fid)
    n_short_fids = len(short_fid_set)
    n_long_fids = len(unique_fids) - n_short_fids
    print(
        f"按 PDF 页数分流：页数 ≤ {RAG_TARGET_PAGES} 的短文档 {n_short_fids} 篇跳过文档 embedding（全页进上下文）；"
        f"其余 {n_long_fids} 篇走 RAG 选页。"
    )

    rag_row_indices = [i for i, row in enumerate(rows) if row["file_id"] not in short_fid_set]
    print(f"预计算查询向量（仅 RAG 题目）共 {len(rag_row_indices)} 条…")
    t_qemb = time.perf_counter()
    if rag_row_indices:
        query_embs = embedder.encode_queries([rows[i]["question"] for i in rag_row_indices])
        row_to_qcol = {row_idx: j for j, row_idx in enumerate(rag_row_indices)}
        print(f"[timing] encode_queries {len(rag_row_indices)}条: {time.perf_counter() - t_qemb:.3f}s")
    else:
        query_embs = None
        row_to_qcol = {}
        print("[info] 全部题目所属 PDF 均为短文档，跳过 encode_queries。")
        print(f"[timing] encode_queries 跳过: {time.perf_counter() - t_qemb:.3f}s")

    print(
        f"按 file_id 分批轮询（唯一文档共 {len(unique_fids)} 个：其中 {n_long_fids} 个会在其所在批次做 embedding+RAG，"
        f"{n_short_fids} 个为短文档仅直出全页、不建向量索引；"
        f"INDEX_BUILD_BATCH={INDEX_BUILD_BATCH or '全量'}，每批≤{batch_n} 个 file_id）…"
    )
    rag_bundles: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    n_fids = len(unique_fids)
    total_batches = (n_fids + batch_n - 1) // batch_n
    t_rag_batches = time.perf_counter()
    for batch_i, start in enumerate(range(0, n_fids, batch_n), start=1):
        t_batch = time.perf_counter()
        fid_batch = unique_fids[start : start + batch_n]
        done_fids = min(start + len(fid_batch), n_fids)
        n_short_in_batch = sum(1 for f in fid_batch if f in short_fid_set)
        print(
            f"RAG 进度: 第 {batch_i}/{total_batches} 批，本批 {len(fid_batch)} 个 file_id "
            f"（短文档 {n_short_in_batch} 篇，累计 {done_fids}/{n_fids}）…"
        )
        short_triples: List[Tuple[str, str, int]] = []
        long_fids: List[str] = []
        for fid in fid_batch:
            op, pc = fid_pdf_meta[fid]
            if pc <= RAG_TARGET_PAGES:
                short_triples.append((fid, op, pc))
            else:
                long_fids.append(fid)

        t_after_short = time.perf_counter()
        for fid, op, pc in short_triples:
            chunks = rag_chunks_all_pages(pc)
            for row_idx in rows_by_fid[fid]:
                rag_bundles[row_idx] = {"chunks": chunks, "origin_pdf": op}

        if not long_fids:
            print(
                f"[timing] RAG batch {batch_i}/{total_batches}: "
                f"短文档全页直出 {t_after_short - t_batch:.3f}s | "
                f"本批无长文档，已跳过文档 embedding | 本批合计 {time.perf_counter() - t_batch:.3f}s"
            )
            release_torch_memory()
            continue

        index_batch = build_indexes_for_files(
            md_root,
            long_fids,
            embedder,
            timing_label=f"batch {batch_i}/{total_batches}（长文档 {len(long_fids)}篇）",
        )
        t_after_index = time.perf_counter()
        chunk_parts: List[np.ndarray] = []
        chunk_ranges: Dict[str, Tuple[int, int]] = {}
        batch_row_indices: List[int] = []
        chunk_offset = 0
        for fid in long_fids:
            emb = index_batch[fid]["chunk_embeddings"]
            chunk_parts.append(emb)
            next_offset = chunk_offset + int(emb.shape[0])
            chunk_ranges[fid] = (chunk_offset, next_offset)
            chunk_offset = next_offset
            batch_row_indices.extend(rows_by_fid[fid])

        assert query_embs is not None
        batch_chunk_embs = np.vstack(chunk_parts)
        batch_q_embs = query_embs[np.array([row_to_qcol[r] for r in batch_row_indices])]
        t_matmul0 = time.perf_counter()
        sims_matrix = similarity_matrix(batch_chunk_embs, batch_q_embs, embedder.device)
        t_matmul1 = time.perf_counter()
        row_to_col = {row_idx: col for col, row_idx in enumerate(batch_row_indices)}
        for fid in long_fids:
            index = index_batch[fid]
            c0, c1 = chunk_ranges[fid]
            row_indices = rows_by_fid[fid]
            op = index.get("origin_pdf")
            rag_slices = index["rag_slices"]
            pdf_n = int(index.get("pdf_page_count") or 0)
            for row_idx in row_indices:
                chunks = pages_from_rag_embed_sims(
                    sims_matrix[c0:c1, row_to_col[row_idx]],
                    rag_slices,
                    pdf_n,
                    rows[row_idx]["question"],
                    reranker,
                )
                rag_bundles[row_idx] = {
                    "chunks": chunks,
                    "origin_pdf": op,
                }
        t_pick1 = time.perf_counter()
        pick_label = "选页(embed+rerank)" if reranker is not None else "选页(embed)"
        short_dt = t_after_short - t_batch
        print(
            f"[timing] RAG batch {batch_i}/{total_batches}: "
            f"短文档直出 {short_dt:.3f}s | "
            f"vstack+取query向量 {t_matmul0 - t_after_index:.3f}s | "
            f"similarity_matrix{sims_matrix.shape} {t_matmul1 - t_matmul0:.3f}s | "
            f"{pick_label} {t_pick1 - t_matmul1:.3f}s | "
            f"本批(索引后→检索完成) {t_pick1 - t_after_index:.3f}s | "
            f"本批整批 {t_pick1 - t_batch:.3f}s"
        )
        del batch_chunk_embs, batch_q_embs, sims_matrix
        del index_batch
        release_torch_memory()

    print(
        f"[timing] RAG 全部批次（encode_queries 之后，索引+相似度+选页）: "
        f"{time.perf_counter() - t_rag_batches:.3f}s"
    )

    del embedder
    if reranker is not None:
        del reranker
    release_torch_memory()

    t_cl_load = time.perf_counter()
    cl_cache: Dict[str, List[Dict[str, Any]]] = {}
    cl_dir_cache: Dict[str, str] = {}
    for fid in unique_fids:
        cl_cache[fid] = load_content_list_raw(md_root, fid)
        cl_dir_cache[fid] = _content_list_dir(md_root, fid)
    print(f"[timing] 加载 content_list {len(unique_fids)} 篇: {time.perf_counter() - t_cl_load:.3f}s")

    t_llm_load = time.perf_counter()
    print(
        f"加载 VLM（作答 + 解析同权重）: {MODEL_PATH}（LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT}）"
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
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[timing] 生成模型 LLM + processor 加载: {time.perf_counter() - t_llm_load:.3f}s")

    ans_inputs: List[dict] = []
    gen_prompts: List[str] = []
    selected_pages_meta: List[List[Dict[str, Any]]] = []
    all_temp_png: List[str] = []

    t_row_prep = time.perf_counter()
    for row_idx, row in enumerate(rows):
        lang = (row.get("language") or "ja").strip().lower()
        q = row["question"]
        fid = row["file_id"]
        bundle = rag_bundles[row_idx]
        assert bundle is not None
        chunks = bundle["chunks"]
        op = bundle.get("origin_pdf")
        selected_pages_meta.append(
            [
                {
                    "page_idx": c["page_idx"],
                    "page_num": c["page_idx"] + 1,
                    "rag_score": c.get("rag_score", 0.0),
                }
                for c in chunks
            ]
        )
        afmt = row.get("answer_format", "string")

        if not chunks:
            raise ValueError(
                f"RAG 未选出任何在 PDF 范围内的页: id={row.get('id')} file_id={fid}"
            )

        page_indices = [c["page_idx"] for c in chunks]
        raw_cl = cl_cache[fid]
        content_dir = cl_dir_cache[fid]
        items = content_items_for_pages(raw_cl, page_indices)
        interleaved_parts, image_paths, temp_pngs = build_interleaved_content(
            items, content_dir, lang, origin_pdf=op,
        )
        all_temp_png.extend(temp_pngs)

        n_imgs = len(image_paths)
        if n_imgs > LIMIT_MM_IMAGES_PER_PROMPT:
            print(
                f"[warn] id={row.get('id')} 图文交错产生 {n_imgs} 张图，"
                f"超过 LIMIT_MM_IMAGES_PER_PROMPT={LIMIT_MM_IMAGES_PER_PROMPT}，截断多余图片"
            )
            keep_imgs = set(image_paths[:LIMIT_MM_IMAGES_PER_PROMPT])
            interleaved_parts = [
                p for p in interleaved_parts
                if p["type"] != "image" or p.get("image") in keep_imgs
            ]
            image_paths = image_paths[:LIMIT_MM_IMAGES_PER_PROMPT]

        if lang == "vi":
            preamble = interleaved_preamble_vi(page_indices)
            question_block = prompt_answer_vi_interleaved(q, afmt, "")
            sys_msg = system_msg_vi()
        else:
            preamble = interleaved_preamble_ja(page_indices)
            question_block = prompt_answer_ja_interleaved(q, afmt, "")
            sys_msg = system_msg_ja()

        msgs = build_interleaved_messages(sys_msg, preamble, interleaved_parts, question_block)
        prompt = apply_generation_prompt_with_brief_thinking(processor, msgs)
        gen_prompts.append(prompt)
        llm_in: Dict[str, Any] = {"prompt": prompt}
        mm = prepare_mm_data(msgs, image_paths)
        if mm:
            llm_in["multi_modal_data"] = mm
        ans_inputs.append(llm_in)

    print(
        f"[timing] 图文交错构建prompt {len(rows)}条: "
        f"{time.perf_counter() - t_row_prep:.3f}s"
    )
    print(f"回答生成 {len(ans_inputs)} 件…")
    t_ans_gen = time.perf_counter()
    ans_outs = llm.generate(ans_inputs, sampling_params=SAMPLING_ANS)
    print(f"[timing] llm.generate 回答: {time.perf_counter() - t_ans_gen:.3f}s")

    parse_inputs: List[Dict[str, Any]] = []
    for row, o in zip(rows, ans_outs):
        parse_msgs = build_messages(
            parse_system_msg(),
            parse_user_msg(row["question"], o.outputs[0].text),
            image_paths=None,
        )
        parse_prompt = apply_generation_prompt_without_thinking(processor, parse_msgs)
        parse_inputs.append({"prompt": parse_prompt})

    print(f"解析生成 {len(parse_inputs)} 件（复用同一 vLLM，不二次加载）…")
    t_parse_gen = time.perf_counter()
    parse_outs = llm.generate(parse_inputs, sampling_params=SAMPLING_PARSE)
    print(f"[timing] llm.generate 解析: {time.perf_counter() - t_parse_gen:.3f}s")

    list_fix_by_row: Dict[int, str] = {}
    list_fix_raw_output_by_row: Dict[int, str] = {}
    n_list_fix_fallback = 0
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
        print(f"列表题格式修复 {len(list_fix_jobs)} 条（复用同一 9B vLLM，不二次加载）…")
        list_fix_inputs: List[Dict[str, Any]] = []
        for job in list_fix_jobs:
            list_fix_msgs = build_messages(
                list_fix_system_msg(),
                list_fix_user_msg(job["question"], job["answer_format"], job["raw_answer"]),
                image_paths=None,
            )
            list_fix_prompt = apply_generation_prompt_without_thinking(processor, list_fix_msgs)
            list_fix_inputs.append({"prompt": list_fix_prompt})
        t_list_fix = time.perf_counter()
        list_fix_outs = llm.generate(list_fix_inputs, sampling_params=SAMPLING_LIST_FIX)
        print(f"[timing] llm.generate 列表题格式修复: {time.perf_counter() - t_list_fix:.3f}s")

        for job, out in zip(list_fix_jobs, list_fix_outs):
            raw_model_output = out.outputs[0].text
            list_fix_raw_output_by_row[job["row_idx"]] = raw_model_output
            items = parse_model_list(raw_model_output)
            if items is None:
                items = fallback_list(job["raw_answer"], job["language"])
                n_list_fix_fallback += 1
            list_fix_by_row[job["row_idx"]] = dump_list(items, job["language"])

        if n_list_fix_fallback:
            print(
                f"注意: {n_list_fix_fallback} 条列表题修复输出无法解析，"
                "已退回整段包成单元素列表。"
            )

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
        zip(rows, ans_outs, parse_outs, gen_prompts, selected_pages_meta)
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
                "language": (row.get("language") or "").strip(),
                "answer_format": (row.get("answer_format") or "").strip(),
                "input": model_input,
                "voted_pages": voted_pages,
                "answer": ans,
                "evidence": evidence,
                "raw_answer": raw,
                "raw_parse": parsed,
                "raw_list_fix": list_fix_raw_output_by_row.get(row_idx, ""),
            }
        )
        submission_rows.append(
            {
                "id": row["id"],
                "answer": ans,
                "evidence_page_number": format_evidence_column(evidence),
            }
        )

    out_path = OUT_JSON if os.path.isabs(OUT_JSON) else os.path.join(script_dir, OUT_JSON)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)

    print(f"已写入: {out_path}")

    sub_path = (
        SUBMISSION_CSV if os.path.isabs(SUBMISSION_CSV) else os.path.join(script_dir, SUBMISSION_CSV)
    )
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidence_page_number"])
        w.writeheader()
        w.writerows(submission_rows)

    print(f"已写入提交表（格式同 sample_submission.csv）: {sub_path}")

    t_elapsed = time.perf_counter() - t_script0
    print(f"[脚本结束] 机器本地时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"[脚本总耗时] {t_elapsed:.3f}s（{t_elapsed / 60.0:.2f} 分钟）")


if __name__ == "__main__":
    main()
