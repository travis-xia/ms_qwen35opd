#!/usr/bin/env python3
"""
仅跑 RAG 索引与选页：对 test.csv（或 QUESTIONS_CSV）中每道题，在其 file_id 对应文档上
选出 Top RAG_TARGET_PAGES（默认 6）页，写入 JSONL，供 pdf_qwen_test 等下游复用。

逻辑与 pdf_qwen_test.py 的 RAG 阶段一致：
  - PDF 页数 ≤ RAG_TARGET_PAGES：短文档，按顺序纳入全部页（不建 embedding 索引）
  - 否则：content_list 切块 → Embedding →（可选）Qwen3-Reranker → 互异页贪心

环境变量（与 pdf_qwen_test 对齐）：
  QUESTIONS_CSV / TEST_CSV、OUTPUT_TEST_DIR、EMBEDDING_MODEL_PATH、RERANKER_MODEL_PATH、
  RAG_USE_RERANKER、INDEX_BUILD_BATCH、RAG_TARGET_PAGES、RAG_RERANK_TOP_K、SEED

输出：
  RAG_PAGES_JSONL（默认 rag_top_pages.jsonl），每行一题 JSON。

运行：
  python3 pdf_qwen_rag_top_pages.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils import (
    EMBED_DOCUMENT_PREFIX,
    EMBED_QUERY_PREFIX,
    QwenEmbedder,
    QwenReranker,
    RAG_RERANK_TOP_K,
    RAG_TARGET_PAGES,
    build_indexes_for_files,
    pages_from_rag_embed_sims,
    pdf_origin_meta,
    rag_chunks_all_pages,
    release_torch_memory,
    set_random_seed,
    similarity_matrix,
)

EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3-Embedding-0.6B",
)
_DEFAULT_RERANKER_PATH = "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3-Reranker-0.6B"
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
QUESTIONS_CSV = os.environ.get(
    "QUESTIONS_CSV",
    os.environ.get("TEST_CSV", "test.csv"),
)
RAG_PAGES_JSONL = os.environ.get("RAG_PAGES_JSONL", "rag_top_pages.jsonl")
INDEX_BUILD_BATCH = int(os.environ.get("INDEX_BUILD_BATCH", "25"))
SEED = int(os.environ.get("SEED", "42"))


def chunks_to_page_records(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "page_idx": int(c["page_idx"]),
            "page_num": int(c["page_idx"]) + 1,
            "rag_score": float(c.get("rag_score", 0.0)),
        }
        for c in chunks
    ]


def row_record(
    row: Dict[str, str],
    *,
    pdf_page_count: int,
    short_document: bool,
    origin_pdf: Optional[str],
    selected_pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "file_id": row["file_id"],
        "question": row["question"],
        "language": (row.get("language") or "").strip(),
        "answer_format": (row.get("answer_format") or "").strip(),
        "pdf_page_count": pdf_page_count,
        "short_document": short_document,
        "rag_target_pages": RAG_TARGET_PAGES,
        "selected_pages": selected_pages,
        "origin_pdf": origin_pdf or "",
    }


def run_rag_for_rows(
    rows: List[Dict[str, str]],
    md_root: str,
    out_path: str,
) -> None:
    t0 = time.perf_counter()
    embedder = QwenEmbedder(EMBEDDING_MODEL_PATH)
    print(f"[timing] QwenEmbedder 加载: {time.perf_counter() - t0:.3f}s")
    q_prev = (EMBED_QUERY_PREFIX[:48] + "…") if len(EMBED_QUERY_PREFIX) > 48 else EMBED_QUERY_PREFIX
    d_prev = (EMBED_DOCUMENT_PREFIX[:32] + "…") if len(EMBED_DOCUMENT_PREFIX) > 32 else EMBED_DOCUMENT_PREFIX
    print(
        f"Embedding: {EMBEDDING_MODEL_PATH} | "
        f"目标页数={RAG_TARGET_PAGES} | Rerank候选块topK={RAG_RERANK_TOP_K} | "
        f"INDEX_BUILD_BATCH={INDEX_BUILD_BATCH or '全量'} | "
        f"查询前缀={q_prev!r} 文档前缀={d_prev!r}"
    )

    reranker: Optional[QwenReranker] = None
    if RAG_USE_RERANKER and RERANKER_MODEL_PATH:
        if os.path.isdir(RERANKER_MODEL_PATH):
            t_rr = time.perf_counter()
            reranker = QwenReranker(RERANKER_MODEL_PATH)
            print(f"[timing] QwenReranker 加载: {time.perf_counter() - t_rr:.3f}s | {RERANKER_MODEL_PATH}")
        else:
            print(f"[warn] RERANKER_MODEL_PATH 非目录，跳过 rerank: {RERANKER_MODEL_PATH}")
    elif not RAG_USE_RERANKER:
        print("RAG_USE_RERANKER=0：仅用 Embedding 选页。")

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
                f"缺少原始 PDF（需要 *_origin.pdf 且页数>0）: file_id={fid}，已查 {md_root}"
            )
        fid_pdf_meta[fid] = (op, pc)
        if pc <= RAG_TARGET_PAGES:
            short_fid_set.add(fid)

    n_short = len(short_fid_set)
    n_long = len(unique_fids) - n_short
    print(
        f"文档 {len(unique_fids)} 篇：短文档（≤{RAG_TARGET_PAGES} 页）{n_short} 篇全页直出；"
        f"长文档 {n_long} 篇建索引+RAG。"
    )

    rag_row_indices = [i for i, row in enumerate(rows) if row["file_id"] not in short_fid_set]
    t_qemb = time.perf_counter()
    if rag_row_indices:
        query_embs = embedder.encode_queries([rows[i]["question"] for i in rag_row_indices])
        row_to_qcol = {row_idx: j for j, row_idx in enumerate(rag_row_indices)}
        print(f"[timing] encode_queries {len(rag_row_indices)} 条: {time.perf_counter() - t_qemb:.3f}s")
    else:
        query_embs = None
        row_to_qcol = {}
        print("[info] 全部为短文档，跳过 encode_queries。")

    rag_bundles: List[Optional[Dict[str, Any]]] = [None] * len(rows)

    for fid in short_fid_set:
        op, pc = fid_pdf_meta[fid]
        chunks = rag_chunks_all_pages(pc)
        for row_idx in rows_by_fid[fid]:
            rag_bundles[row_idx] = {
                "chunks": chunks,
                "origin_pdf": op,
                "pdf_page_count": pc,
                "short_document": True,
            }

    long_fids = [fid for fid in unique_fids if fid not in short_fid_set]
    n_fids = len(long_fids)
    total_batches = (n_fids + batch_n - 1) // batch_n if n_fids else 0
    t_batches = time.perf_counter()

    for batch_i, start in enumerate(range(0, n_fids, batch_n), start=1):
        t_batch = time.perf_counter()
        fid_batch = long_fids[start : start + batch_n]
        print(
            f"索引批次 {batch_i}/{total_batches}，长文档 {len(fid_batch)} 篇 "
            f"（累计 {min(start + len(fid_batch), n_fids)}/{n_fids}）…"
        )
        index_batch = build_indexes_for_files(
            md_root,
            fid_batch,
            embedder,
            timing_label=f"batch {batch_i}/{total_batches}",
        )
        t_after_index = time.perf_counter()

        chunk_parts: List[np.ndarray] = []
        chunk_ranges: Dict[str, Tuple[int, int]] = {}
        batch_row_indices: List[int] = []
        chunk_offset = 0
        for fid in fid_batch:
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

        for fid in fid_batch:
            index = index_batch[fid]
            c0, c1 = chunk_ranges[fid]
            op = index.get("origin_pdf")
            rag_slices = index["rag_slices"]
            pdf_n = int(index.get("pdf_page_count") or 0)
            for row_idx in rows_by_fid[fid]:
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
                    "pdf_page_count": pdf_n,
                    "short_document": False,
                }

        pick_label = "embed+rerank" if reranker is not None else "embed"
        print(
            f"[timing] batch {batch_i}/{total_batches}: "
            f"建索引→检索 {time.perf_counter() - t_batch:.3f}s "
            f"(matmul {t_matmul1 - t_matmul0:.3f}s, {pick_label} {time.perf_counter() - t_matmul1:.3f}s)"
        )
        del batch_chunk_embs, batch_q_embs, sims_matrix, index_batch
        release_torch_memory()

    if n_long:
        print(f"[timing] 长文档全部批次: {time.perf_counter() - t_batches:.3f}s")

    del embedder
    if reranker is not None:
        del reranker
    release_torch_memory()

    missing = [i for i, b in enumerate(rag_bundles) if b is None]
    if missing:
        raise RuntimeError(f"RAG 未完成，缺 {len(missing)} 条，例如 row={missing[0]}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_written = 0
    t_write = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for row_idx, row in enumerate(rows):
            bundle = rag_bundles[row_idx]
            assert bundle is not None
            chunks = bundle["chunks"]
            if not chunks:
                raise ValueError(
                    f"未选出任何页: id={row.get('id')} file_id={row['file_id']}"
                )
            rec = row_record(
                row,
                pdf_page_count=int(bundle["pdf_page_count"]),
                short_document=bool(bundle["short_document"]),
                origin_pdf=bundle.get("origin_pdf"),
                selected_pages=chunks_to_page_records(chunks),
            )
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[timing] 写入 JSONL {n_written} 行: {time.perf_counter() - t_write:.3f}s")
    print(f"已写入: {out_path}")


def main() -> None:
    t_script = time.perf_counter()
    print(f"[开始] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    set_random_seed(SEED)
    print(f"SEED={SEED}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = (
        QUESTIONS_CSV
        if os.path.isabs(QUESTIONS_CSV)
        else os.path.join(script_dir, QUESTIONS_CSV)
    )
    md_root = (
        OUTPUT_TEST_DIR
        if os.path.isabs(OUTPUT_TEST_DIR)
        else os.path.join(script_dir, OUTPUT_TEST_DIR)
    )
    out_path = (
        RAG_PAGES_JSONL
        if os.path.isabs(RAG_PAGES_JSONL)
        else os.path.join(script_dir, RAG_PAGES_JSONL)
    )

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(
        f"QUESTIONS_CSV={csv_path} | OUTPUT_TEST_DIR={md_root} | "
        f"题目数={len(rows)} | 输出={out_path}"
    )

    run_rag_for_rows(rows, md_root, out_path)

    elapsed = time.perf_counter() - t_script
    print(f"[结束] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"[总耗时] {elapsed:.3f}s（{elapsed / 60.0:.2f} 分钟）")


if __name__ == "__main__":
    main()
