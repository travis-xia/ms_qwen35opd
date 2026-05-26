"""pdf_qwen_train.py 与 pdf_qwen_test.py 共用的 RAG、Embedding、可选 Qwen3-Reranker 精排选页、VLM 消息与解析工具。"""

import ast
import gc
import json
import os
import random
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from qwen_vl_utils import process_vision_info  # pyright: ignore[reportMissingImports]
except ImportError:
    process_vision_info = None


# RAG 文本切块：仅指正文按字符滑窗
RAG_CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "512"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
# 单个 MinerU 块在切块前的最大字符数（防止异常长块占满内存）
SOURCE_BLOCK_CHAR_LIMIT = int(os.environ.get("SOURCE_BLOCK_CHAR_LIMIT", "50000"))
EMBED_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "1536"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "120"))
# 为 1（默认）时：每次 encode 结束打印 tokenizer / H2D / forward / pool+L2 / cpu+numpy / vstack 累计耗时；设为 0 关闭
EMBED_TIMING_DETAIL = int(os.environ.get("EMBED_TIMING_DETAIL", "1"))
# Qwen3-Embedding 官方推荐（非对称检索）：查询侧 Instruct +「Query: 」+ 原题；设 EMBED_QUERY_PREFIX= 可退回无前缀
_DEFAULT_EMBED_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)
EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", _DEFAULT_EMBED_QUERY_PREFIX)
# 文档侧：官方允许不加前缀或加「Passage: 」；默认加前缀与查询 Instruct 对齐。设 EMBED_DOCUMENT_PREFIX= 可关闭
EMBED_DOCUMENT_PREFIX = os.environ.get("EMBED_DOCUMENT_PREFIX", "Passage: ")
# 未设置时自动选 cuda/cpu；可与 vLLM 错开，例如 EMBEDDING_DEVICE=cuda:1 或 cpu
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "").strip() or None

# 检索：按切片相似度从高到低扫，直到凑满 RAG_TARGET_PAGES 个互异页（全书不足则更少）
RAG_TARGET_PAGES = int(os.environ.get("RAG_TARGET_PAGES", "6"))
# Embedding 取 top-K 块进 Qwen3-Reranker 精排后再按块序选互异页；需 transformers>=4.51
RAG_RERANK_TOP_K = int(os.environ.get("RAG_RERANK_TOP_K", "48"))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "8192"))
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "32"))
# 与官方评测一致；可按任务用环境变量覆盖（建议英文 instruct）
_DEFAULT_RERANK_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_INSTRUCT = os.environ.get("RERANK_INSTRUCT", _DEFAULT_RERANK_INSTRUCT).strip() or _DEFAULT_RERANK_INSTRUCT
# 未设置时与 EMBEDDING_DEVICE 相同逻辑：空则 cuda/cpu
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "").strip() or None

INTERLEAVED_PAGE_BLOCK_LIMIT = int(os.environ.get("INTERLEAVED_PAGE_BLOCK_LIMIT", "10"))
INTERLEAVED_FALLBACK_DPI = int(os.environ.get("INTERLEAVED_FALLBACK_DPI", "100"))
# crop 图的像素上限：Qwen VL 按像素面积分 tile 算视觉 token，
# 默认 max_pixels=360000（约 600×600）既保证图表可读，又控制总 token
CROP_IMAGE_MAX_PIXELS = int(os.environ.get("CROP_IMAGE_MAX_PIXELS", "360000"))
CROP_IMAGE_MIN_PIXELS = int(os.environ.get("CROP_IMAGE_MIN_PIXELS", "3136"))


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """与 Qwen3-Embedding 官方 README 一致的 last-token 池化。"""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class QwenEmbedder:
    """Qwen3-Embedding：查询为 EMBED_QUERY_PREFIX + 文本；文档为 EMBED_DOCUMENT_PREFIX + 文本；全维 L2。"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        resolved = device or EMBEDDING_DEVICE or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if str(resolved).startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        self.device = torch.device(resolved)
        self.model.to(self.device)
        self.model.eval()

    def _encode_texts(self, texts: List[str], is_query: bool, batch_size: int) -> np.ndarray:
        out_list: List[np.ndarray] = []
        prefix = EMBED_QUERY_PREFIX if is_query else EMBED_DOCUMENT_PREFIX
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        detail = EMBED_TIMING_DETAIL != 0
        acc_tok = acc_h2d = acc_fwd = acc_post = acc_d2h = 0.0
        n_batches = 0

        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            n_batches += 1

            t0 = time.perf_counter()
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            t1 = time.perf_counter()
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            if detail and use_cuda:
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                if detail and use_cuda:
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                emb_f = emb.detach().float()
            t4 = time.perf_counter()
            out_list.append(emb_f.cpu().numpy())
            t5 = time.perf_counter()

            if detail:
                acc_tok += t1 - t0
                acc_h2d += t2 - t1
                acc_fwd += t3 - t2
                acc_post += t4 - t3
                acc_d2h += t5 - t4

        tv0 = time.perf_counter()
        stacked = np.vstack(out_list)
        tv1 = time.perf_counter()

        if detail:
            kind = "query" if is_query else "document"
            total_enc = acc_tok + acc_h2d + acc_fwd + acc_post + acc_d2h + (tv1 - tv0)
            print(
                f"[timing] encode 明细 ({kind}): tokenizer {acc_tok:.3f}s | "
                f"H2D {acc_h2d:.3f}s | forward {acc_fwd:.3f}s | pool+L2 {acc_post:.3f}s | "
                f"cpu+numpy {acc_d2h:.3f}s | vstack {tv1 - tv0:.3f}s | "
                f"累计(不含vstack) {acc_tok + acc_h2d + acc_fwd + acc_post + acc_d2h:.3f}s | "
                f"encode+vstack {total_enc:.3f}s | 批次×batch={n_batches}×{batch_size} 文本数={len(texts)}"
            )
        return stacked

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode_texts(texts, is_query=True, batch_size=min(EMBED_BATCH_SIZE, 8))

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode_texts(texts, is_query=False, batch_size=EMBED_BATCH_SIZE)


def _format_rerank_pair(instruction: str, query: str, doc: str) -> str:
    ins = instruction.strip() or _DEFAULT_RERANK_INSTRUCT
    return f"<Instruct>: {ins}\n<Query>: {query}\n<Document>: {doc}"


class QwenReranker:
    """
    Qwen3-Reranker（CausalLM）：对末位「yes」「no」logit 做 softmax 得 P(yes)，与 HF 模型卡 transformers 示例一致。
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        resolved = device or RERANKER_DEVICE or EMBEDDING_DEVICE or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if str(resolved).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        self.device = torch.device(resolved)
        self.model.to(self.device)
        self.model.eval()
        self.token_false_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.token_true_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        im_end = "<|im_end|>"
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".'
            f"{im_end}\n"
            "<|im_start|>user\n"
        )
        suffix = f"{im_end}\n<|im_start|>assistant\n\n"
        self._prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_length = RERANK_MAX_LENGTH

    @torch.no_grad()
    def score_documents(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> List[float]:
        ins = (instruction or RERANK_INSTRUCT).strip() or _DEFAULT_RERANK_INSTRUCT
        pairs = [_format_rerank_pair(ins, query, doc) for doc in documents]
        budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        budget = max(int(budget), 64)
        all_scores: List[float] = []
        for start in range(0, len(pairs), RERANK_BATCH_SIZE):
            batch_pairs = pairs[start : start + RERANK_BATCH_SIZE]
            enc = self.tokenizer(
                batch_pairs,
                padding=False,
                truncation="longest_first",
                max_length=budget,
                return_attention_mask=False,
            )
            input_ids_batch: List[List[int]] = []
            mask_batch: List[List[int]] = []
            for ids in enc["input_ids"]:
                full = self._prefix_ids + list(ids) + self._suffix_ids
                input_ids_batch.append(full)
                mask_batch.append([1] * len(full))
            batch = self.tokenizer.pad(
                {"input_ids": input_ids_batch, "attention_mask": mask_batch},
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits[:, -1, :]
            t_yes = logits[:, self.token_true_id].float()
            t_no = logits[:, self.token_false_id].float()
            two = torch.stack([t_no, t_yes], dim=1)
            prob_yes = F.log_softmax(two, dim=1)[:, 1].exp()
            all_scores.extend(prob_yes.cpu().tolist())
        return all_scores


def content_list_path(root: str, file_id: str) -> str:
    candidates = [
        os.path.join(root, file_id, "vlm", f"{file_id}_content_list.json"),
        os.path.join(root, file_id, "hybrid_auto", f"{file_id}_content_list.json"),
        os.path.join(root, file_id, "vlm", "content_list.json"),
        os.path.join(root, file_id, "hybrid_auto", "content_list.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "找不到 content_list，已尝试: " + " | ".join(candidates)
    )


def origin_pdf_path(root: str, file_id: str) -> Optional[str]:
    """MinerU 原始 PDF：{file_id}_origin.pdf（vlm / hybrid_auto）。"""
    candidates = [
        os.path.join(root, file_id, "vlm", f"{file_id}_origin.pdf"),
        os.path.join(root, file_id, "hybrid_auto", f"{file_id}_origin.pdf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _configure_mupdf_quiet() -> None:
    """默认关闭 MuPDF stderr 噪声（结构树损坏等仍可正常渲染）。"""
    if os.environ.get("MUPDF_DISPLAY_ERRORS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
    except Exception:
        pass


_configure_mupdf_quiet()


def strip_broken_pdf_struct_tree(doc: fitz.Document) -> bool:
    """
    移除损坏的 PDF StructTreeRoot，避免 MuPDF 报:
    format error: No common ancestor in structure tree
    参考: https://github.com/pymupdf/PyMuPDF/issues/4867
    """
    try:
        cat = doc.pdf_catalog()
        _key, val = doc.xref_get_key(cat, "StructTreeRoot")
        if val and val != "null":
            doc.xref_set_key(cat, "StructTreeRoot", "null")
            return True
    except Exception:
        pass
    return False


def open_pdf_for_render(pdf_path: str) -> fitz.Document:
    """打开 PDF 并修补已知结构树问题，供页渲染使用。"""
    doc = fitz.open(pdf_path)
    strip_broken_pdf_struct_tree(doc)
    return doc


def pdf_page_count(pdf_path: str) -> int:
    with open_pdf_for_render(pdf_path) as doc:
        return int(doc.page_count)


def pdf_origin_meta(root: str, file_id: str) -> Tuple[Optional[str], int]:
    """若存在 origin PDF 则返回 (绝对路径, 总页数)；否则 (None, 0)。"""
    p = origin_pdf_path(root, file_id)
    if not p:
        return None, 0
    return p, pdf_page_count(p)


def render_pdf_pages_to_png_paths(
    pdf_path: str, page_indices: List[int], dpi: int
) -> List[str]:
    """
    将 PDF 中按 MinerU 的 0-based page_idx 的页面渲染为临时 PNG 路径列表（与 page_indices 顺序一致）。
    页码越界则跳过该页（不插入占位，与调用方约定：page_indices 应已有效）。
    """
    out: List[str] = []
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    doc = open_pdf_for_render(pdf_path)
    try:
        n = int(doc.page_count)
        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= n:
                continue
            page = doc.load_page(page_idx)
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
            except RuntimeError:
                pix = page.get_pixmap(matrix=mat, alpha=False, annots=False)
            fd, path = tempfile.mkstemp(suffix=".png", prefix="origin_pdf_")
            os.close(fd)
            pix.save(path)
            out.append(path)
    finally:
        doc.close()
    return out


def _html_to_plain_text(html: str) -> str:
    """MinerU table 的 table_body 常为 HTML，去掉标签后供检索切块。"""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(html))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def block_search_text(item: Dict[str, Any]) -> str:
    t = item.get("type", "")
    parts: List[str] = []
    if t == "chart":
        parts.append(str(item.get("content", "")))
        for k in ("chart_caption", "chart_footnote"):
            v = item.get(k)
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
            elif v:
                parts.append(str(v))
    elif t == "table":
        for k in ("table_caption", "table_footnote"):
            v = item.get(k)
            if isinstance(v, list):
                parts.extend(str(x) for x in v if str(x).strip())
            elif v:
                parts.append(str(v))
        body = item.get("table_body") or item.get("text") or item.get("content")
        if body:
            parts.append(_html_to_plain_text(str(body)))
    else:
        tx = item.get("text") or item.get("content")
        if tx:
            parts.append(str(tx))
    return "\n".join(parts).strip()


def load_blocks(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        body = block_search_text(item)
        if not body or body.strip() == "[Non-Text]":
            continue
        out.append(
            {
                "list_idx": i,
                "page_idx": int(item.get("page_idx", 0)),
                "type": item.get("type", ""),
                "text": body[:SOURCE_BLOCK_CHAR_LIMIT],
            }
        )
    return out


def sliding_char_chunks(text: str, size: int, overlap: int) -> List[str]:
    """字符滑窗切块：每块至多 size 字符，相邻块重叠 overlap（步长 size - overlap）。"""
    text = text.strip()
    if not text:
        return []
    if overlap >= size:
        overlap = max(0, size - 1)
    stride = size - overlap
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        piece = text[start : start + size]
        out.append(piece)
        if start + size >= n:
            break
        start += stride
    return out


def blocks_to_rag_slices(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把一个 MinerU 块正文切成多条检索单元（512 字、重叠 64），每条保留 page_idx 汇总到页。"""
    rag_rows: List[Dict[str, Any]] = []
    for b in blocks:
        page_idx = b["page_idx"]
        chunks = sliding_char_chunks(b["text"], RAG_CHUNK_CHARS, RAG_CHUNK_OVERLAP)
        for si, chunk in enumerate(chunks):
            rag_rows.append(
                {
                    "page_idx": page_idx,
                    "text": chunk,
                    "block_list_idx": b["list_idx"],
                    "slice_idx": si,
                }
            )
    return rag_rows


def build_index_for_file(
    md_root: str, file_id: str, embedder: QwenEmbedder
) -> Dict[str, Any]:
    return build_indexes_for_files(md_root, [file_id], embedder)[file_id]


def build_indexes_for_files(
    md_root: str,
    file_ids: List[str],
    embedder: QwenEmbedder,
    *,
    timing_label: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    批量构建同一批 file_id 的索引：I/O 与切块仍逐文档做，但 embedding 合并成一个大队列，
    避免小文档逐个 encode_documents 导致 GPU 利用不足。
    """
    payloads: Dict[str, Dict[str, Any]] = {}
    all_texts: List[str] = []
    t_prep = time.perf_counter()
    for file_id in file_ids:
        path = content_list_path(md_root, file_id)
        blocks = load_blocks(path)
        if not blocks:
            raise ValueError(f"content_list 无有效文本块: {path}")
        rag_slices = blocks_to_rag_slices(blocks)
        if not rag_slices:
            raise ValueError(f"滑窗切块后无检索单元: {path}")
        op, pc = pdf_origin_meta(md_root, file_id)
        if not op or pc <= 0:
            raise FileNotFoundError(
                f"缺少原始 PDF（需要 *_origin.pdf 且页数>0），无法继续: file_id={file_id}，"
                f"已查 {md_root}/{file_id}/vlm 与 hybrid_auto"
            )
        start = len(all_texts)
        all_texts.extend(r["text"] for r in rag_slices)
        payloads[file_id] = {
            "blocks": blocks,
            "rag_slices": rag_slices,
            "origin_pdf": op,
            "pdf_page_count": pc,
            "emb_start": start,
            "emb_end": len(all_texts),
        }

    t_emb0 = time.perf_counter()
    all_embeddings = embedder.encode_documents(all_texts)
    t_emb1 = time.perf_counter()
    lb = timing_label.strip() or "索引"
    print(
        f"[timing] {lb}: 预处理(I/O+切块) {t_emb0 - t_prep:.3f}s | "
        f"encode_documents {len(all_texts)}切片 {t_emb1 - t_emb0:.3f}s | 合计 {t_emb1 - t_prep:.3f}s"
    )
    indexes: Dict[str, Dict[str, Any]] = {}
    for file_id, payload in payloads.items():
        start = int(payload.pop("emb_start"))
        end = int(payload.pop("emb_end"))
        indexes[file_id] = {
            **payload,
            "chunk_embeddings": all_embeddings[start:end],
        }
    return indexes


def _pages_from_sims(
    sims: np.ndarray,
    rag_slices: List[Dict[str, Any]],
    pdf_n: int,
) -> List[Dict[str, Any]]:
    """sims: (n_chunks,) 与 rag_slices 逐行对齐。返回按原 PDF page_idx 升序（阅读顺序），便于 preamble 与正文一致。"""
    n = int(sims.shape[0])
    if n == 0 or pdf_n <= 0:
        return []
    order = np.argsort(-sims)
    picked: List[Dict[str, Any]] = []
    seen_pages: set[int] = set()
    for pos in order:
        if len(picked) >= RAG_TARGET_PAGES:
            break
        i = int(pos)
        page_idx = rag_slices[i]["page_idx"]
        if page_idx in seen_pages:
            continue
        if page_idx < 0 or page_idx >= pdf_n:
            continue
        seen_pages.add(page_idx)
        picked.append(
            {
                "page_idx": page_idx,
                "type": "page",
                "rag_score": float(sims[i]),
            }
        )
    return sorted(picked, key=lambda d: int(d["page_idx"]))


def rag_chunks_all_pages(pdf_n: int) -> List[Dict[str, Any]]:
    """总页数不超过 RAG_TARGET_PAGES 时整本进上下文；每项形状与 _pages_from_sims 返回一致。"""
    if pdf_n <= 0:
        return []
    return [
        {"page_idx": i, "type": "page", "rag_score": 1.0} for i in range(pdf_n)
    ]


def pages_from_rag_embed_sims(
    embed_sims: np.ndarray,
    rag_slices: List[Dict[str, Any]],
    pdf_n: int,
    query: str,
    reranker: Optional[QwenReranker],
    *,
    rerank_top_k: int = RAG_RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    先用 embedding 相似度取 top rerank_top_k 个块，再用 Qwen3-Reranker 按 P(yes) 重排后做互异页贪心（同 _pages_from_sims 遍历逻辑）。
    reranker 为 None 或 query 为空时退化为 _pages_from_sims(embed_sims, ...)。
    返回的页块按 page_idx 升序排列（原 PDF 阅读顺序）。
    """
    if reranker is None or not (query or "").strip():
        return _pages_from_sims(embed_sims, rag_slices, pdf_n)
    n = int(embed_sims.shape[0])
    if n == 0 or pdf_n <= 0:
        return []
    k = min(max(1, int(rerank_top_k)), n)
    pool = np.argsort(-embed_sims)[:k].astype(int).tolist()
    docs = [rag_slices[i]["text"] for i in pool]
    rr_scores = reranker.score_documents(query, docs)
    score_by_chunk: Dict[int, float] = {
        pool[j]: float(rr_scores[j]) for j in range(len(pool))
    }
    order = sorted(pool, key=lambda idx: -score_by_chunk[idx])
    picked: List[Dict[str, Any]] = []
    seen_pages: set[int] = set()
    for i in order:
        if len(picked) >= RAG_TARGET_PAGES:
            break
        page_idx = rag_slices[i]["page_idx"]
        if page_idx in seen_pages:
            continue
        if page_idx < 0 or page_idx >= pdf_n:
            continue
        seen_pages.add(page_idx)
        picked.append(
            {
                "page_idx": page_idx,
                "type": "page",
                "rag_score": score_by_chunk[i],
            }
        )
    return sorted(picked, key=lambda d: int(d["page_idx"]))


def retrieve_pages(index: Dict[str, Any], q_emb: np.ndarray) -> List[Dict[str, Any]]:
    """
    沿相似度从高到低遍历全部切片，直到纳入 RAG_TARGET_PAGES 个不同 PDF 页（0-based 须在 origin 页数内）。
    仅返回页码与分数；正文一律由后续 PDF 渲染图提供。返回列表按 page_idx 升序（原 PDF 阅读顺序）。
    q_emb：单条查询向量（与 build_index 时 embedder.encode_queries 一致），由调用方预计算。
    """
    chunk_embs: np.ndarray = index["chunk_embeddings"]
    q = np.asarray(q_emb, dtype=chunk_embs.dtype).reshape(-1)
    sims = chunk_embs @ q
    return _pages_from_sims(sims, index["rag_slices"], int(index.get("pdf_page_count") or 0))


def retrieve_pages_batch(
    index: Dict[str, Any], q_embs: np.ndarray
) -> List[List[Dict[str, Any]]]:
    """
    同一文档索引下对多条查询一次性算 chunk–query 相似度（chunk_embs @ q_embs.T），
    再逐条做选页；避免重复 large matmul 的 Python 调度开销，数值与逐条 retrieve_pages 一致。
    每条返回的页块按 page_idx 升序。
    q_embs: (Q, dim) 或单条 (dim,)。
    """
    chunk_embs: np.ndarray = index["chunk_embeddings"]
    q = np.asarray(q_embs, dtype=chunk_embs.dtype)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    sims = chunk_embs @ q.T
    rag_slices: List[Dict[str, Any]] = index["rag_slices"]
    pdf_n = int(index.get("pdf_page_count") or 0)
    return [_pages_from_sims(sims[:, j], rag_slices, pdf_n) for j in range(sims.shape[1])]


def similarity_matrix(
    chunk_embs: np.ndarray,
    q_embs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """一次性计算 chunk-query 大相似度矩阵；CUDA 可用时把 matmul 放到 GPU 上做。"""
    if chunk_embs.size == 0 or q_embs.size == 0:
        return np.empty((int(chunk_embs.shape[0]), int(q_embs.shape[0])), dtype=np.float32)
    q = np.asarray(q_embs, dtype=chunk_embs.dtype)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if device.type == "cuda":
        with torch.no_grad():
            c_t = torch.as_tensor(chunk_embs, dtype=torch.float32, device=device)
            q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
            return (c_t @ q_t.T).cpu().numpy()
    return chunk_embs @ q.T


def format_pdf_rag_preamble(chunks: List[Dict[str, Any]], lang: str) -> str:
    """
    说明「下方图像 = 原 PDF 某页切图（1 始）」；chunks 宜为按 page_idx 升序（与 pdf_qwen_test 选页返回一致）。
    """
    if lang == "vi":
        return (
            "Bên dưới nối tiếp là từng vùng ảnh được cắt từ PDF gốc theo các trang đã tìm được, "
            "mỗi cặp (Ảnh k, trang p) đều tính từ 1; các trang liệt kê theo thứ tự số trang trong PDF (tăng dần).\n"
            "Chỉ dựa vào nội dung hình, không dùng kiến thức bên ngoài.\n\n"
            + "\n".join(
                f"- Ảnh thứ {i}: trang {c['page_idx'] + 1} (PDF, đếm từ 1)."
                for i, c in enumerate(chunks, start=1)
            )
            + "\n"
        )
    return (
        "下に続く画像は、検索にヒットしたページを**原PDF**から抜粋したものです。"
        "解答は**これらの画像に写っている内容のみ**に厳密に従い、他の知識は用いないでください。\n\n"
        "【ページ番号：PDF では 1 ページ目を 1 と数える。以下の並び＝原PDFのページ番号の昇順（読み順）】\n"
        + "\n".join(
            f"- 画像{j} → PDF ページ {c['page_idx'] + 1}"
            for j, c in enumerate(chunks, start=1)
        )
        + "\n"
    )


def prompt_answer_ja_from_pdf(
    question: str, answer_format: str, pdf_preamble: str
) -> str:
    """第2段·日语 + PDF 页图：先输出答案，再输出证据页码列表。"""
    fmt = {
        "string": "最終回答は1行の短文のみ。理由や前置きは書かない。",
        "number": "問題が求める数値・単位のみを1行で。",
        "ordered_list": "答えは順序付きの複数項目。1行のPythonリストリテラルで、例: ['第一','第二']。",
        "unordered_list": "答えは複数項目。1行のPythonリストリテラルで、例: ['甲','乙']。順不同。",
    }.get(answer_format, "最終回答は1行の短文のみ。")

    return (
        f"{pdf_preamble}\n"
        f"設問：\n{question}\n\n"
        f"出力要件：{fmt}\n"
        "画像は上から順に提示されています。まず最終回答だけを <answer> に入れてください。"
        "その後、回答の根拠が写っているPDFページ番号（1始まり）だけをPythonの整数リストで "
        "<evidence> に入れてください。根拠ページは提示された画像に対応するページから選び、"
        "不確かなページは入れないでください。説明文やコードフェンスは不要です。\n"
        "<answer>ここに最終回答</answer>\n"
        "<evidence>[1, 2]</evidence>\n"
    )


def prompt_answer_vi_from_pdf(
    question: str, answer_format: str, pdf_preamble: str
) -> str:
    """第2段·越南语 + PDF 页图：先输出答案，再输出证据页码列表。"""
    fmt = {
        "string": "Chỉ một dòng câu trả lời cuối cùng, không giải thích.",
        "number": "Một dòng: chỉ số/đơn vị theo yêu cầu đề bài.",
        "ordered_list": "Nhiều mục có thứ tự: một dòng literal list Python, ví dụ ['một','hai'].",
        "unordered_list": "Nhiều mục: một dòng literal list Python, ví dụ ['A','B'], thứ tự tự do.",
    }.get(answer_format, "Chỉ một dòng câu trả lời cuối cùng.")

    return (
        f"{pdf_preamble}\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Yêu cầu định dạng: {fmt}\n"
        "Các hình theo thứ tự từ trên xuống. Trước hết, chỉ đặt câu trả lời cuối cùng vào <answer>. "
        "Sau đó, đặt các số trang PDF chứa bằng chứng cần thiết (đếm từ 1) vào "
        "<evidence> dưới dạng một Python list các số nguyên. Chỉ chọn trong các trang "
        "tương ứng với hình/đoạn đã cung cấp, không thêm trang không chắc chắn. Không thêm giải thích hay markdown.\n"
        "<answer>câu trả lời cuối cùng</answer>\n"
        "<evidence>[1, 2]</evidence>\n"
    )


def system_msg_ja() -> str:
    """日语 system：严格依据所给材料回答。"""
    # 中文翻译：
    # 你是一位亲切且有帮助、严格基于文档作答的助理。
    # 你可以进行必要的内部思考，但不要反复重审同一假设；
    # 一旦确认了证据页，就简洁地给出结论。
    # 最终输出只包含指定标签。
    return (
        "あなたは文書に厳密に基づいて回答する親切で有用なアシスタントです。"
        "内部では必要なだけ考えてよいですが、同じ仮説の再検討を繰り返さず、"
        "根拠ページを一度確認したら簡潔に結論へ進んでください。"
        "最終出力は指定されたタグだけにしてください。"
    )


def system_msg_vi() -> str:
    """越南语 system：仅根据所给语料回答。"""
    # 中文翻译：
    # 你是一位有帮助的助理，只能基于给定语料回答。
    # 可以在内部进行必要推理，但不要反复纠结同一假设或过度自检；
    # 一旦确定证据页，直接给出简洁结论。
    # 最终输出只包含要求的标签。
    return (
        "Bạn là trợ lý hữu ích, chỉ trả lời dựa trên ngữ liệu được cung cấp. "
        "Bạn có thể suy luận nội bộ khi cần, nhưng không lặp lại cùng một giả thuyết hay tự kiểm tra quá nhiều lần; "
        "sau khi xác định trang bằng chứng, hãy đi thẳng đến kết luận ngắn gọn. "
        "Đầu ra cuối cùng chỉ gồm đúng các thẻ được yêu cầu."
    )


def build_messages(
    system: str,
    user_text: str,
    image_paths: Optional[List[str]] = None,
) -> List[dict]:
    """
    组装 Qwen 对话：system + user；若有 image_paths 则多图 + 文本（与 qwen35 / qwen_train_chunk 一致）。
    中文注释：有图时 user 为多段 content
    """
    if image_paths:
        content: List[Dict[str, Any]] = []
        for p in image_paths:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def prepare_mm_data(messages: List[dict], image_paths: List[str]) -> Dict[str, Any]:
    if not image_paths:
        return {}
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        from PIL import Image

        image_inputs = [Image.open(p).convert("RGB") for p in image_paths]
        video_inputs = None
    mm_data: Dict[str, Any] = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs
    return mm_data


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


def fallback_list(raw_answer: str, language: str) -> List[str]:
    text = (raw_answer or "").strip()
    if is_none_placeholder(text):
        return [no_answer_text(language)]
    return [text] if text else [no_answer_text(language)]


def sanitize_scalar_answer(answer: str, answer_format: str, language: str) -> str:
    """与 pdf_qwen_infer_from_rag / pdf_qwen_test 提交 CSV 的 string/number 规范化一致。"""
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


def normalize_submission_answer(
    raw: str,
    answer_format: str,
    language: str,
) -> str:
    """将 Gemini/模型 raw 答案规范为与 submission CSV 一致的 answer 字符串。"""
    afmt = (answer_format or "string").strip()
    lang = (language or "ja").strip()
    raw_s = (raw or "").strip()
    if afmt in ("unordered_list", "ordered_list"):
        parsed = try_parse_list(raw_s)
        if parsed is None:
            try:
                jv = json.loads(raw_s)
                if isinstance(jv, list):
                    parsed = jv
            except json.JSONDecodeError:
                parsed = None
        if parsed is not None:
            return dump_list(parsed, lang)
        return dump_list(fallback_list(raw_s, lang), lang)
    return sanitize_scalar_answer(raw_s, afmt, lang)


def resolve_training_evidence_pages(
    ref_pages: List[int],
    ctx_page_nums: List[int],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    训练用证据页（固定题库记忆向）：
      1) 非空则 ref∩ctx（保持 ref 顺序）；
      2) 交集为空则用答案/Gemini 标签中的完整 ref 列表（保证与提交一致）；
      3) ref 也为空时才用 RAG 上下文全部页；最后兜底 [1]。
    """
    ctx_sorted = sorted({int(p) for p in ctx_page_nums if int(p) > 0})
    ref_clean: List[int] = []
    seen: set[int] = set()
    for p in ref_pages:
        n = int(p)
        if n > 0 and n not in seen:
            seen.add(n)
            ref_clean.append(n)
    ctx_set = set(ctx_sorted)
    intersection = [p for p in ref_clean if p in ctx_set]
    missing = [p for p in ref_clean if p not in ctx_set]
    used_ref_label_fallback = False
    used_ctx_fallback = False
    if intersection:
        train = intersection
    elif ref_clean:
        train = list(ref_clean)
        used_ref_label_fallback = True
    elif ctx_sorted:
        train = list(ctx_sorted)
        used_ctx_fallback = True
    else:
        train = [1]
        used_ctx_fallback = True
    return train, {
        "ref_evidence_full": ref_clean,
        "ctx_page_nums": ctx_sorted,
        "missing_from_ctx": missing,
        "used_ref_label_fallback": used_ref_label_fallback,
        "used_ctx_fallback": used_ctx_fallback,
    }


def parse_answer_tag(text: str) -> str:
    """只解析 <answer>；无标签时去掉证据标签，避免污染原答案评测。"""
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if ans_match:
        return ans_match.group(1).strip()
    cleaned = re.sub(
        r"<evidence>\s*.*?\s*</evidence>",
        "",
        text,
        flags=re.S | re.I,
    )
    cleaned = re.sub(
        r"<evidence_page_number>\s*.*?\s*</evidence_page_number>",
        "",
        cleaned,
        flags=re.S | re.I,
    )
    return cleaned.strip()


def parse_evidence(text: str) -> List[int]:
    """解析模型输出的 <evidence>（兼容旧 <evidence_page_number>）。"""
    ev_match = re.search(
        r"<evidence>\s*(.*?)\s*</evidence>",
        text,
        flags=re.S | re.I,
    )
    if not ev_match:
        ev_match = re.search(
            r"<evidence_page_number>\s*(.*?)\s*</evidence_page_number>",
            text,
            flags=re.S | re.I,
        )
    if not ev_match:
        ev_match = re.search(
            r'"evidence"\s*:\s*(\[[^\]]*\])',
            text,
            flags=re.S | re.I,
        )
    if not ev_match:
        ev_match = re.search(
            r'"evidence_page_number"\s*:\s*(\[[^\]]*\])',
            text,
            flags=re.S | re.I,
        )
    if not ev_match:
        return []

    raw = ev_match.group(1).strip()
    values: List[Any]
    try:
        parsed = ast.literal_eval(raw)
        values = parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        values = re.findall(r"\d+", raw)

    out: List[int] = []
    seen: set[int] = set()
    for v in values:
        try:
            page_num = int(v)
        except (TypeError, ValueError):
            continue
        if page_num <= 0 or page_num in seen:
            continue
        seen.add(page_num)
        out.append(page_num)
    return out


def apply_generation_prompt_with_brief_thinking(processor: Any, msgs: List[dict]) -> str:
    """开启 Qwen3 «思考»，但由采样预算和 system prompt 限制反复长推理。"""
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return processor.apply_chat_template(msgs, enable_thinking=True, **kw)
    except TypeError:
        return processor.apply_chat_template(msgs, **kw)


def apply_generation_prompt_without_thinking(processor: Any, msgs: List[dict]) -> str:
    """关闭思考模式，直接生成结构化解析结果。"""
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return processor.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return processor.apply_chat_template(msgs, **kw)


def parse_system_msg() -> str:
    return (
        "你是一个信息抽取与问答助手。"
        "给你 question 和 raw_answer，请提取并归一化答案。"
        "只输出两个标签：<answer>...</answer> 与 <evidence>[页码整数数组]</evidence>。"
        "若页码不确定则输出空数组 []。"
    )


def parse_user_msg(question: str, raw_answer: str) -> str:
    return (
        "请从 raw_answer 中提取最终答案与证据页码。\n"
        "要求：\n"
        "1) answer 保留原意，去掉多余解释。\n"
        "2) evidence 仅保留正整数页码，去重，按出现顺序。\n"
        "3) 如果实在没有符合格式的内容，请你根据问题和原始回答的分析自行回答，直接给出answer和evidence。\n"
        "4) 禁止输出除两个标签外的任何内容。\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<raw_answer>\n{raw_answer}\n</raw_answer>"
    )


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 图文交错：content_list 文字 + chart/table 的 crop 图；image 仅文字块（IMAGE_AS_TEXT 包裹）
# ---------------------------------------------------------------------------

_INTERLEAVED_SKIP_TYPES = {"header", "footer", "page_number", "aside_text"}


def _image_text_block_wrappers(lang: str) -> Tuple[str, str]:
    """
    标记「以下为 PDF 中 image 类块对应的文字信息、不附带像素图」的包裹字符串。
    起止标记便于模型识别边界；语言与 interleaved 页眉一致（ja / vi）。
    """
    if (lang or "").strip().lower() == "vi":
        return (
            "<<<IMAGE_AS_TEXT_BLOCK_BEGIN>>> "
            "(Khối mô tả hình ảnh trong PDF — chỉ nội dung chữ, không kèm file ảnh)",
            "<<<IMAGE_AS_TEXT_BLOCK_END>>>",
        )
    return (
        "<<<IMAGE_AS_TEXT_BLOCK_BEGIN>>> "
        "（PDF 内の画像ブロックに対応する文字情報のみ／画像ピクセルは添付していません）",
        "<<<IMAGE_AS_TEXT_BLOCK_END>>>",
    )


def _image_placeholder_no_extracted_text(lang: str) -> str:
    if (lang or "").strip().lower() == "vi":
        return "(Có ảnh cắt trong PDF nhưng không có chú thích hay văn bản trích xuất.)"
    return "（PDF 上は画像として切り出されているが、キャプション・抽出本文は空です。）"


def load_content_list_raw(root: str, file_id: str) -> List[Dict[str, Any]]:
    """加载 content_list.json 原始数组。"""
    path = content_list_path(root, file_id)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _content_list_dir(root: str, file_id: str) -> str:
    """content_list.json 所在目录（img_path 相对于此目录）。"""
    path = content_list_path(root, file_id)
    return os.path.dirname(path)


def content_items_for_pages(
    raw_list: List[Dict[str, Any]],
    page_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    从 content_list 里筛出指定页的条目，保持原列表顺序（即阅读顺序），
    跳过 header/footer/page_number/aside_text 等对问答无用的块。
    """
    wanted = set(page_indices)
    out: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        if int(item.get("page_idx", -1)) not in wanted:
            continue
        tp = item.get("type", "")
        if tp in _INTERLEAVED_SKIP_TYPES:
            continue
        out.append(item)
    return out



def build_interleaved_content(
    items: List[Dict[str, Any]],
    content_dir: str,
    lang: str,
    *,
    origin_pdf: Optional[str] = None,
    page_block_limit: int = INTERLEAVED_PAGE_BLOCK_LIMIT,
    fallback_dpi: int = INTERLEAVED_FALLBACK_DPI,
    crop_max_pixels: int = CROP_IMAGE_MAX_PIXELS,
    crop_min_pixels: int = CROP_IMAGE_MIN_PIXELS,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    把 content_list 条目按阅读顺序转成 Qwen user.content 列表（图文交错）。
    - text / page_footnote → 纯文本段
    - chart / table → 有 img_path 时插入 crop 图 + caption / footnote 文字
    - image → 不送图；仅用 <<<IMAGE_AS_TEXT_BLOCK_BEGIN>>> … <<<IMAGE_AS_TEXT_BLOCK_END>>>
      包裹该块对应的 caption / content / footnote 等文字（见 _image_text_block_wrappers）

    当某页的 block 数 > page_block_limit 时，直接渲染整页 PDF 图替代图文交错（需要
    origin_pdf 有效；否则仍走图文交错）。

    返回 (content_parts, image_paths, temp_png_paths)：
      content_parts: user.content 列表
      image_paths: 所有实际引用的图片路径（用于 prepare_mm_data）
      temp_png_paths: 整页渲染产生的临时 PNG（调用方负责删除）
    """
    page_counts: Dict[int, int] = {}
    for item in items:
        if isinstance(item, dict):
            pidx = int(item.get("page_idx", -1))
            page_counts[pidx] = page_counts.get(pidx, 0) + 1

    fullpage_set: set[int] = set()
    if origin_pdf and os.path.isfile(origin_pdf) and page_block_limit > 0:
        for pidx, cnt in page_counts.items():
            if cnt > page_block_limit:
                fullpage_set.add(pidx)

    fullpage_pngs: Dict[int, str] = {}
    temp_png_paths: List[str] = []
    if fullpage_set:
        rendered = render_pdf_pages_to_png_paths(
            origin_pdf, sorted(fullpage_set), fallback_dpi
        )
        for pidx, png_path in zip(sorted(fullpage_set), rendered):
            fullpage_pngs[pidx] = png_path
            temp_png_paths.append(png_path)

    parts: List[Dict[str, Any]] = []
    image_paths: List[str] = []
    cur_page: Optional[int] = None

    for item in items:
        page_idx = int(item.get("page_idx", 0))
        tp = item.get("type", "")

        if page_idx != cur_page:
            cur_page = page_idx
            page_num = page_idx + 1
            if lang == "vi":
                parts.append({"type": "text", "text": f"\n--- Trang {page_num} ---\n"})
            else:
                parts.append({"type": "text", "text": f"\n--- ページ {page_num} ---\n"})

            if page_idx in fullpage_pngs:
                parts.append({"type": "image", "image": fullpage_pngs[page_idx]})
                image_paths.append(fullpage_pngs[page_idx])
                continue

        if page_idx in fullpage_set:
            continue

        if tp == "image":
            img_rel = item.get("img_path", "")
            abs_ok = bool(
                img_rel and os.path.isfile(os.path.join(content_dir, img_rel))
            )
            cap_parts_img: List[str] = []
            for cap_key in ("chart_caption", "table_caption", "image_caption"):
                cap = item.get(cap_key)
                if isinstance(cap, list):
                    cap_parts_img.extend(str(x) for x in cap if str(x).strip())
                elif cap:
                    cap_parts_img.append(str(cap))
            fn_parts_img: List[str] = []
            for fn_key in ("chart_footnote", "table_footnote", "image_footnote"):
                fn = item.get(fn_key)
                if isinstance(fn, list):
                    fn_parts_img.extend(str(x) for x in fn if str(x).strip())
                elif fn:
                    fn_parts_img.append(str(fn))
            body_img = item.get("content") or ""
            inner_lines_img: List[str] = []
            if cap_parts_img:
                inner_lines_img.append(" ".join(cap_parts_img))
            if body_img:
                inner_lines_img.append(str(body_img)[:2000])
            if fn_parts_img:
                inner_lines_img.append(" ".join(fn_parts_img))
            if not inner_lines_img and abs_ok:
                inner_lines_img.append(_image_placeholder_no_extracted_text(lang))
            if inner_lines_img:
                start_m, end_m = _image_text_block_wrappers(lang)
                block_img = start_m + "\n" + "\n".join(inner_lines_img) + "\n" + end_m
                parts.append({"type": "text", "text": block_img})
            continue

        if tp in ("chart", "table"):
            img_rel = item.get("img_path", "")
            if img_rel:
                abs_path = os.path.join(content_dir, img_rel)
                if os.path.isfile(abs_path):
                    cap_parts: List[str] = []
                    for cap_key in ("chart_caption", "table_caption", "image_caption"):
                        cap = item.get(cap_key)
                        if isinstance(cap, list):
                            cap_parts.extend(str(x) for x in cap if str(x).strip())
                        elif cap:
                            cap_parts.append(str(cap))
                    if cap_parts:
                        parts.append({"type": "text", "text": " ".join(cap_parts)})
                    img_entry: Dict[str, Any] = {
                        "type": "image",
                        "image": abs_path,
                        "min_pixels": crop_min_pixels,
                        "max_pixels": crop_max_pixels,
                    }
                    parts.append(img_entry)
                    image_paths.append(abs_path)
                    fn_parts: List[str] = []
                    for fn_key in ("chart_footnote", "table_footnote", "image_footnote"):
                        fn = item.get(fn_key)
                        if isinstance(fn, list):
                            fn_parts.extend(str(x) for x in fn if str(x).strip())
                        elif fn:
                            fn_parts.append(str(fn))
                    if fn_parts:
                        parts.append({"type": "text", "text": " ".join(fn_parts)})
                    continue
            body = item.get("content") or ""
            if not body and tp == "table":
                body = _html_to_plain_text(item.get("table_body", ""))
            if body:
                parts.append({"type": "text", "text": str(body)[:2000]})
        else:
            text = item.get("text") or item.get("content") or ""
            if text:
                parts.append({"type": "text", "text": str(text)})

    _merge_adjacent_text(parts)
    return parts, image_paths, temp_png_paths


def _merge_adjacent_text(parts: List[Dict[str, Any]]) -> None:
    """原地合并连续的 text 段，减少对话 content 碎片。"""
    i = 0
    while i < len(parts) - 1:
        if parts[i]["type"] == "text" and parts[i + 1]["type"] == "text":
            parts[i]["text"] += "\n" + parts[i + 1]["text"]
            parts.pop(i + 1)
        else:
            i += 1


def interleaved_preamble_ja(page_indices: List[int]) -> str:
    pages_str = "、".join(str(p + 1) for p in page_indices)
    return (
        "以下は、検索で選ばれたPDFページ（"
        + pages_str
        + "）の内容を**読み取り順**で示したものです。\n"
        "テキスト部分はそのまま文字で、図・表（chart/table）は切り抜き画像で提示します。"
        "type が image のブロックは画像ピクセルは送らず、"
        "<<<IMAGE_AS_TEXT_BLOCK_BEGIN>>> と <<<IMAGE_AS_TEXT_BLOCK_END>>> で囲んだ文字情報のみです。\n"
        "解答は**以下の内容のみ**に基づき、他の知識は使わないでください。\n"
    )


def interleaved_preamble_vi(page_indices: List[int]) -> str:
    pages_str = ", ".join(str(p + 1) for p in page_indices)
    return (
        "Dưới đây là nội dung các trang PDF đã tìm được ("
        + pages_str
        + ") theo **thứ tự đọc**.\n"
        "Phần văn bản là chữ; biểu đồ/bảng (chart/table) là ảnh cắt. "
        "Khối loại image không gửi ảnh, chỉ có chữ trong "
        "<<<IMAGE_AS_TEXT_BLOCK_BEGIN>>> … <<<IMAGE_AS_TEXT_BLOCK_END>>>.\n"
        "Chỉ dựa vào nội dung dưới đây, không dùng kiến thức bên ngoài.\n"
    )


def prompt_answer_ja_interleaved(
    question: str, answer_format: str, preamble: str
) -> str:
    fmt = {
        "string": "最終回答は1行の短文のみ。理由や前置きは書かない。",
        "number": "問題が求める数値・単位のみを1行で。",
        "ordered_list": "答えは順序付きの複数項目。1行のPythonリストリテラルで、例: ['第一','第二']。",
        "unordered_list": "答えは複数項目。1行のPythonリストリテラルで、例: ['甲','乙']。順不同。",
    }.get(answer_format, "最終回答は1行の短文のみ。")

    return (
        f"{preamble}\n"
        f"設問：\n{question}\n\n"
        f"出力要件：{fmt}\n"
        "上の資料（テキスト＋画像）を参照して回答してください。"
        "まず最終回答だけを <answer> に入れてください。"
        "その後、回答の根拠があるPDFページ番号（1始まり）だけをPythonの整数リストで "
        "<evidence> に入れてください。\n"
        "<answer>ここに最終回答</answer>\n"
        "<evidence>[1, 2]</evidence>\n"
    )


def prompt_answer_vi_interleaved(
    question: str, answer_format: str, preamble: str
) -> str:
    fmt = {
        "string": "Chỉ một dòng câu trả lời cuối cùng, không giải thích.",
        "number": "Một dòng: chỉ số/đơn vị theo yêu cầu đề bài.",
        "ordered_list": "Nhiều mục có thứ tự: một dòng literal list Python, ví dụ ['một','hai'].",
        "unordered_list": "Nhiều mục: một dòng literal list Python, ví dụ ['A','B'], thứ tự tự do.",
    }.get(answer_format, "Chỉ một dòng câu trả lời cuối cùng.")

    return (
        f"{preamble}\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Yêu cầu định dạng: {fmt}\n"
        "Hãy dựa vào tài liệu trên (văn bản + hình ảnh) để trả lời. "
        "Trước hết, chỉ đặt câu trả lời cuối cùng vào <answer>. "
        "Sau đó, đặt các số trang PDF chứa bằng chứng (đếm từ 1) vào "
        "<evidence> dưới dạng Python list các số nguyên.\n"
        "<answer>câu trả lời cuối cùng</answer>\n"
        "<evidence>[1, 2]</evidence>\n"
    )


def build_interleaved_messages(
    system: str,
    preamble_text: str,
    interleaved_parts: List[Dict[str, Any]],
    question_text: str,
) -> List[dict]:
    """
    组装图文交错的 Qwen 对话。
    user content = preamble(纯文本) + 图文交错部分 + 题目/输出要求(纯文本)
    """
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": preamble_text})
    content.extend(interleaved_parts)
    content.append({"type": "text", "text": question_text})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


