#!/usr/bin/env python3
"""
使用 GPT 从 IPC-A-610G 生成 PCBA standard 知识 QA。

默认模式使用 MinerU 解析结果：按页组织文本块 + 指定 crop 图，生成：
  1) text_qa：基于解析文本/表格内容的事实性 QA；
  2) image_qa：明确绑定到某张 MinerU crop 图的图像 QA。

也保留旧的整页 PDF 图模式：
  GENERATION_MODE=page_image python3 gpt_generate_ipc610g_standard_qa.py

直接改下方「运行配置」，然后执行：
  python3 gpt_generate_ipc610g_standard_qa.py

策略：MinerU page unit / PDF 页面固定缓存 + GPT 多模态逐页标注 + jsonl checkpoint 断点续跑。
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError as e:  # pragma: no cover - runtime dependency check
    raise SystemExit("缺少依赖 PyMuPDF，请先安装: pip install PyMuPDF") from e

# 复用 rag/utils.py 里的 PDF 打开修补逻辑。
REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from utils import open_pdf_for_render, pdf_page_count  # noqa: E402

# =============================================================================
# 运行配置（只改这里）
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# "mineru_interleaved"（默认）：使用 PCBA/IPC-A-610G_en 的 MinerU 解析文本和 crop 图。
# "page_image"：旧模式，逐页渲染整页 PDF 图给 GPT。
GENERATION_MODE = os.environ.get("GENERATION_MODE", "mineru_interleaved").strip().lower()

PDF_PATH = SCRIPT_DIR / "IPC-A-610 Acceptability of Electronic Assemblies (IPC-A-610G) (IPC).pdf"
MINERU_DIR = SCRIPT_DIR / "IPC-A-610G_en"
CONTENT_LIST_JSON = MINERU_DIR / "IPC-A-610G_en_content_list.json"
MINERU_ORIGIN_PDF = MINERU_DIR / "IPC-A-610G_en_origin.pdf"

_DEFAULT_OUTPUT_DIR = "ipc610g_standard_qa_mineru" if GENERATION_MODE == "mineru_interleaved" else "ipc610g_standard_qa"
OUTPUT_DIR = SCRIPT_DIR / os.environ.get("IPC610_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
PAGE_IMAGE_DIR = OUTPUT_DIR / "page_images"

_CHECKPOINT_NAME = "ipc610g_mineru_page_checkpoint.jsonl" if GENERATION_MODE == "mineru_interleaved" else "ipc610g_page_checkpoint.jsonl"
CHECKPOINT_JSONL = OUTPUT_DIR / _CHECKPOINT_NAME
FAILURE_JSONL = OUTPUT_DIR / "ipc610g_standard_qa_failure.jsonl"
RAW_QA_JSONL = OUTPUT_DIR / "ipc610g_standard_qa_raw.jsonl"
SFT_JSONL = OUTPUT_DIR / "ipc610g_standard_qa_sft.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "ipc610g_standard_qa_summary.json"

# 1-based 页码；END_PAGE=0 表示到最后一页；LIMIT_PAGES=0 表示不限制。
START_PAGE = int(os.environ.get("START_PAGE", "1"))
END_PAGE = int(os.environ.get("END_PAGE", "0"))
LIMIT_PAGES = int(os.environ.get("LIMIT_PAGES", "0"))

PDF_DPI = int(os.environ.get("IPC610_PDF_DPI", os.environ.get("PDF_DPI", "180")))
NO_RESUME = os.environ.get("NO_RESUME", "0").strip().lower() in ("1", "true", "yes")
FORCE_RERENDER = os.environ.get("FORCE_RERENDER", "0").strip().lower() in ("1", "true", "yes")
QUIET = os.environ.get("QUIET", "0").strip().lower() in ("1", "true", "yes")
WORKERS = int(os.environ.get("WORKERS", "8"))

# MinerU 模式限制。
MAX_IMAGES_PER_PAGE = int(os.environ.get("MAX_IMAGES_PER_PAGE", "5"))
MAX_TEXT_CHARS_PER_PAGE = int(os.environ.get("MAX_TEXT_CHARS_PER_PAGE", "6000"))
MAX_TABLE_BODY_CHARS = int(os.environ.get("MAX_TABLE_BODY_CHARS", "2500"))
MAX_LOCAL_TEXT_CHARS = int(os.environ.get("MAX_LOCAL_TEXT_CHARS", "1200"))
MAX_TEXT_QA_PER_PAGE = int(os.environ.get("MAX_TEXT_QA_PER_PAGE", "2"))
MAX_IMAGE_QA_PER_PAGE = int(os.environ.get("MAX_IMAGE_QA_PER_PAGE", "3"))

BASE = os.environ.get("OPENAI_BASE_URL", "https://ai.deeptoken.site/v1")
MODEL = os.environ.get("GPT_IPC610_MODEL", "gpt-5.4")
REASONING_EFFORT = os.environ.get("GPT_IPC610_REASONING_EFFORT", "medium")
API_KEY = os.environ.get(
    "API_KEY",
    "sk-0ee0017ec97ddea77286375770b4bd5bbd378c7bd8ade9badf02d72c17e634b0",
)
MAX_COMPLETION_TOKENS = int(os.environ.get("GPT_IPC610_MAX_TOKENS", "3072"))
MAX_RETRIES = int(os.environ.get("GPT_IPC610_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("GPT_IPC610_TIMEOUT", "180"))

SOURCE_NAME = "IPC-A-610G"
TASK_NAME = "standard_knowledge"

TOPICS = (
    "missing_component",
    "insufficient_solder",
    "tombstoning",
    "flipped_component",
    "wrong_polarity",
    "smt_acceptability",
    "solder_joint",
    "component_orientation",
    "polarity",
    "ipc_class",
    "general_pcba_standard",
)

SKIP_BLOCK_TYPES = frozenset({"header", "footer", "page_number", "aside_text"})
VISUAL_BLOCK_TYPES = frozenset({"image", "table", "chart"})

BLOCKED_TEXT_MARKERS = (
    "copyright",
    "publisher",
    "publishing",
    "table of contents",
    "contents",
    "index",
    "foreword",
    "disclaimer",
    "page number",
    "isbn",
    "all rights reserved",
)

UNCERTAIN_ANSWER_MARKERS = (
    "cannot determine",
    "can't determine",
    "not visible",
    "unclear",
    "illegible",
    "not enough information",
    "not specified",
)

SFT_SYSTEM_PROMPT = (
    "You are an expert in PCBA visual inspection and manufacturing standards. "
    "Answer the factual question concisely using the provided IPC-A-610 standard content."
)

_io_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 日志与 JSONL
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    if not QUIET:
        with _io_lock:
            print(msg, flush=True)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with _io_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"[warn] checkpoint 第 {line_no} 行 JSON 解析失败，已跳过: {e}")
                continue
            if (row.get("status") or "ok") != "ok":
                continue
            try:
                page = int(row.get("page"))
            except (TypeError, ValueError):
                continue
            done[page] = row
    return done


# ---------------------------------------------------------------------------
# 通用文本/路径处理
# ---------------------------------------------------------------------------


def _norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_norm_text(v) for v in value if _norm_text(v)]
    text = _norm_text(value)
    return [text] if text else []


def html_to_plain_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(html))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, limit: int) -> str:
    text = _norm_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def output_rel_path(path: Path) -> str:
    """Return paths in dataset files relative to PCBA/, matching task_type JSON style."""
    try:
        return path.resolve().relative_to(SCRIPT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def output_rel_path_or_none(path: Path | None) -> str | None:
    return output_rel_path(path) if path is not None else None


def resolve_output_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _bad_text(s: str) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True
    return any(m in low for m in BLOCKED_TEXT_MARKERS)


def _uncertain_answer(s: str) -> bool:
    low = (s or "").strip().lower()
    return any(m in low for m in UNCERTAIN_ANSWER_MARKERS)


def normalize_topic(topic: Any) -> str:
    topic_s = str(topic or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "standing": "tombstoning",
        "standing_component": "tombstoning",
        "tombstone": "tombstoning",
        "insufficient_soldering": "insufficient_solder",
        "wrong_polarity_defect": "wrong_polarity",
        "polarity_error": "wrong_polarity",
        "class": "ipc_class",
    }
    topic_s = aliases.get(topic_s, topic_s)
    if topic_s not in TOPICS:
        return "general_pcba_standard"
    return topic_s


def normalize_sft_image_question(question: str, image_index: int) -> str:
    q = question.strip()
    q = re.sub(rf"\b[Ii]n\s+[Ii]mage\s+{image_index}\s*,?", "In the provided image,", q)
    q = re.sub(r"\b[Ii]mage\s+\d+\b", "the provided image", q)
    if "image" not in q.lower() and "table" not in q.lower() and "figure" not in q.lower():
        q = "In the provided image, " + q[0].lower() + q[1:] if q else q
    return q


# ---------------------------------------------------------------------------
# PDF 页面渲染（旧 page_image 模式）
# ---------------------------------------------------------------------------


def cached_page_path(page_index: int) -> Path:
    return PAGE_IMAGE_DIR / f"ipc610g_page_{page_index + 1:04d}.png"


def render_page_to_cached_png(pdf_path: Path, page_index: int, *, dpi: int) -> Path:
    out_path = cached_page_path(page_index)
    if out_path.is_file() and not FORCE_RERENDER:
        return out_path

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    with open_pdf_for_render(str(pdf_path)) as doc:
        n = int(doc.page_count)
        if page_index < 0 or page_index >= n:
            raise IndexError(f"页码越界: page_index={page_index}, total={n}")
        page = doc.load_page(page_index)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
        except RuntimeError:
            pix = page.get_pixmap(matrix=mat, alpha=False, annots=False)
        pix.save(str(out_path))
    return out_path


def resolve_pages(total_pages: int) -> list[int]:
    start = max(1, START_PAGE)
    end = END_PAGE if END_PAGE > 0 else total_pages
    end = min(end, total_pages)
    if start > end:
        return []
    pages = list(range(start, end + 1))
    if LIMIT_PAGES > 0:
        pages = pages[:LIMIT_PAGES]
    return pages


# ---------------------------------------------------------------------------
# MinerU content_list 处理
# ---------------------------------------------------------------------------


def load_content_list_raw(path: Path = CONTENT_LIST_JSON) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 content_list 顶层为 list: {path}")
    return [x for x in data if isinstance(x, dict)]


def content_total_pages(raw_list: list[dict[str, Any]]) -> int:
    max_idx = -1
    for item in raw_list:
        try:
            max_idx = max(max_idx, int(item.get("page_idx")))
        except (TypeError, ValueError):
            continue
    pdf_path = MINERU_ORIGIN_PDF if MINERU_ORIGIN_PDF.is_file() else PDF_PATH
    if pdf_path.is_file():
        try:
            return max(pdf_page_count(str(pdf_path)), max_idx + 1)
        except Exception:
            pass
    return max_idx + 1


def group_content_by_page(raw_list: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for list_idx, item in enumerate(raw_list):
        try:
            page_idx = int(item.get("page_idx"))
        except (TypeError, ValueError):
            continue
        row = dict(item)
        row["_list_idx"] = list_idx
        by_page[page_idx].append(row)
    for rows in by_page.values():
        rows.sort(key=lambda x: int(x.get("_list_idx", 0)))
    return by_page


def make_block_id(page_idx: int, block_type: str, ordinal: int) -> str:
    safe_type = re.sub(r"[^a-z0-9]+", "_", (block_type or "block").lower()).strip("_") or "block"
    return f"p{page_idx + 1:04d}-{safe_type}-{ordinal:02d}"


def extract_caption_footnote(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    captions: list[str] = []
    footnotes: list[str] = []
    for key in ("image_caption", "table_caption", "chart_caption", "caption"):
        captions.extend(as_text_list(item.get(key)))
    for key in ("image_footnote", "table_footnote", "chart_footnote", "footnote"):
        footnotes.extend(as_text_list(item.get(key)))
    return captions, footnotes


def block_body_text(item: dict[str, Any]) -> str:
    tp = str(item.get("type") or "")
    parts: list[str] = []
    captions, footnotes = extract_caption_footnote(item)
    parts.extend(captions)
    if tp == "table":
        body = item.get("table_body") or item.get("text") or item.get("content")
        if body:
            parts.append(html_to_plain_text(str(body))[:MAX_TABLE_BODY_CHARS])
    else:
        body = item.get("text") or item.get("content")
        if body:
            if isinstance(body, (dict, list)):
                parts.append(_norm_text(json.dumps(body, ensure_ascii=False)))
            else:
                parts.append(_norm_text(body))
    parts.extend(footnotes)
    return truncate_text("\n".join(p for p in parts if p), MAX_TABLE_BODY_CHARS if tp == "table" else MAX_LOCAL_TEXT_CHARS)


def resolve_img_path(item: dict[str, Any]) -> tuple[Path | None, str]:
    rel = str(item.get("img_path") or "").strip()
    if not rel:
        return None, ""
    path = (MINERU_DIR / rel).resolve()
    if not path.is_file():
        return None, rel
    return path, rel


def is_low_value_text(text: str) -> bool:
    low = (text or "").strip().lower()
    if len(low) < 12:
        return True
    return any(m in low for m in BLOCKED_TEXT_MARKERS)


def nearby_text_for(rows: list[dict[str, Any]], center_pos: int, *, window: int = 3) -> str:
    parts: list[str] = []
    start = max(0, center_pos - window)
    end = min(len(rows), center_pos + window + 1)
    for pos in range(start, end):
        if pos == center_pos:
            continue
        item = rows[pos]
        tp = str(item.get("type") or "")
        if tp in SKIP_BLOCK_TYPES or tp in VISUAL_BLOCK_TYPES:
            continue
        text = block_body_text(item)
        if text and not is_low_value_text(text):
            parts.append(text)
    return truncate_text("\n".join(parts), MAX_LOCAL_TEXT_CHARS)


def visual_priority(v: dict[str, Any]) -> tuple[int, int, int]:
    type_score = {"table": 0, "chart": 1, "image": 2}.get(str(v.get("visual_type")), 3)
    has_caption = 0 if v.get("caption") or v.get("footnote") else 1
    text_len = -len(str(v.get("body_text") or "") + str(v.get("nearby_text") or ""))
    return (type_score, has_caption, text_len)


def build_page_unit(page: int, content_by_page: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    page_idx = page - 1
    rows = content_by_page.get(page_idx, [])
    text_blocks: list[dict[str, Any]] = []
    visual_blocks: list[dict[str, Any]] = []
    ordinal_by_type: Counter[str] = Counter()

    for pos, item in enumerate(rows):
        tp = str(item.get("type") or "").strip().lower()
        if tp in SKIP_BLOCK_TYPES:
            continue
        ordinal_by_type[tp] += 1
        block_id = make_block_id(page_idx, tp, ordinal_by_type[tp])
        captions, footnotes = extract_caption_footnote(item)
        body_text = block_body_text(item)

        if tp in VISUAL_BLOCK_TYPES:
            img_path_abs, img_path_rel = resolve_img_path(item)
            if img_path_abs is None:
                continue
            # 没有任何语义线索的图片多为装饰/页眉图，跳过。
            nearby = nearby_text_for(rows, pos)
            if tp == "image" and not (captions or footnotes or body_text or nearby):
                continue
            visual_blocks.append(
                {
                    "block_id": block_id,
                    "list_idx": int(item.get("_list_idx", -1)),
                    "page": page,
                    "page_index": page_idx,
                    "page_idx": page_idx,
                    "visual_type": tp,
                    "img_path_abs": str(img_path_abs),
                    "img_path_rel": output_rel_path(img_path_abs),
                    "mineru_img_path_rel": img_path_rel,
                    "caption": captions,
                    "footnote": footnotes,
                    "body_text": body_text,
                    "nearby_text": nearby,
                    "bbox": item.get("bbox"),
                }
            )
            # table/chart 的文本也可作为 text QA 依据。
            if body_text and not is_low_value_text(body_text):
                text_blocks.append(
                    {
                        "block_id": block_id,
                        "list_idx": int(item.get("_list_idx", -1)),
                        "page": page,
                        "page_index": page_idx,
                        "block_type": tp,
                        "text": body_text,
                        "bbox": item.get("bbox"),
                    }
                )
            continue

        if body_text and not is_low_value_text(body_text):
            text_blocks.append(
                {
                    "block_id": block_id,
                    "list_idx": int(item.get("_list_idx", -1)),
                    "page": page,
                    "page_index": page_idx,
                    "block_type": tp,
                    "text": body_text,
                    "bbox": item.get("bbox"),
                }
            )

    visual_blocks = sorted(visual_blocks, key=visual_priority)[:MAX_IMAGES_PER_PAGE]
    for image_index, block in enumerate(visual_blocks):
        block["image_index"] = image_index

    text_chars = 0
    trimmed_text_blocks: list[dict[str, Any]] = []
    for block in text_blocks:
        text = str(block.get("text") or "")
        remaining = MAX_TEXT_CHARS_PER_PAGE - text_chars
        if remaining <= 0:
            break
        block = dict(block)
        block["text"] = truncate_text(text, remaining)
        trimmed_text_blocks.append(block)
        text_chars += len(block["text"])

    return {
        "page": page,
        "page_index": page_idx,
        "page_idx": page_idx,
        "text_blocks": trimmed_text_blocks,
        "visual_blocks": visual_blocks,
        "provided_images": [v["img_path_rel"] for v in visual_blocks],
        "raw_block_count": len(rows),
    }


# ---------------------------------------------------------------------------
# GPT API
# ---------------------------------------------------------------------------


def image_to_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_page_image_api_messages(page: int, image_path: Path) -> tuple[str, list[dict[str, Any]]]:
    system = (
        "You are an expert PCBA visual inspection dataset annotator.\n"
        "You create factual, knowledge-oriented training QA pairs grounded ONLY in the provided "
        "IPC-A-610G PDF page image.\n\n"
        "The dataset is for SMT PCB assembly inspection and is grounded in IPC-A-610 acceptability criteria. "
        "Important target defect themes include Missing Component, Insufficient Solder, Standing/Tombstoning, "
        "Flipped Component, and Wrong Polarity.\n\n"
        "Only create reliable PCBA/IPC-A-610 knowledge questions. If the page is irrelevant, front matter, "
        "copyright, table of contents, index, blank, unreadable, or unsuitable, output an empty items list. "
        "Output ONLY valid JSON, no markdown."
    )
    user_text = (
        f"This is page {page} of IPC-A-610G. Generate 1 to 3 factual/knowledge QA items, or 0 if unsuitable.\n"
        "Return strict JSON: {\"page\": <integer>, \"items\": [{\"question\": \"...\", \"answer\": \"...\", "
        "\"topic\": \"missing_component|insufficient_solder|tombstoning|flipped_component|wrong_polarity|smt_acceptability|solder_joint|component_orientation|polarity|ipc_class|general_pcba_standard\", "
        "\"evidence\": \"short quote or visual evidence\", \"confidence\": \"high|medium\"}]}"
    )
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
    ]
    return system, [{"role": "user", "content": content}]


def format_text_blocks_for_prompt(text_blocks: list[dict[str, Any]]) -> str:
    if not text_blocks:
        return "(no useful extracted text blocks)"
    parts: list[str] = []
    for block in text_blocks:
        parts.append(
            f"[TEXT_BLOCK {block['block_id']} | type={block.get('block_type')} | list_idx={block.get('list_idx')}]\n"
            f"{block.get('text', '')}"
        )
    return "\n\n".join(parts)


def visual_metadata_text(v: dict[str, Any]) -> str:
    lines = [
        f"IMAGE {v['image_index']}:",
        f"- block_id: {v['block_id']}",
        f"- type: {v.get('visual_type')}",
        f"- list_idx: {v.get('list_idx')}",
    ]
    if v.get("caption"):
        lines.append("- caption: " + " | ".join(v["caption"]))
    if v.get("footnote"):
        lines.append("- footnote: " + " | ".join(v["footnote"]))
    if v.get("body_text"):
        lines.append("- extracted_text: " + truncate_text(v["body_text"], MAX_TABLE_BODY_CHARS))
    if v.get("nearby_text"):
        lines.append("- nearby_text: " + truncate_text(v["nearby_text"], MAX_LOCAL_TEXT_CHARS))
    return "\n".join(lines)


def build_mineru_api_messages(page_unit: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    page = int(page_unit["page"])
    system = (
        "You are an expert PCBA visual inspection dataset annotator.\n"
        "You generate high-quality QA pairs grounded ONLY in the provided IPC-A-610G extracted text "
        "and the explicitly indexed images.\n\n"
        "Produce two categories:\n"
        "1. text_qa: questions answerable from extracted text, table text, captions, or footnotes.\n"
        "2. image_qa: questions that require looking at ONE specified image.\n\n"
        "For every image_qa: set image_index to one provided IMAGE index; the question must mention "
        "Image N or the provided image; the answer must be grounded in that image plus local caption/context; "
        "do not create image_qa if the answer can be fully answered from text without inspecting the image.\n"
        "Prefer IPC-A-610/SMT topics: Missing Component, Insufficient Solder, Standing/Tombstoning, "
        "Flipped Component, Wrong Polarity, solder joints, component orientation, polarity, IPC classes, "
        "and acceptability criteria. If content is irrelevant or unreliable, output empty lists.\n"
        "Output ONLY valid JSON, no markdown."
    )

    user_intro = (
        f"This is page {page} of IPC-A-610G, parsed by MinerU.\n\n"
        "Extracted text blocks:\n"
        f"{format_text_blocks_for_prompt(page_unit.get('text_blocks') or [])}\n\n"
        "Available images on this page are listed below. Each IMAGE N metadata is immediately followed by that image.\n"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_intro}]
    visual_blocks = page_unit.get("visual_blocks") or []
    if visual_blocks:
        for v in visual_blocks:
            content.append({"type": "text", "text": "\n" + visual_metadata_text(v) + "\n"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(Path(v["img_path_abs"]))}})
    else:
        content.append({"type": "text", "text": "(no usable images on this page)\n"})

    schema = (
        "\nReturn strict JSON with this schema:\n"
        "{\n"
        f"  \"page\": {page},\n"
        "  \"text_qa\": [\n"
        "    {\"question\": \"...\", \"answer\": \"...\", \"topic\": \"missing_component|insufficient_solder|tombstoning|flipped_component|wrong_polarity|smt_acceptability|solder_joint|component_orientation|polarity|ipc_class|general_pcba_standard\", \"evidence\": {\"block_ids\": [\"...\"], \"quote\": \"...\"}, \"confidence\": \"high|medium\"}\n"
        "  ],\n"
        "  \"image_qa\": [\n"
        "    {\"image_index\": 0, \"question\": \"In Image 0, ...?\", \"answer\": \"...\", \"topic\": \"...\", \"visual_evidence\": \"...\", \"text_evidence\": {\"block_ids\": [\"...\"], \"quote\": \"...\"}, \"confidence\": \"high|medium\"}\n"
        "  ]\n"
        "}\n\n"
        f"Rules:\n"
        f"- text_qa length must be 0 to {MAX_TEXT_QA_PER_PAGE}.\n"
        f"- image_qa length must be 0 to {MAX_IMAGE_QA_PER_PAGE}; at most one image_qa per image_index.\n"
        "- If no images are provided, image_qa must be [].\n"
        "- image_qa.image_index must be one of the provided IMAGE indices.\n"
        "- For table/chart images, it is OK to ask questions like 'According to the table in Image N...'.\n"
        "- Do not generate questions about copyright, publisher, page number, table of contents, or index.\n"
        "- If uncertain, omit the item. JSON only."
    )
    content.append({"type": "text", "text": schema})
    return system, [{"role": "user", "content": content}]


def call_gpt(system: str, user_messages: list[dict[str, Any]]) -> str:
    if not API_KEY:
        raise RuntimeError("未设置 API_KEY 或 OPENAI_API_KEY 环境变量")

    url = f"{BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, *user_messages],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "curl/8.7.1",
            "Accept": "application/json",
        },
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def call_gpt_with_retry(system: str, user_messages: list[dict[str, Any]]) -> str:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gpt(system, user_messages)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GPT 请求失败（重试 {MAX_RETRIES} 次）: {last_err}") from last_err


# ---------------------------------------------------------------------------
# GPT 输出解析与校验
# ---------------------------------------------------------------------------


def _extract_json_text(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{") or text.startswith("["):
        return text

    code = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    if code:
        inner = code.group(1).strip()
        if inner.startswith("{") or inner.startswith("["):
            return inner

    start_positions = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    return text[start : end + 1]


def parse_gpt_json(text: str) -> tuple[dict[str, Any] | None, str]:
    json_text = _extract_json_text(text)
    if not json_text:
        return None, "empty_or_no_json"
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"
    if isinstance(obj, list):
        return {"items": obj}, "ok_list_wrapped"
    if isinstance(obj, dict):
        return obj, "ok"
    return None, "json_not_object_or_list"


def validate_common_item(item: dict[str, Any]) -> tuple[str | None, str | None, str, str, Any, str | None]:
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    if not question or not answer:
        return None, None, "", "", None, "empty_question_or_answer"
    if _bad_text(question) or _bad_text(answer):
        return None, None, "", "", None, "blocked_text"
    if _uncertain_answer(answer):
        return None, None, "", "", None, "uncertain_answer"
    topic = normalize_topic(item.get("topic"))
    confidence = str(item.get("confidence") or "medium").strip().lower()
    if confidence not in ("high", "medium"):
        confidence = "medium"
    evidence = item.get("evidence") or item.get("text_evidence") or ""
    visual_evidence = str(item.get("visual_evidence") or "").strip()
    return question, answer, topic, confidence, evidence, visual_evidence


def validate_page_image_items(obj: dict[str, Any], *, page: int, page_index: int, image_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    raw_items = obj.get("items")
    if raw_items is None and isinstance(obj.get("questions"), list):
        raw_items = obj.get("questions")
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        return [], ["items_not_list"]

    out: list[dict[str, Any]] = []
    dropped: list[str] = []
    for item in raw_items:
        if len(out) >= 3:
            dropped.append("over_3_items")
            break
        if not isinstance(item, dict):
            dropped.append("item_not_object")
            continue
        question, answer, topic, confidence, evidence, visual_or_drop = validate_common_item(item)
        if question is None or answer is None:
            dropped.append(str(visual_or_drop))
            continue
        qid = f"ipc610g-page-p{page:04d}-q{len(out) + 1:02d}"
        out.append(
            {
                "qid": qid,
                "qa_type": "page_image_qa",
                "source": SOURCE_NAME,
                "pdf_path": output_rel_path(PDF_PATH),
                "page": page,
                "page_index": page_index,
                "question": question,
                "sft_question": question,
                "answer": answer,
                "topic": topic,
                "evidence": evidence,
                "confidence": confidence,
                "image_paths": [output_rel_path(image_path)],
                "task": TASK_NAME,
            }
        )
    return out, dropped


def _qa_list(obj: dict[str, Any], key: str) -> list[Any]:
    value = obj.get(key)
    return value if isinstance(value, list) else []


def validate_mineru_items(obj: dict[str, Any], page_unit: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    page = int(page_unit["page"])
    page_index = int(page_unit["page_index"])
    visual_blocks = {int(v["image_index"]): v for v in page_unit.get("visual_blocks") or []}
    out: list[dict[str, Any]] = []
    dropped: list[str] = []

    text_items = _qa_list(obj, "text_qa")
    image_items = _qa_list(obj, "image_qa")
    if not text_items and not image_items and isinstance(obj.get("items"), list):
        # 兼容旧/错误 schema：全部当 text_qa 处理，避免直接丢弃。
        text_items = obj.get("items") or []

    for item in text_items[: MAX_TEXT_QA_PER_PAGE + 1]:
        if len([r for r in out if r.get("qa_type") == "text_qa"]) >= MAX_TEXT_QA_PER_PAGE:
            dropped.append("text_qa_over_limit")
            break
        if not isinstance(item, dict):
            dropped.append("text_item_not_object")
            continue
        question, answer, topic, confidence, evidence, visual_or_drop = validate_common_item(item)
        if question is None or answer is None:
            dropped.append(f"text_{visual_or_drop}")
            continue
        qid = f"ipc610g-text-p{page:04d}-q{len([r for r in out if r.get('qa_type') == 'text_qa']) + 1:02d}"
        out.append(
            {
                "qid": qid,
                "qa_type": "text_qa",
                "source": SOURCE_NAME,
                "task": TASK_NAME,
                "pdf_path": output_rel_path(MINERU_ORIGIN_PDF if MINERU_ORIGIN_PDF.is_file() else PDF_PATH),
                "mineru_dir": output_rel_path(MINERU_DIR),
                "content_list_path": output_rel_path(CONTENT_LIST_JSON),
                "page": page,
                "page_index": page_index,
                "question": question,
                "sft_question": question,
                "answer": answer,
                "topic": topic,
                "confidence": confidence,
                "evidence": evidence,
                "image_paths": [],
            }
        )

    used_image_indices: set[int] = set()
    image_count = 0
    for item in image_items:
        if image_count >= MAX_IMAGE_QA_PER_PAGE:
            dropped.append("image_qa_over_limit")
            break
        if not isinstance(item, dict):
            dropped.append("image_item_not_object")
            continue
        try:
            image_index = int(item.get("image_index"))
        except (TypeError, ValueError):
            dropped.append("image_index_invalid")
            continue
        if image_index in used_image_indices:
            dropped.append("duplicate_image_index")
            continue
        if image_index not in visual_blocks:
            dropped.append("image_index_out_of_range")
            continue

        question, answer, topic, confidence, evidence, visual_or_drop = validate_common_item(item)
        if question is None or answer is None:
            dropped.append(f"image_{visual_or_drop}")
            continue
        if "image" not in question.lower() and "table" not in question.lower() and "figure" not in question.lower():
            dropped.append("image_question_without_visual_anchor")
            continue

        v = visual_blocks[image_index]
        image_path = output_rel_path(Path(v["img_path_abs"]))
        qid = f"ipc610g-img-p{page:04d}-i{image_index:02d}-q01"
        used_image_indices.add(image_index)
        image_count += 1
        out.append(
            {
                "qid": qid,
                "qa_type": "image_qa",
                "source": SOURCE_NAME,
                "task": TASK_NAME,
                "pdf_path": output_rel_path(MINERU_ORIGIN_PDF if MINERU_ORIGIN_PDF.is_file() else PDF_PATH),
                "mineru_dir": output_rel_path(MINERU_DIR),
                "content_list_path": output_rel_path(CONTENT_LIST_JSON),
                "page": page,
                "page_index": page_index,
                "question": question,
                "sft_question": normalize_sft_image_question(question, image_index),
                "answer": answer,
                "topic": topic,
                "confidence": confidence,
                "evidence": {
                    "text_evidence": evidence,
                    "visual_evidence": visual_or_drop,
                    "image_block_id": v.get("block_id"),
                    "image_index": image_index,
                },
                "image_paths": [image_path],
                "image_index": image_index,
                "image_block_id": v.get("block_id"),
                "image_type": v.get("visual_type"),
                "image_path_rel": v.get("img_path_rel"),
                "caption": v.get("caption") or [],
                "footnote": v.get("footnote") or [],
                "bbox": v.get("bbox"),
            }
        )
    return out, dropped


# ---------------------------------------------------------------------------
# 单页处理与产物重建
# ---------------------------------------------------------------------------


def process_page_image_mode(page: int) -> dict[str, Any]:
    page_index = page - 1
    t0 = time.perf_counter()
    try:
        image_path = render_page_to_cached_png(PDF_PATH, page_index, dpi=PDF_DPI)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "failed", "page": page, "page_index": page_index, "reason": "render_failed", "detail": str(e), "elapsed_sec": round(elapsed, 3)}

    system, user_messages = build_page_image_api_messages(page, image_path)
    try:
        raw_gpt = call_gpt_with_retry(system, user_messages)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "failed", "page": page, "page_index": page_index, "image_path": str(image_path.resolve()), "reason": "api_error", "detail": str(e), "elapsed_sec": round(elapsed, 3)}

    obj, parse_reason = parse_gpt_json(raw_gpt)
    if obj is None:
        elapsed = time.perf_counter() - t0
        return {"status": "failed", "page": page, "page_index": page_index, "image_path": str(image_path.resolve()), "reason": "parse_failed", "detail": parse_reason, "raw_gpt": raw_gpt, "elapsed_sec": round(elapsed, 3)}

    items, dropped = validate_page_image_items(obj, page=page, page_index=page_index, image_path=image_path)
    elapsed = time.perf_counter() - t0
    return {"status": "ok", "page": page, "page_index": page_index, "image_path": str(image_path.resolve()), "num_items": len(items), "items": items, "dropped_reasons": dropped, "parse_reason": parse_reason, "raw_gpt": raw_gpt, "elapsed_sec": round(elapsed, 3)}


def process_mineru_page(page: int, content_by_page: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    page_index = page - 1
    t0 = time.perf_counter()
    page_unit = build_page_unit(page, content_by_page)
    if not page_unit.get("text_blocks") and not page_unit.get("visual_blocks"):
        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "page": page,
            "page_index": page_index,
            "num_items": 0,
            "items": [],
            "unit_meta": {
                "raw_block_count": page_unit.get("raw_block_count", 0),
                "text_block_count": 0,
                "visual_block_count": 0,
                "provided_image_count": 0,
            },
            "reason": "no_useful_mineru_blocks",
            "elapsed_sec": round(elapsed, 3),
        }

    system, user_messages = build_mineru_api_messages(page_unit)
    try:
        raw_gpt = call_gpt_with_retry(system, user_messages)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "failed", "page": page, "page_index": page_index, "reason": "api_error", "detail": str(e), "unit_meta": unit_meta(page_unit), "elapsed_sec": round(elapsed, 3)}

    obj, parse_reason = parse_gpt_json(raw_gpt)
    if obj is None:
        elapsed = time.perf_counter() - t0
        return {"status": "failed", "page": page, "page_index": page_index, "reason": "parse_failed", "detail": parse_reason, "raw_gpt": raw_gpt, "unit_meta": unit_meta(page_unit), "elapsed_sec": round(elapsed, 3)}

    items, dropped = validate_mineru_items(obj, page_unit)
    elapsed = time.perf_counter() - t0
    return {
        "status": "ok",
        "page": page,
        "page_index": page_index,
        "num_items": len(items),
        "num_text_qa": sum(1 for x in items if x.get("qa_type") == "text_qa"),
        "num_image_qa": sum(1 for x in items if x.get("qa_type") == "image_qa"),
        "items": items,
        "dropped_reasons": dropped,
        "parse_reason": parse_reason,
        "unit_meta": unit_meta(page_unit),
        "raw_gpt": raw_gpt,
        "elapsed_sec": round(elapsed, 3),
    }


def unit_meta(page_unit: dict[str, Any]) -> dict[str, Any]:
    visual_blocks = page_unit.get("visual_blocks") or []
    return {
        "raw_block_count": page_unit.get("raw_block_count", 0),
        "text_block_count": len(page_unit.get("text_blocks") or []),
        "visual_block_count": len(visual_blocks),
        "provided_image_count": len(page_unit.get("provided_images") or []),
        "provided_visual_blocks": [
            {
                "image_index": v.get("image_index"),
                "block_id": v.get("block_id"),
                "type": v.get("visual_type"),
                "img_path_rel": v.get("img_path_rel"),
            }
            for v in visual_blocks
        ],
    }


def checkpoint_records(done: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [done[p] for p in sorted(done)]


def build_sft_sample(row: dict[str, Any]) -> dict[str, Any]:
    qa_type = row.get("qa_type") or "image_qa"
    images = [output_rel_path(resolve_output_path(p)) for p in (row.get("image_paths") or []) if str(p).strip()]
    if qa_type == "text_qa":
        user_content = row.get("sft_question") or row["question"]
        sample = {
            "id": row["qid"],
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["answer"]},
            ],
            "qa_type": qa_type,
            "task": row.get("task", TASK_NAME),
            "source": row.get("source", SOURCE_NAME),
            "page": row.get("page"),
            "topic": row.get("topic"),
        }
        return sample

    user_content = "<image>" + (row.get("sft_question") or row["question"])
    return {
        "id": row["qid"],
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["answer"]},
        ],
        "images": images,
        "qa_type": qa_type,
        "task": row.get("task", TASK_NAME),
        "source": row.get("source", SOURCE_NAME),
        "page": row.get("page"),
        "topic": row.get("topic"),
        "image_block_id": row.get("image_block_id"),
        "image_type": row.get("image_type"),
    }


def validate_sft_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    stats = Counter()
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id in seen:
            stats["duplicate_id"] += 1
        seen.add(row_id)
        messages = row.get("messages") or []
        text = "".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))
        images = row.get("images") or []
        if text.count("<image>") != len(images):
            stats["bad_image_token_count"] += 1
        for p in images:
            if not resolve_output_path(p).is_file():
                stats["missing_image_files"] += 1
        if row.get("qa_type") == "image_qa" and len(images) != 1:
            stats["image_qa_not_single_image"] += 1
        if row.get("qa_type") == "text_qa" and (images or "<image>" in text):
            stats["text_qa_has_image"] += 1
    return dict(stats)


def rebuild_outputs(done: dict[int, dict[str, Any]], *, total_pages: int, target_pages: list[int]) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    for rec in checkpoint_records(done):
        for item in rec.get("items") or []:
            if isinstance(item, dict):
                raw_rows.append(item)

    raw_rows.sort(key=lambda r: (int(r.get("page") or 0), str(r.get("qid") or "")))
    sft_rows = [build_sft_sample(r) for r in raw_rows]

    write_jsonl(RAW_QA_JSONL, raw_rows)
    write_jsonl(SFT_JSONL, sft_rows)

    by_topic = Counter(str(r.get("topic") or "_missing") for r in raw_rows)
    by_qa_type = Counter(str(r.get("qa_type") or "_missing") for r in raw_rows)
    by_image_type = Counter(str(r.get("image_type") or "_none") for r in raw_rows if r.get("qa_type") == "image_qa")
    drop_reasons = Counter()
    for rec in done.values():
        drop_reasons.update(str(x) for x in (rec.get("dropped_reasons") or []))

    pages_ok = len(done)
    pages_with_items = sum(1 for r in done.values() if int(r.get("num_items") or 0) > 0)
    pages_with_visual_blocks = sum(1 for r in done.values() if int((r.get("unit_meta") or {}).get("visual_block_count") or 0) > 0)
    summary = {
        "source": SOURCE_NAME,
        "generation_mode": GENERATION_MODE,
        "pdf_path": output_rel_path(MINERU_ORIGIN_PDF if GENERATION_MODE == "mineru_interleaved" and MINERU_ORIGIN_PDF.is_file() else PDF_PATH),
        "mineru_dir": output_rel_path_or_none(MINERU_DIR) if GENERATION_MODE == "mineru_interleaved" else None,
        "content_list_path": output_rel_path_or_none(CONTENT_LIST_JSON) if GENERATION_MODE == "mineru_interleaved" else None,
        "pdf_pages": total_pages,
        "target_pages": len(target_pages),
        "pages_ok": pages_ok,
        "pages_with_items": pages_with_items,
        "pages_without_items": pages_ok - pages_with_items,
        "pages_with_visual_blocks": pages_with_visual_blocks,
        "pages_pending": max(0, len(target_pages) - pages_ok),
        "qa_total": len(raw_rows),
        "by_qa_type": dict(sorted(by_qa_type.items())),
        "by_topic": dict(sorted(by_topic.items())),
        "by_image_type": dict(sorted(by_image_type.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "sft_validation": validate_sft_rows(sft_rows),
        "outputs": {
            "checkpoint_jsonl": output_rel_path(CHECKPOINT_JSONL),
            "failure_jsonl": output_rel_path(FAILURE_JSONL),
            "raw_qa_jsonl": output_rel_path(RAW_QA_JSONL),
            "sft_jsonl": output_rel_path(SFT_JSONL),
            "page_image_dir": output_rel_path(PAGE_IMAGE_DIR),
        },
        "config": {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "pdf_dpi": PDF_DPI,
            "start_page": START_PAGE,
            "end_page": END_PAGE,
            "limit_pages": LIMIT_PAGES,
            "workers": WORKERS,
            "max_images_per_page": MAX_IMAGES_PER_PAGE,
            "max_text_chars_per_page": MAX_TEXT_CHARS_PER_PAGE,
            "max_text_qa_per_page": MAX_TEXT_QA_PER_PAGE,
            "max_image_qa_per_page": MAX_IMAGE_QA_PER_PAGE,
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare_mineru_inputs() -> tuple[int, dict[int, list[dict[str, Any]]]]:
    if not CONTENT_LIST_JSON.is_file():
        raise FileNotFoundError(f"找不到 MinerU content_list: {CONTENT_LIST_JSON}")
    raw_list = load_content_list_raw(CONTENT_LIST_JSON)
    content_by_page = group_content_by_page(raw_list)
    total_pages = content_total_pages(raw_list)
    return total_pages, content_by_page


def main() -> int:
    if GENERATION_MODE not in ("mineru_interleaved", "page_image"):
        print(f"无效 GENERATION_MODE={GENERATION_MODE!r}，应为 mineru_interleaved 或 page_image", file=sys.stderr)
        return 2
    if GENERATION_MODE == "page_image" and not PDF_PATH.is_file():
        print(f"找不到 PDF: {PDF_PATH}", file=sys.stderr)
        return 2

    ensure_dirs()
    if NO_RESUME:
        for path in (CHECKPOINT_JSONL, FAILURE_JSONL):
            if path.is_file():
                path.unlink()

    try:
        if GENERATION_MODE == "mineru_interleaved":
            total_pages, content_by_page = prepare_mineru_inputs()
        else:
            total_pages = pdf_page_count(str(PDF_PATH))
            content_by_page = {}
    except Exception as e:
        print(f"初始化输入失败: {e}", file=sys.stderr)
        return 2

    pages = resolve_pages(total_pages)
    if not pages:
        print(
            f"没有可处理页: total_pages={total_pages}, START_PAGE={START_PAGE}, END_PAGE={END_PAGE}, LIMIT_PAGES={LIMIT_PAGES}",
            file=sys.stderr,
        )
        return 2

    done = load_checkpoint(CHECKPOINT_JSONL) if not NO_RESUME else {}
    pending = [p for p in pages if p not in done]

    input_path = CONTENT_LIST_JSON if GENERATION_MODE == "mineru_interleaved" else PDF_PATH
    log(
        f"GENERATION_MODE = {GENERATION_MODE}\n"
        f"INPUT = {input_path.resolve()}\n"
        f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}\n"
        f"总页数 {total_pages} | 目标页 {len(pages)} | 已完成 {len(pages) - len(pending)} | 待处理 {len(pending)} | "
        f"model={MODEL} reasoning_effort={REASONING_EFFORT} workers={WORKERS}"
    )

    run_failures: list[int] = []

    def handle_record(rec: dict[str, Any]) -> None:
        page = int(rec.get("page") or 0)
        if rec.get("status") == "ok":
            append_jsonl(CHECKPOINT_JSONL, rec)
            done[page] = rec
            extra = ""
            if GENERATION_MODE == "mineru_interleaved":
                extra = f" text={rec.get('num_text_qa', 0)} image={rec.get('num_image_qa', 0)}"
            log(f"[page {page}] OK items={rec.get('num_items')}{extra} ({rec.get('elapsed_sec')}s)")
        else:
            append_jsonl(FAILURE_JSONL, rec)
            run_failures.append(page)
            log(f"[page {page}] FAIL reason={rec.get('reason')} detail={rec.get('detail')} ({rec.get('elapsed_sec')}s)")

    def process_one(page: int) -> dict[str, Any]:
        return process_mineru_page(page, content_by_page) if GENERATION_MODE == "mineru_interleaved" else process_page_image_mode(page)

    if pending:
        workers = max(1, WORKERS)
        if workers <= 1:
            for i, page in enumerate(pending, start=1):
                log(f"({i}/{len(pending)}) page={page} ...")
                handle_record(process_one(page))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_one, page): page for page in pending}
                for i, fut in enumerate(as_completed(futures), start=1):
                    page = futures[fut]
                    log(f"({i}/{len(pending)}) page={page} done")
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = {
                            "status": "failed",
                            "page": page,
                            "page_index": page - 1,
                            "reason": "worker_error",
                            "detail": str(e),
                            "elapsed_sec": 0,
                        }
                    handle_record(rec)

    summary = rebuild_outputs(done, total_pages=total_pages, target_pages=pages)
    log(f"已写出 raw QA: {RAW_QA_JSONL.resolve()} ({summary['qa_total']} 条)")
    log(f"已写出 SFT: {SFT_JSONL.resolve()} ({summary['qa_total']} 条)")
    log(f"已写出 summary: {SUMMARY_JSON.resolve()}")
    log("QA 类型分布: " + ", ".join(f"{k}={v}" for k, v in summary["by_qa_type"].items()))
    log("topic 分布: " + ", ".join(f"{k}={v}" for k, v in summary["by_topic"].items()))

    missing = [p for p in pages if p not in done]
    if missing:
        log(f"仍缺 {len(missing)} 页，重新运行可续跑")
        if run_failures:
            log("本轮失败页: " + ", ".join(map(str, run_failures[:30])) + (" ..." if len(run_failures) > 30 else ""))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
