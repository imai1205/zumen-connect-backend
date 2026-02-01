"""
ベクトル化・Pinecone連携サービス
図面の extracted_json（OCR + AI抽出）から検索用テキストを組み立て、
Gemini Embedding API でベクトル化し、Pinecone に upsert する。
"""
import logging
from typing import Any

import google.generativeai as genai
from pinecone import Pinecone
from supabase import create_client

from app.config import settings

logger = logging.getLogger(__name__)

# Gemini embedding-001 は output_dimensionality で 1536 を指定可能（既存 Pinecone インデックスと互換）
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSIONS = 1536


def _is_pinecone_configured() -> bool:
    """PINECONE_* がすべて設定されているか"""
    return bool(
        settings.PINECONE_API_KEY
        and settings.PINECONE_INDEX_NAME
    )


def _build_search_text(extracted_json: dict[str, Any]) -> str:
    """
    extracted_json（ocr.ocr_text + ai の各フィールド）から検索用テキストを組み立てる。
    """
    parts: list[str] = []

    ocr = extracted_json.get("ocr") or {}
    ocr_text = ocr.get("ocr_text") or ""
    if ocr_text:
        parts.append(ocr_text)

    ai = extracted_json.get("ai") or {}
    for key in ("title", "drawing_no", "part_name", "material", "surface_treatment", "process_note"):
        val = ai.get(key)
        if val and isinstance(val, str):
            parts.append(val)
    tags = ai.get("tags")
    if tags and isinstance(tags, list):
        parts.append(" ".join(str(t) for t in tags))

    text = "\n".join(parts).strip()
    return text[:8000]  # トークン上限を考慮して切り詰め


def _get_embedding(text: str) -> list[float] | None:
    """Gemini Embedding API でテキストをベクトル化する。"""
    if not text.strip():
        return None
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; skipping embedding")
        return None
    genai.configure(api_key=api_key)
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
        vec = result.get("embedding")
        return vec if isinstance(vec, list) else None
    except Exception as e:
        logger.exception("Gemini embedding failed: %s", e)
        return None


def vectorize_and_upsert(drawing_id: str, company_id: str, extracted_json: dict[str, Any]) -> None:
    """
    図面の extracted_json から検索用テキストを組み立て、ベクトル化して Pinecone に upsert する。
    PINECONE_* 未設定またはテキストが空の場合は何もしない。
    """
    if not _is_pinecone_configured():
        logger.info("Pinecone not configured (PINECONE_API_KEY/INDEX_NAME); skipping vectorize")
        return

    search_text = _build_search_text(extracted_json)
    if not search_text:
        logger.info("No search text for drawing_id=%s; skipping vectorize", drawing_id)
        return

    vec = _get_embedding(search_text)
    if not vec:
        logger.warning("Failed to get embedding for drawing_id=%s", drawing_id)
        return

    api_key = settings.PINECONE_API_KEY
    index_name = settings.PINECONE_INDEX_NAME
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # metadata は string / number / boolean / list of strings のみ
    metadata: dict[str, str | list[str]] = {
        "company_id": str(company_id),
        "drawing_id": str(drawing_id),
    }

    try:
        index.upsert(
            vectors=[
                {
                    "id": str(drawing_id),
                    "values": vec,
                    "metadata": metadata,
                }
            ],
        )
        logger.info("Pinecone upsert ok: drawing_id=%s", drawing_id)
    except Exception as e:
        logger.exception("Pinecone upsert failed: drawing_id=%s error=%s", drawing_id, e)
        raise


def vectorize_drawing_for_search(drawing_id: str) -> None:
    """
    図面 ID を受け取り、Supabase から extracted_json と company_id を取得して
    ベクトル化・Pinecone upsert を実行する。
    """
    if not _is_pinecone_configured():
        logger.info("Pinecone not configured; skipping vectorize_drawing_for_search")
        return

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    row = (
        supabase.table("drawings")
        .select("company_id, extracted_json")
        .eq("id", drawing_id)
        .single()
        .execute()
    )
    if not row.data:
        logger.warning("Drawing not found: drawing_id=%s", drawing_id)
        return

    company_id = row.data.get("company_id")
    extracted_json = row.data.get("extracted_json") or {}
    if not isinstance(extracted_json, dict):
        extracted_json = {}

    if not company_id:
        logger.warning("Drawing has no company_id: drawing_id=%s", drawing_id)
        return

    vectorize_and_upsert(str(drawing_id), str(company_id), extracted_json)
