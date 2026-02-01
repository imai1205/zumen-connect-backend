"""
ジョブ処理ループ
queued ジョブをポーリングし、process_drawing_job で処理する。
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client

from app.config import settings
from app.services.pdf_processor import process_pdf
from app.services.ocr_service import process_ocr_for_drawing
from app.services.ai_extractor import process_ai_extraction_for_drawing, save_extracted_fields_to_drawing
from app.services.vectorizer import vectorize_drawing_for_search

logger = logging.getLogger(__name__)


def _get_supabase() -> Client | None:
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_SERVICE_ROLE_KEY
    if not url or not key:
        return None
    return create_client(url, key)


def _fetch_one_queued_job() -> dict[str, Any] | None:
    """status=queued のジョブを1件取得。なければ None。"""
    supabase = _get_supabase()
    if not supabase:
        return None
    r = (
        supabase.table("processing_jobs")
        .select("id, drawing_id, status, created_at")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        return None
    return r.data[0]


def _update_job(
    job_id: str,
    status: str,
    *,
    step: str | None = None,
    error_message: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> None:
    supabase = _get_supabase()
    if not supabase:
        return
    payload: dict[str, Any] = {"status": status}
    if step is not None:
        payload["step"] = step
    if error_message is not None:
        payload["error_message"] = error_message
    if started_at is not None:
        payload["started_at"] = started_at.isoformat()
    if finished_at is not None:
        payload["finished_at"] = finished_at.isoformat()
    supabase.table("processing_jobs").update(payload).eq("id", job_id).execute()


def process_drawing_job(drawing_id: str) -> None:
    """
    図面ジョブの処理。
    PDF処理を実行し、drawings.status を 'processing' に更新する。
    後で OCR・AI抽出・ベクトル化を組み込む。
    """
    logger.info("process_drawing_job started: drawing_id=%s", drawing_id)
    supabase = _get_supabase()
    if not supabase:
        logger.warning("Supabase not configured; skipping job processing")
        return

    supabase.table("drawings").update({"status": "processing"}).eq(
        "id", drawing_id
    ).execute()
    logger.info("drawings.status updated to 'processing': drawing_id=%s", drawing_id)

    # PDFファイルを検索（type=original, mime=application/pdf）
    pdf_files = (
        supabase.table("drawing_files")
        .select("id, gcs_path")
        .eq("drawing_id", drawing_id)
        .eq("type", "original")
        .eq("mime", "application/pdf")
        .execute()
    )

    if pdf_files.data and len(pdf_files.data) > 0:
        # PDF がある場合: PDF を処理して画像化
        pdf_file = pdf_files.data[0]
        pdf_gcs_path = pdf_file["gcs_path"]
        logger.info("Processing PDF: drawing_id=%s gcs_path=%s", drawing_id, pdf_gcs_path)
        process_pdf(drawing_id, pdf_gcs_path)
        logger.info("PDF processing completed: drawing_id=%s", drawing_id)
    else:
        # PDF がない場合: page_image があるか確認
        page_images = (
            supabase.table("drawing_files")
            .select("id, gcs_path, mime")
            .eq("drawing_id", drawing_id)
            .eq("type", "page_image")
            .execute()
        )
        if page_images.data and len(page_images.data) > 0:
            logger.info(
                "No PDF found, but %d page_image(s) exist: drawing_id=%s",
                len(page_images.data),
                drawing_id,
            )
            # 画像は既にあるので、PDF処理はスキップして OCR・AI 抽出へ進む
        else:
            logger.info(
                "No PDF or page_image found for drawing_id=%s; skipping processing",
                drawing_id,
            )
            return

    # OCR処理を実行
    try:
        ocr_result = process_ocr_for_drawing(drawing_id)
        if ocr_result.get("ocr_text") or ocr_result.get("pages"):
            # extracted_jsonにOCR結果を保存（既存のデータがあればマージ）
            current_drawing = (
                supabase.table("drawings")
                .select("extracted_json")
                .eq("id", drawing_id)
                .single()
                .execute()
            )
            existing_json = current_drawing.data.get("extracted_json") if current_drawing.data else {}
            if not isinstance(existing_json, dict):
                existing_json = {}

            existing_json["ocr"] = ocr_result
            supabase.table("drawings").update({"extracted_json": existing_json}).eq(
                "id", drawing_id
            ).execute()
            logger.info("OCR results saved: drawing_id=%s pages=%d", drawing_id, len(ocr_result.get("pages", [])))
        else:
            logger.info("No OCR text extracted: drawing_id=%s", drawing_id)
    except Exception as e:
        logger.exception("OCR processing failed (continuing): drawing_id=%s error=%s", drawing_id, e)
        # OCR失敗時もジョブ全体を止めない

    # AI項目抽出を実行
    try:
        ai_result = process_ai_extraction_for_drawing(drawing_id)
        if ai_result:
            save_extracted_fields_to_drawing(drawing_id, ai_result)
            logger.info("AI extraction completed: drawing_id=%s fields=%s", drawing_id, list(ai_result.keys()))
        else:
            logger.info("No AI extraction results: drawing_id=%s", drawing_id)
    except Exception as e:
        logger.exception("AI extraction failed (continuing): drawing_id=%s error=%s", drawing_id, e)
        # AI抽出失敗時もジョブ全体を止めない

    # ベクトル化・Pinecone upsert
    try:
        vectorize_drawing_for_search(drawing_id)
        logger.info("Vectorize completed: drawing_id=%s", drawing_id)
    except Exception as e:
        logger.exception("Vectorize failed (continuing): drawing_id=%s error=%s", drawing_id, e)


async def _run_one_cycle() -> None:
    """1サイクル: キューから1件取得し、処理して完了まで更新する。"""
    loop = asyncio.get_event_loop()
    job = await asyncio.to_thread(_fetch_one_queued_job)
    if not job:
        return
    job_id = job["id"]
    drawing_id = job["drawing_id"]
    now = datetime.now(timezone.utc)

    # 取得直後に running に更新（重複実行防止）
    await asyncio.to_thread(
        _update_job,
        job_id,
        "running",
        step="convert",
        started_at=now,
    )

    try:
        await asyncio.to_thread(process_drawing_job, drawing_id)
        await asyncio.to_thread(
            _update_job,
            job_id,
            "success",
            finished_at=datetime.now(timezone.utc),
        )
        logger.info("job completed: job_id=%s drawing_id=%s", job_id, drawing_id)
    except Exception as e:
        logger.exception("job failed: job_id=%s drawing_id=%s", job_id, drawing_id)
        await asyncio.to_thread(
            _update_job,
            job_id,
            "error",
            error_message=str(e),
            finished_at=datetime.now(timezone.utc),
        )


async def run_poll_loop(*, stop: asyncio.Event) -> None:
    """
    バックグラウンドポーリングループ。
    stop が set されるまで、JOB_POLL_INTERVAL_SEC ごとにキューをチェックする。
    """
    interval = max(1, settings.JOB_POLL_INTERVAL_SEC)
    logger.info("job poll loop started (interval=%ss)", interval)

    while not stop.is_set():
        try:
            if _get_supabase() is None:
                logger.debug("Supabase not configured; skipping poll cycle")
            else:
                await _run_one_cycle()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("poll cycle error: %s", e)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            pass

    logger.info("job poll loop stopped")
