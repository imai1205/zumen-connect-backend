"""
PDF処理サービス
PDFを画像化し、サムネイルを生成してGCSに保存、drawing_filesに登録する。
"""
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.cloud import storage
from google.oauth2 import service_account
from pdf2image import convert_from_bytes
from PIL import Image
from supabase import create_client, Client

from app.config import settings

logger = logging.getLogger(__name__)


def _get_supabase() -> Client | None:
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_SERVICE_ROLE_KEY
    if not url or not key:
        return None
    return create_client(url, key)


def _get_gcs_client() -> storage.Client:
    project_id = settings.GCS_PROJECT_ID
    bucket_name = settings.GCS_BUCKET_NAME
    service_account_key = settings.GCS_SERVICE_ACCOUNT_KEY

    if not project_id or not bucket_name or not service_account_key:
        raise ValueError(
            "GCS環境変数が設定されていません: GCS_PROJECT_ID, GCS_BUCKET_NAME, GCS_SERVICE_ACCOUNT_KEY"
        )

    credentials = None
    try:
        creds_dict = json.loads(service_account_key)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
    except (json.JSONDecodeError, ValueError):
        creds_path = Path(service_account_key)
        if creds_path.exists():
            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path)
            )
        else:
            raise ValueError(f"GCS_SERVICE_ACCOUNT_KEY が無効です: {service_account_key}")

    return storage.Client(project=project_id, credentials=credentials)


def _download_pdf_from_gcs(gcs_path: str) -> bytes:
    """GCSからPDFをダウンロードしてbytesを返す。"""
    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    return blob.download_as_bytes()


def _upload_image_to_gcs(
    image_bytes: bytes, gcs_path: str, content_type: str = "image/png"
) -> None:
    """画像をGCSにアップロードする。"""
    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(image_bytes, content_type=content_type)


def _generate_thumbnail(image: Image.Image, max_size: tuple[int, int] = (400, 400)) -> Image.Image:
    """画像からサムネイルを生成する。"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def _get_company_id(drawing_id: str) -> str | None:
    """drawing_idからcompany_idを取得する。"""
    supabase = _get_supabase()
    if not supabase:
        return None
    r = supabase.table("drawings").select("company_id").eq("id", drawing_id).execute()
    if not r.data or len(r.data) == 0:
        return None
    return r.data[0].get("company_id")


def _register_drawing_file(
    drawing_id: str,
    type: str,
    gcs_path: str,
    mime: str,
    size: int | None = None,
    page_no: int | None = None,
) -> None:
    """drawing_filesにファイル情報を登録する。"""
    supabase = _get_supabase()
    if not supabase:
        logger.warning("Supabase not configured; skipping file registration")
        return
    payload: dict[str, Any] = {
        "drawing_id": drawing_id,
        "type": type,
        "gcs_path": gcs_path,
        "mime": mime,
    }
    if size is not None:
        payload["size"] = size
    if page_no is not None:
        payload["page_no"] = page_no
    supabase.table("drawing_files").insert(payload).execute()


def process_pdf(drawing_id: str, pdf_gcs_path: str) -> None:
    """
    PDFを処理する。
    1. GCSからPDFをダウンロード
    2. 各ページを画像化
    3. 1ページ目からサムネイル生成
    4. 画像・サムネをGCSにアップロード
    5. drawing_filesに登録
    """
    logger.info("process_pdf started: drawing_id=%s pdf_gcs_path=%s", drawing_id, pdf_gcs_path)

    company_id = _get_company_id(drawing_id)
    if not company_id:
        raise ValueError(f"company_id not found for drawing_id={drawing_id}")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    pdf_bytes = _download_pdf_from_gcs(pdf_gcs_path)
    logger.info("PDF downloaded: size=%d bytes", len(pdf_bytes))

    images = convert_from_bytes(pdf_bytes, dpi=200)
    logger.info("PDF converted to images: %d pages", len(images))

    if len(images) == 0:
        raise ValueError("PDFから画像が生成されませんでした")

    thumbnail_image = _generate_thumbnail(images[0].copy())
    thumbnail_bytes = io.BytesIO()
    thumbnail_image.save(thumbnail_bytes, format="PNG")
    thumbnail_bytes.seek(0)
    thumbnail_data = thumbnail_bytes.read()

    thumbnail_gcs_path = f"drawings/{company_id}/{date_str}/{timestamp}_thumbnail.png"
    _upload_image_to_gcs(thumbnail_data, thumbnail_gcs_path, "image/png")
    logger.info("Thumbnail uploaded: %s", thumbnail_gcs_path)

    _register_drawing_file(
        drawing_id=drawing_id,
        type="thumbnail",
        gcs_path=thumbnail_gcs_path,
        mime="image/png",
        size=len(thumbnail_data),
    )

    for page_no, image in enumerate(images, start=1):
        page_bytes = io.BytesIO()
        image.save(page_bytes, format="PNG")
        page_bytes.seek(0)
        page_data = page_bytes.read()

        page_gcs_path = f"drawings/{company_id}/{date_str}/{timestamp}_page_{page_no:03d}.png"
        _upload_image_to_gcs(page_data, page_gcs_path, "image/png")
        logger.info("Page %d uploaded: %s", page_no, page_gcs_path)

        _register_drawing_file(
            drawing_id=drawing_id,
            type="page_image",
            gcs_path=page_gcs_path,
            mime="image/png",
            size=len(page_data),
            page_no=page_no,
        )

    logger.info("process_pdf completed: drawing_id=%s pages=%d", drawing_id, len(images))
