"""
OCR処理サービス
Google Cloud Vision API を使用して画像からテキストを抽出する。
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from io import BytesIO

from google.cloud import storage, vision
from google.oauth2 import service_account
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def _get_credentials():
    """GCS/Vision 共通の認証情報を取得。"""
    service_account_key = settings.GCS_SERVICE_ACCOUNT_KEY
    if not service_account_key:
        raise ValueError(
            "GCS_SERVICE_ACCOUNT_KEY が設定されていません。"
        )
    try:
        creds_dict = json.loads(service_account_key)
        return service_account.Credentials.from_service_account_info(creds_dict)
    except (json.JSONDecodeError, ValueError):
        creds_path = Path(service_account_key)
        if creds_path.exists():
            return service_account.Credentials.from_service_account_file(
                str(creds_path)
            )
        raise ValueError(f"GCS_SERVICE_ACCOUNT_KEY が無効です: {service_account_key}")


def _get_gcs_client() -> storage.Client:
    """GCSクライアントを取得（pdf_processor.py と同じロジック）。"""
    project_id = settings.GCS_PROJECT_ID
    bucket_name = settings.GCS_BUCKET_NAME
    if not project_id or not bucket_name:
        raise ValueError(
            "GCS環境変数が設定されていません: GCS_PROJECT_ID, GCS_BUCKET_NAME, GCS_SERVICE_ACCOUNT_KEY"
        )
    return storage.Client(
        project=project_id,
        credentials=_get_credentials(),
    )


def _get_vision_client() -> vision.ImageAnnotatorClient:
    """Vision API クライアントを取得。"""
    return vision.ImageAnnotatorClient(credentials=_get_credentials())


def _download_image_from_gcs(gcs_path: str) -> bytes:
    """GCSから画像をダウンロードしてbytesを返す。"""
    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    return blob.download_as_bytes()


def get_image_bytes_from_gcs(gcs_path: str) -> bytes | None:
    """
    GCSから画像をダウンロードしてbytesを返す（公開API）。
    AI抽出のマルチモーダル用。エラー時はNoneを返す。
    """
    try:
        return _download_image_from_gcs(gcs_path)
    except Exception as e:
        logger.warning("Failed to download image from GCS for multimodal: %s", e)
        return None


def _normalize_bounding_box(vertices: List[vision.Vertex]) -> Tuple[float, float, float, float]:
    """
    Vision APIのバウンディングボックス（4頂点）を正規化（x_min, y_min, x_max, y_max）に変換。
    
    Args:
        vertices: Vision APIのバウンディングボックスの頂点リスト（通常4点）
    
    Returns:
        (x_min, y_min, x_max, y_max) のタプル
    """
    if not vertices:
        return (0.0, 0.0, 0.0, 0.0)
    
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def extract_text_with_coordinates(
    image_bytes: bytes, language: str = "jpn"
) -> Dict[str, Any]:
    """
    画像からテキストと座標情報を抽出する（Vision API DOCUMENT_TEXT_DETECTION 使用）。
    
    Args:
        image_bytes: 画像のバイトデータ
        language: 未使用（互換用）
    
    Returns:
        {
            "text": "抽出されたテキスト",
            "tokens": [
                {
                    "text": "単語",
                    "x_min": 10.0,
                    "y_min": 20.0,
                    "x_max": 50.0,
                    "y_max": 30.0,
                    "confidence": 0.95,
                    "level": "word"
                },
                ...
            ],
            "image_width": 800,
            "image_height": 600
        }
    """
    try:
        # 画像サイズを取得
        img = Image.open(BytesIO(image_bytes))
        image_width, image_height = img.size
        
        client = _get_vision_client()
        image = vision.Image(content=image_bytes)
        image_context = vision.ImageContext(language_hints=["ja"])
        response = client.document_text_detection(
            image=image,
            image_context=image_context,
        )
        
        if response.error.message:
            logger.warning("Vision API error: %s", response.error.message)
            return {
                "text": "",
                "tokens": [],
                "image_width": image_width,
                "image_height": image_height,
            }
        
        if not response.full_text_annotation:
            return {
                "text": "",
                "tokens": [],
                "image_width": image_width,
                "image_height": image_height,
            }
        
        full_text = response.full_text_annotation.text.strip()
        tokens = []
        
        # ページ、ブロック、段落、単語の階層を走査してトークンを抽出
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        if word_text.strip():
                            bbox = _normalize_bounding_box(word.bounding_box.vertices)
                            confidence = (
                                word.confidence if hasattr(word, "confidence") else None
                            )
                            tokens.append({
                                "text": word_text,
                                "x_min": bbox[0],
                                "y_min": bbox[1],
                                "x_max": bbox[2],
                                "y_max": bbox[3],
                                "confidence": confidence,
                                "level": "word",
                            })
        
        return {
            "text": full_text,
            "tokens": tokens,
            "image_width": image_width,
            "image_height": image_height,
        }
    except Exception as e:
        logger.exception("OCR処理エラー: %s", e)
        return {
            "text": "",
            "tokens": [],
            "image_width": 0,
            "image_height": 0,
        }


def extract_text_from_image(image_bytes: bytes, language: str = "jpn") -> str:
    """
    画像からテキストを抽出する（Vision API DOCUMENT_TEXT_DETECTION 使用）。
    language は Vision では未使用。互換のため残す。

    Args:
        image_bytes: 画像のバイトデータ
        language: 未使用（互換用）

    Returns:
        抽出されたテキスト。エラー時は空文字列。
    """
    try:
        client = _get_vision_client()
        image = vision.Image(content=image_bytes)
        # 図面は日本語が多いため、言語ヒントで精度向上
        image_context = vision.ImageContext(language_hints=["ja"])
        response = client.document_text_detection(
            image=image,
            image_context=image_context,
        )

        if response.error.message:
            logger.warning("Vision API error: %s", response.error.message)
            return ""

        if not response.full_text_annotation:
            return ""

        return response.full_text_annotation.text.strip()
    except Exception as e:
        logger.exception("OCR処理エラー: %s", e)
        return ""


def extract_text_from_image_safe(image_bytes: bytes, language: str = "jpn") -> str:
    """
    画像からテキストを抽出する（安全版：例外を上げずに空文字列を返す）。
    """
    try:
        return extract_text_from_image(image_bytes, language)
    except Exception as e:
        logger.warning("OCR処理をスキップしました: %s", e)
        return ""


def process_ocr_for_drawing(drawing_id: str) -> dict:
    """
    図面の全ページ画像に対してOCRを実行し、結果を返す。
    座標情報も取得してDBに保存する。

    Returns:
        {"ocr_text": "全ページ結合テキスト", "pages": [{"page_no": 1, "text": "..."}, ...]}
    """
    from supabase import create_client

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    page_images = (
        supabase.table("drawing_files")
        .select("id, gcs_path, page_no")
        .eq("drawing_id", drawing_id)
        .eq("type", "page_image")
        .order("page_no", desc=False)
        .execute()
    )

    if not page_images.data or len(page_images.data) == 0:
        logger.info("No page images found for drawing_id=%s; skipping OCR", drawing_id)
        return {"ocr_text": "", "pages": []}

    pages = []
    all_texts = []

    for page_file in page_images.data:
        page_file_id = page_file["id"]
        page_no = page_file.get("page_no") or 0
        gcs_path = page_file["gcs_path"]

        try:
            image_bytes = _download_image_from_gcs(gcs_path)

            # 座標情報付きOCRを実行
            ocr_result = extract_text_with_coordinates(image_bytes, language="jpn")
            text = ocr_result["text"]
            tokens = ocr_result["tokens"]
            image_width = ocr_result["image_width"]
            image_height = ocr_result["image_height"]

            # ★ DB 更新より先に pages / all_texts へ追加（例外時も OCR テキストを捨てない）
            pages.append({"page_no": page_no, "text": text})
            if text:
                all_texts.append(f"[ページ {page_no}]\n{text}")

            # drawing_files / ocr_tokens 更新は別 try。失敗しても pages / all_texts は変更しない
            try:
                supabase.table("drawing_files").update({
                    "image_width": image_width,
                    "image_height": image_height,
                }).eq("id", page_file_id).execute()

                supabase.table("ocr_tokens").delete().eq("page_file_id", page_file_id).execute()

                if tokens:
                    token_records = [
                        {
                            "drawing_id": drawing_id,
                            "page_file_id": page_file_id,
                            "page_no": page_no,
                            "text": token["text"],
                            "x_min": token["x_min"],
                            "y_min": token["y_min"],
                            "x_max": token["x_max"],
                            "y_max": token["y_max"],
                            "confidence": token.get("confidence"),
                            "level": token.get("level", "word"),
                        }
                        for token in tokens
                    ]
                    batch_size = 1000
                    for i in range(0, len(token_records), batch_size):
                        batch = token_records[i : i + batch_size]
                        supabase.table("ocr_tokens").insert(batch).execute()
                    logger.info(
                        "Saved %d OCR tokens for page %d: drawing_id=%s",
                        len(tokens),
                        page_no,
                        drawing_id,
                    )
            except Exception as db_err:
                logger.warning(
                    "DB update failed (OCR text saved): page=%d drawing_id=%s err=%s",
                    page_no,
                    drawing_id,
                    db_err,
                )

            logger.info("OCR completed for page %d: drawing_id=%s", page_no, drawing_id)
        except Exception as e:
            logger.warning(
                "OCR failed for page %d (gcs_path=%s): %s", page_no, gcs_path, e
            )
            pages.append({"page_no": page_no, "text": ""})

    ocr_text = "\n\n".join(all_texts)

    return {"ocr_text": ocr_text, "pages": pages}
