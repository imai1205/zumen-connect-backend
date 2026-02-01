"""
ジョブ依頼 API
Next.js 司令塔から呼ばれ、processing_jobs に queued を登録する
"""
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from supabase import create_client, Client

from app.config import settings

router = APIRouter()


def _get_supabase() -> Client:
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_SERVICE_ROLE_KEY
    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured",
        )
    return create_client(url, key)


def _verify_worker_key(x_worker_api_key: str | None = Header(None, alias="X-Worker-API-Key")):
    """WORKER_API_KEY が設定されている場合のみヘッダー検証"""
    if not settings.WORKER_API_KEY:
        return
    if x_worker_api_key != settings.WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Worker-API-Key")


class ProcessRequest(BaseModel):
    drawing_id: str


@router.post("/process")
async def process_drawing(
    body: ProcessRequest,
    _: None = Depends(_verify_worker_key),
):
    """
    図面処理ジョブをキューに登録する。
    processing_jobs に status=queued で1件 insert し、job_id を返す。
    """
    supabase = _get_supabase()
    drawing_id = body.drawing_id

    row = (
        supabase.table("processing_jobs")
        .insert(
            {
                "drawing_id": drawing_id,
                "status": "queued",
                "progress": 0,
            }
        )
        .execute()
    )

    if not row.data or len(row.data) == 0:
        raise HTTPException(status_code=500, detail="Failed to create processing job")

    job = row.data[0]
    return {
        "job_id": job["id"],
        "drawing_id": drawing_id,
        "status": "queued",
    }
