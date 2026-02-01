"""
FastAPI ワーカー - 重処理実行
PDF処理、OCR、AI項目抽出、ベクトル化、3D変換などを実行
"""
import asyncio
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import jobs
from app.services.job_processor import run_poll_loop

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動時・終了時の処理"""
    print("FastAPI Worker starting...")

    stop = asyncio.Event()
    poll_task = asyncio.create_task(run_poll_loop(stop=stop))

    yield

    print("FastAPI Worker shutting down...")
    stop.set()
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Zumen Connect Worker",
    description="図面処理ワーカー（PDF/OCR/AI/ベクトル化/3D変換）",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS設定（Next.js API からの呼び出しを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "service": "zumen-connect-worker",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """ヘルスチェック（Phase 4.1 で Supabase/GCS 接続確認を追加予定）"""
    return {
        "status": "ok",
        "service": "worker",
    }


app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
