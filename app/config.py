"""
設定管理（環境変数から読み込み）
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# 起動ディレクトリに依存せず backend/.env を読む
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """アプリケーション設定"""

    # FastAPI設定
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Supabase設定
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # GCS設定
    GCS_PROJECT_ID: str = ""
    GCS_BUCKET_NAME: str = ""
    GCS_SERVICE_ACCOUNT_KEY: str = ""  # JSON文字列 or ファイルパス

    # Pinecone設定
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = ""

    # AI/LLM設定
    OPENAI_API_KEY: str = ""  # 未使用（検索用ベクトルは Gemini に統一済み）
    GEMINI_API_KEY: str = ""  # AI抽出に使用
    GEMINI_MODEL: str = "gemini-2.0-flash"  # 未設定時は gemini-2.0-flash（404 回避）
    GOOGLE_VISION_API_KEY: str = ""

    # FastAPI URL（Next.js API から呼び出す際のURL）
    FASTAPI_URL: str = "http://localhost:8000"

    # ワーカー認証（Next.js → FastAPI 呼び出し時、X-Worker-API-Key で検証）
    WORKER_API_KEY: str = ""

    # ジョブポーリング間隔（秒）。未設定時は 5 秒。
    JOB_POLL_INTERVAL_SEC: int = 5

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        case_sensitive=True,
    )


settings = Settings()
