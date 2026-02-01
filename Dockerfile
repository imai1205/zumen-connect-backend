# FastAPI ワーカー用 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係（PDF処理、OCR、画像処理用）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY ./app /app/app

# Cloud Run は PORT 環境変数を設定する
ENV PORT=8080
EXPOSE 8080

# 起動コマンド（Cloud Run の PORT に従う）
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
