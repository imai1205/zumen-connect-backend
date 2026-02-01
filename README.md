# Zumen Connect Worker (FastAPI)

図面処理ワーカー - PDF処理、OCR、AI項目抽出、ベクトル化、3D変換などを実行

## セットアップ

### 1. 依存関係インストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数設定

`.env` ファイルを作成し、必要な環境変数を設定してください。
（ルートの `.env.example` を参照）

### 3. 起動

```bash
# 開発モード
uvicorn app.main:app --reload

# または
python -m app.main
```

### 4. ヘルスチェック

```bash
curl http://localhost:8000/health
```

## Docker での実行

```bash
docker build -t zumen-worker .
docker run -p 8000:8000 --env-file .env zumen-worker
```

## 実装フェーズ

- Phase 4.1: ワーカー基盤（main.py、ジョブ受付エンドポイント）
- Phase 4.2: PDF処理（pdf_processor.py）
- Phase 4.3: OCR処理（ocr_service.py）
- Phase 4.4: AI項目抽出（ai_extractor.py）
- Phase 4.5: ベクトル化・Pinecone（vectorizer.py）
- Phase 4.6: 3D変換（step_converter.py）
