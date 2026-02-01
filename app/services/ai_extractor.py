"""
AI項目抽出サービス
ルールベース抽出を優先し、取れない場合に Gemini API で補完する。
フォールバック時は画像+OCRテキストのマルチモーダル（n8n方式）を試し、最後にテキストのみ。
"""
import base64
import json
import logging
import re
from datetime import datetime
from typing import Any

import google.generativeai as genai
from supabase import create_client

from app.config import settings

logger = logging.getLogger(__name__)


def _first_page_ocr_text(ocr_text: str) -> str:
    """
    複数ページの場合は先頭ページ（[ページ 1] または [ページ 0]）のテキストのみ返す。
    単一ページの場合は ocr_text をそのまま返す。
    """
    if not ocr_text:
        return ocr_text
    for sep in ("[ページ 2]", "[ページ 3]", "[ページ 4]", "[ページ 5]"):
        if sep in ocr_text:
            return ocr_text.split(sep)[0].strip()
    return ocr_text


# 図面でよく使うラベル一覧。前処理で「ラベル: 値」の形に整える際に使用
_KNOWN_LABELS = frozenset({
    "名称", "品名", "材質", "表面処理", "図番", "熱処理", "処理指示",
    "用紙", "尺度", "作成者", "確認者", "承認者", "部品図", "組立図",
})
# AI抽出で使うラベルのみ。前処理出力をこれに絞りノイズを減らす
_EXTRACTION_LABELS = frozenset({
    "名称", "品名", "材質", "表面処理", "図番", "熱処理", "処理指示", "出図日",
})

# ルールベース: ラベル → 出力キー（part_name, material 等）
_LABEL_TO_KEY: dict[str, str] = {
    "名称": "part_name",
    "品名": "part_name",
    "材質": "material",
    "表面処理": "surface_treatment",
    "図番": "drawing_no",
    "熱処理": "process_note",
    "処理指示": "process_note",
    "出図日": "issue_date",
    "部品図": "title",
    "組立図": "title",
}

# 図番の正規表現（2509-0012 形式）
_DRAWING_NO_PATTERN = re.compile(r"\d{4}-\d{4}")

# 値として採用しないパターン（[ページ 0] など）
_PAGE_SEP_PATTERN = re.compile(r"^\[ページ\s*\d+\]$")

# 表面処理・材質コードに多い語。品名で value_next がこれらなら value_prev を優先（他列の値取り込み防止）
_SURFACE_OR_MATERIAL_TERMS = frozenset({
    "酸洗い", "電着塗装", "黒染め", "バフ研磨", "無処理", "めっき",
    "研磨", "塗装", "クロメート", "アルマイト", "リン酸塩処理",
})

# 図番として採用しない語（基準記号 IA/IB/IC、用紙サイズ等）
_NON_DRAWING_NO_TERMS = frozenset({"IC", "IA", "IB", "A", "B", "A3", "A4", "1/1"})


def _extract_by_rules(ocr_text: str) -> dict[str, Any]:
    """
    OCRテキストからラベル直後／直前の値をルールで抽出する。
    1件でも取れたら呼び出し元で確定値として保存し、Gemini は呼ばない。
    """
    result: dict[str, Any] = {
        "title": None,
        "drawing_no": None,
        "part_name": None,
        "material": None,
        "surface_treatment": None,
        "process_note": None,
        "issue_date": None,
        "tags": [],
    }
    if not ocr_text or not ocr_text.strip():
        return result

    logger.debug("_extract_by_rules input (first 500 chars): %s", (ocr_text or "")[:500])

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return result

    for i, line in enumerate(lines):
        if line not in _KNOWN_LABELS:
            continue
        out_key = _LABEL_TO_KEY.get(line)
        if not out_key:
            continue

        # 直後: 次の行が非ラベルならそれを値に
        value_next = None
        if i + 1 < len(lines) and lines[i + 1] not in _KNOWN_LABELS:
            value_next = lines[i + 1].strip()

        # 直後がラベルの場合: 名称・品名は直前の行を値とする（ハウジングカバー → 名称）
        # 材質・熱処理等は「直後以降で最初の非ラベル」を採用（材質→熱処理→ADC12 → ADC12）
        value_prev = None
        if i - 1 >= 0 and lines[i - 1] not in _KNOWN_LABELS:
            value_prev = lines[i - 1].strip()

        value_first_non_label = None
        j = i + 1
        while j < len(lines):
            if lines[j] not in _KNOWN_LABELS:
                value_first_non_label = lines[j].strip()
                break
            j += 1

        if out_key in ("part_name", "title"):
            # value_next が表面処理らしい語なら value_prev を優先（他列の値取り込み防止）
            if value_next and value_next in _SURFACE_OR_MATERIAL_TERMS:
                value = value_prev if value_prev else value_next
            else:
                value = value_next if value_next else value_prev
        else:
            # 材質, 表面処理, 図番, process_note: 直後 or 直後以降の最初の非ラベル
            value = value_next if value_next else value_first_non_label
            if value is None:
                value = value_prev

        # 値の正規化: 先頭の | を除去、[ページ N] は無効値としてスキップ
        if value:
            value = value.lstrip("|").strip()
            if _PAGE_SEP_PATTERN.match(value):
                value = None

        # 品名・名称: 表面処理らしい語・図番形式は採用しない（他列の値取り込み防止）
        if value and out_key in ("part_name", "title"):
            if value in _SURFACE_OR_MATERIAL_TERMS:
                value = None
            elif _DRAWING_NO_PATTERN.match(value):
                value = None

        # 図番: 基準記号（IC, IA 等）・用紙サイズは採用しない
        if value and out_key == "drawing_no" and value in _NON_DRAWING_NO_TERMS:
            value = None

        # 処理指示: 材質の値と同一なら採用しない（熱処理の直後が材質値になるレイアウト対策）
        if value and out_key == "process_note" and value == result.get("material"):
            value = None

        if value and out_key in ("part_name", "material", "surface_treatment", "process_note", "drawing_no", "title"):
            result[out_key] = value
            logger.debug("Rule extracted %s=%r from label %r at line %d", out_key, value, line, i)
        elif value and out_key == "issue_date":
            if re.search(r"\d{4}-\d{2}-\d{2}", value) or re.search(r"\d{4}/\d{2}/\d{2}", value):
                result[out_key] = value

    # 図番: 正規表現で補助検出。ラベルで取れた値が図番形式でなければ上書き
    regex_match = _DRAWING_NO_PATTERN.search(ocr_text)
    if regex_match:
        candidate = regex_match.group(0)
        current = result.get("drawing_no")
        if not current or not _DRAWING_NO_PATTERN.match(current):
            result["drawing_no"] = candidate
            logger.debug("Rule extracted drawing_no=%r from regex (overrode %r)", candidate, current)

    logger.info("_extract_by_rules result: %s", {k: v for k, v in result.items() if v})
    return result


def _has_any_extracted_value(data: dict[str, Any]) -> bool:
    """抽出結果に1件でも有効な値があるか"""
    for k, v in data.items():
        if k == "tags":
            if v and isinstance(v, list) and len(v) > 0:
                return True
        elif v is not None and v != "":
            return True
    return False


def _preprocess_ocr_for_extraction(ocr_text: str) -> str:
    """
    OCRテキストを「ラベル: 値」形式に前処理する。
    図面ではラベルと値が別行になることが多いため、対応関係を推定して1行にまとめる。
    """
    if not ocr_text or not ocr_text.strip():
        return ocr_text

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return ocr_text

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line in _KNOWN_LABELS:
            # 次の行以降で「ラベルでない最初の行」を値とする
            # 材質→熱処理→SS400: 熱処理をスキップし SS400 を材質の値に
            # 用紙→表面処理→黒染め: iを1ずつ進めることで表面処理も処理し、表面処理: 黒染め を取得
            j = i + 1
            value = ""
            while j < len(lines):
                candidate = lines[j]
                if candidate not in _KNOWN_LABELS:
                    value = candidate.strip()
                    break
                j += 1
            if value:
                pairs.append((line, value))
            i += 1  # 1行ずつ進め、連続ラベル(用紙→表面処理)も順に処理
        else:
            i += 1

    if not pairs:
        return ocr_text

    # 抽出に必要なラベルのみ残し、ノイズ（作成者: IC 等）を除去
    filtered = [(k, v) for k, v in pairs if k in _EXTRACTION_LABELS]
    if not filtered:
        return ocr_text

    return "\n".join(f"{k}: {v}" for k, v in filtered)


def _get_gemini_model():
    """Gemini モデルを取得。APIキーが設定されていない場合はNoneを返す。"""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model_name = getattr(settings, "GEMINI_MODEL", None) or "gemini-2.0-flash"
    return genai.GenerativeModel(model_name)


def extract_drawing_fields(ocr_text: str) -> dict[str, Any]:
    """
    OCRテキストから図面情報を抽出する（Google Gemini API使用）。
    
    Args:
        ocr_text: OCRで抽出されたテキスト
    
    Returns:
        抽出された図面情報の辞書
        {
            "title": "タイトル",
            "drawing_no": "図番",
            "part_name": "品名",
            "material": "材質",
            "surface_treatment": "表面処理",
            "process_note": "処理指示",
            "issue_date": "出図日（YYYY-MM-DD形式）",
            "tags": ["タグ1", "タグ2"]
        }
    """
    model = _get_gemini_model()
    if not model:
        logger.warning("Gemini API key not configured; skipping Gemini extraction")
        return {}

    # Gemini に渡す入力の先頭をログ（原因調査用）
    logger.info("Gemini input (first 300 chars): %s", (ocr_text or "")[:300])

    prompt = """あなたは図面から情報を抽出する専門家です。
OCRで抽出されたテキストから、以下の項目を抽出してください。

【最重要ルール - 絶対に守ること】
- OCRテキストに**一字一句、実際に書かれている文字だけ**を根拠に抽出すること。
- 図面に書かれていない項目は**推測・類推・創作を一切せず**、必ず null または空文字列にすること。
- 「たぶんこれだろう」「一般的にはこう」などの推測で埋めない。不明な項目は必ず null にする。
- OCRテキストに書かれている情報は漏れなく抽出すること。

【OCRテキストの構造】
入力が「ラベル: 値」形式（例: 名称: ロックプレート、材質: SCM440）の場合は、**その値をそのまま**各フィールドに出力すること。
改行区切りの生OCRの場合は、ラベル行の**次行または数行先**に値がある。同じ段落・ブロック内を探すこと。
- 例：名称→ベースプレート、材質→SS400、表面処理→黒染め、図番→2509-0017
- 「材質」の直後に「熱処理」とある場合：「熱処理」は別のラベルなので材質の値ではない。その次の行の「SS400」が材質の値。
- ラベル（名称・材質・表面処理・図番・品名・熱処理など）を**値として出力してはいけない**。値は材料名・品名・処理名・図番などの実データのみ。

抽出項目とマッピング:
- title: 図面のタイトル（OCRに「部品図」「組立図」と明記されている場合のみ）
- drawing_no: 図番（「図番」ラベルの下・近くにある値。例: "2509-0017"）
- part_name: 品名（「名称」または「品名」ラベルの下・近くにある値）
- material: 材質（「材質」ラベルの下・近くにある値。SS400, SCM440 等。「熱処理」は材質の値ではない）
- surface_treatment: 表面処理（「表面処理」ラベルの下・近くにある値。黒染め、酸洗い、めっき等）
- process_note: 処理指示（「熱処理」「加工指示」ラベルの下・近くにある値）
- issue_date: 出図日（日付が記載されている場合のみ。YYYY-MM-DD形式。無ければnull）
- tags: タグ配列（図面種類・用途が明らかな場合のみ。無ければ空配列[]）

出力は必ずJSON形式。書かれていない項目は必ず null または空配列[]にすること。

【正解例1】改行区切りの場合:
入力:
名称
ベースプレート
材質
熱処理
SS400
表面処理
黒染め
図番
2509-0017
品名
ベースプレート
正しい出力:
{{"title": null, "drawing_no": "2509-0017", "part_name": "ベースプレート", "material": "SS400", "surface_treatment": "黒染め", "process_note": null, "issue_date": null, "tags": []}}

【正解例2】同一行の場合:
入力: 名称 ロックプレート 材質 SCM440 表面処理 酸洗い 図番 2509-0016
正しい出力:
{{"title": null, "drawing_no": "2509-0016", "part_name": "ロックプレート", "material": "SCM440", "surface_treatment": "酸洗い", "process_note": null, "issue_date": null, "tags": []}}

【禁止 - 絶対にやってはいけないこと】
- OCRに「ベースプレート」と書かれているのに part_name に「シャフト」を入れるのは**間違い**
- OCRに「SS400」と書かれているのに material に「S45C」を入れるのは**間違い**
- OCRに「黒染め」と書かれているのに surface_treatment に「無処理」を入れるのは**間違い**
- OCRに書かれていない値を推測で埋めない

【複数ページの場合】
入力に [ページ 1], [ページ 2] 等が含まれる場合は、**先頭ページのブロックのみ**を参照すること。

以下のテキストから、ラベルと値の対応を正確に抽出してください。書かれている文字だけを使い、推測は一切しないこと。
入力がすでに「ラベル: 値」の形式（例: 名称: ロックプレート）になっている場合は、その値をそのまま出力すること。

=== 入力テキスト ===
{ocr_text[:8000]}
=== 入力終了 ==="""

    content = ""
    try:
        gen_config = genai.types.GenerationConfig(temperature=0)
        response = model.generate_content(prompt, generation_config=gen_config)

        content = (response.text or "").strip()
        # レスポンス全文の先頭をログ（原因調査用）
        logger.info("Gemini raw response (first 2000 chars): %s", (content or "")[:2000])

        if not content:
            logger.warning("Gemini API returned empty content; returning empty result")
            return {}

        # JSON を抽出（```json で囲まれている場合がある）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        extracted = json.loads(content)
        logger.info("Gemini extraction completed: fields=%s", list(extracted.keys()))
        return extracted

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s; response (first 500): %s", e, (content or "")[:500])
        return {}
    except Exception as e:
        logger.error(
            "Gemini API error: type=%s message=%s",
            type(e).__name__,
            str(e),
        )
        if hasattr(e, "response") and e.response is not None:
            logger.error(
                "Gemini API response status_code=%s",
                getattr(e.response, "status_code", None),
            )
        logger.exception("AI extraction failed (stacktrace above)")
        return {}


def extract_drawing_fields_multimodal(
    image_bytes: bytes, ocr_text: str, mime_type: str = "image/png"
) -> dict[str, Any]:
    """
    画像＋OCRテキストの両方をGeminiに渡して抽出する（マルチモーダル、n8n方式）。
    レイアウト・位置関係を参照できるため、品名と表面処理の取り違えなどを軽減できる。
    """
    model = _get_gemini_model()
    if not model:
        logger.warning("Gemini API key not configured; skipping multimodal extraction")
        return {}

    logger.info(
        "Gemini multimodal input: image=%d bytes, ocr_text (first 300 chars)=%s",
        len(image_bytes),
        (ocr_text or "")[:300],
    )

    prompt = """あなたは図面から情報を抽出する専門家です。
図面画像とOCRテキストの**両方**を参照し、タイトルブロック・表のレイアウトを考慮して、以下の項目を抽出してください。
OCRテキストに明記されている値を優先し、画像でレイアウトの曖昧さ（例：品名と表面処理の列の取り違え）を解消してください。

【最重要ルール】
- OCRテキストに実際に書かれている文字だけを根拠に抽出すること。
- 推測・創作は一切しない。不明な項目は必ず null にする。

抽出項目:
- title: 図面のタイトル（部品図・組立図等）
- drawing_no: 図番（2509-0017 形式など）
- part_name: 品名（名称・品名ラベルの値）
- material: 材質（材質ラベルの値）
- surface_treatment: 表面処理（表面処理ラベルの値）
- process_note: 処理指示・熱処理
- issue_date: 出図日（YYYY-MM-DD形式、無ければnull）
- tags: タグ配列（無ければ[]）

出力は必ずJSON形式。書かれていない項目は null または空配列[]にすること。

=== OCRテキスト ===
{ocr_text[:8000]}
=== OCR終了 ===

上記OCRと画像を照らし合わせ、ラベルと値の対応を正確に抽出してください。"""

    content = ""
    try:
        # 画像をinline_dataで渡す（base64）
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        image_part = {"inline_data": {"mime_type": mime_type, "data": b64_data}}
        text_part = prompt.format(ocr_text=ocr_text)
        parts = [image_part, text_part]

        gen_config = genai.types.GenerationConfig(temperature=0)
        response = model.generate_content(parts, generation_config=gen_config)

        content = (response.text or "").strip()
        logger.info("Gemini multimodal raw response (first 2000 chars): %s", (content or "")[:2000])

        if not content:
            logger.warning("Gemini multimodal returned empty content")
            return {}

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        extracted = json.loads(content)
        logger.info("Gemini multimodal extraction completed: fields=%s", list(extracted.keys()))
        return extracted

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse Gemini multimodal response: %s; response (first 500): %s",
            e,
            (content or "")[:500],
        )
        return {}
    except Exception as e:
        logger.error("Gemini multimodal API error: %s", e)
        logger.exception("Multimodal extraction failed")
        return {}


def process_ai_extraction_for_drawing(drawing_id: str) -> dict[str, Any]:
    """
    図面に対してAI項目抽出を実行する。
    
    Args:
        drawing_id: 図面ID
    
    Returns:
        AI抽出結果の辞書
    """
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # OCR結果を取得
    drawing = (
        supabase.table("drawings")
        .select("extracted_json")
        .eq("id", drawing_id)
        .single()
        .execute()
    )

    if not drawing.data:
        logger.warning("Drawing not found: drawing_id=%s", drawing_id)
        return {}

    extracted_json = drawing.data.get("extracted_json") or {}
    ocr_result = extracted_json.get("ocr", {})

    ocr_text = ocr_result.get("ocr_text", "")
    if not ocr_text:
        logger.warning("No OCR text found for drawing_id=%s; skipping extraction", drawing_id)
        return {}

    # 先頭ページのみ使用
    ocr_first = _first_page_ocr_text(ocr_text)

    # 1. ルールベース抽出を最優先
    rule_result = _extract_by_rules(ocr_first)
    if _has_any_extracted_value(rule_result):
        logger.info(
            "Rule-based extraction got values for drawing_id=%s: %s",
            drawing_id,
            [k for k, v in rule_result.items() if v and (k != "tags" or (isinstance(v, list) and v))],
        )
        # 日付の形式を検証
        issue_date_str = rule_result.get("issue_date")
        if issue_date_str:
            try:
                datetime.strptime(issue_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                logger.warning("Invalid date format from rules: %s; setting to null", issue_date_str)
                rule_result["issue_date"] = None
        return rule_result

    # 2. ルールで1件も取れなかった場合: マルチモーダル（画像+OCR）を試し、失敗時はテキストのみ
    logger.info(
        "Rule-based extraction got no values for drawing_id=%s; trying multimodal then text-only",
        drawing_id,
    )
    preprocessed = _preprocess_ocr_for_extraction(ocr_first)
    ocr_text_for_ai = preprocessed if (preprocessed and preprocessed != ocr_first) else ocr_first
    if preprocessed and preprocessed != ocr_first:
        logger.debug("OCR preprocessed to %d chars for Gemini", len(ocr_text_for_ai))

    ai_result: dict[str, Any] = {}

    # 2a. マルチモーダル（画像+OCR）を試行（n8n方式）
    page_images = (
        supabase.table("drawing_files")
        .select("id, gcs_path, mime")
        .eq("drawing_id", drawing_id)
        .eq("type", "page_image")
        .order("page_no", desc=False)
        .limit(1)
        .execute()
    )
    if page_images.data and len(page_images.data) > 0:
        first_page = page_images.data[0]
        gcs_path = first_page.get("gcs_path")
        mime = first_page.get("mime") or "image/png"
        if gcs_path:
            try:
                from app.services.ocr_service import get_image_bytes_from_gcs

                image_bytes = get_image_bytes_from_gcs(gcs_path)
                if image_bytes:
                    ai_result = extract_drawing_fields_multimodal(
                        image_bytes, ocr_text_for_ai, mime_type=mime or "image/png"
                    )
                    if _has_any_extracted_value(ai_result):
                        logger.info("Multimodal extraction succeeded for drawing_id=%s", drawing_id)
            except Exception as e:
                logger.warning("Multimodal extraction failed (will try text-only): %s", e)

    # 2b. マルチモーダルで取れなければテキストのみ
    if not ai_result or not _has_any_extracted_value(ai_result):
        logger.info("Trying text-only Gemini for drawing_id=%s", drawing_id)
        ai_result = extract_drawing_fields(ocr_text_for_ai)

    if not ai_result or not _has_any_extracted_value(ai_result):
        logger.warning("Gemini (multimodal and text-only) returned no usable values for drawing_id=%s", drawing_id)
        return {}

    # 日付の形式を検証・変換
    issue_date_str = ai_result.get("issue_date")
    if issue_date_str:
        try:
            datetime.strptime(issue_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            logger.warning("Invalid date format from Gemini: %s; setting to null", issue_date_str)
            ai_result["issue_date"] = None

    return ai_result


# AI返却の日本語キー → drawings カラム（英語キー）の対応
_AI_KEY_TO_COLUMN: dict[str, str] = {
    "品名": "part_name",
    "材質": "material",
    "図番": "drawing_no",
    "表面処理": "surface_treatment",
    "処理指示": "process_note",
    "熱処理": "process_note",
    "出図日": "issue_date",
    "タイトル": "title",
    "タグ": "tags",
}


def _normalize_ai_result(ai_result: dict[str, Any]) -> dict[str, Any]:
    """
    AI返却のキーを正規化する。日本語キーがあれば英語キーに写す。
    既に英語キーがある場合はそれを優先する。
    """
    column_keys = (
        "title",
        "drawing_no",
        "part_name",
        "material",
        "surface_treatment",
        "process_note",
        "issue_date",
        "tags",
    )
    normalized: dict[str, Any] = {}
    for en_key in column_keys:
        val = ai_result.get(en_key)
        if val is None or val == "":
            for ja_key, mapped_en in _AI_KEY_TO_COLUMN.items():
                if mapped_en == en_key:
                    v = ai_result.get(ja_key)
                    if v is not None and v != "":
                        val = v
                        break
        if val is not None and val != "" and (en_key != "tags" or isinstance(val, list)):
            normalized[en_key] = val
    return normalized


def save_extracted_fields_to_drawing(drawing_id: str, ai_result: dict[str, Any]) -> None:
    """
    AI抽出結果をdrawingsテーブルに保存する。
    
    Args:
        drawing_id: 図面ID
        ai_result: AI抽出結果（英語キーまたは日本語キー）
    """
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # 日本語キーを英語キーに正規化（カラム保存用）。extracted_json.ai には元の ai_result をそのまま保存
    normalized = _normalize_ai_result(ai_result)

    # drawingsテーブルに保存するフィールドを準備
    update_fields: dict[str, Any] = {}

    if normalized.get("title"):
        update_fields["title"] = normalized["title"]
    if normalized.get("drawing_no"):
        update_fields["drawing_no"] = normalized["drawing_no"]
    if normalized.get("part_name"):
        update_fields["part_name"] = normalized["part_name"]
    if normalized.get("material"):
        update_fields["material"] = normalized["material"]
    if normalized.get("surface_treatment"):
        update_fields["surface_treatment"] = normalized["surface_treatment"]
    if normalized.get("process_note"):
        update_fields["process_note"] = normalized["process_note"]
    if normalized.get("issue_date"):
        update_fields["issue_date"] = normalized["issue_date"]
    if normalized.get("tags") and isinstance(normalized["tags"], list):
        update_fields["tags"] = normalized["tags"]

    # extracted_jsonにも保存
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

    existing_json["ai"] = ai_result
    update_fields["extracted_json"] = existing_json

    # 更新を実行
    if update_fields:
        supabase.table("drawings").update(update_fields).eq("id", drawing_id).execute()
        logger.info("AI extraction results saved to drawings table: drawing_id=%s fields=%s", drawing_id, list(update_fields.keys()))
