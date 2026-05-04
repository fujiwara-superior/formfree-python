"""
ConversionService
- PDFテキスト抽出（pdfplumber）
- スキャンPDF → Claude Vision
- Claude APIで構造化データ抽出
- Supabase Storageとの連携
"""

import base64
import io
import json
import logging
import os
from typing import Any

import httpx
import pdfplumber
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたはPDF帳票からデータを抽出する専門AIです。
以下のルールを厳密に守って動作してください。

【抽出ルール】
1. ユーザーが指定した列定義に従い、PDFから該当データを抽出する
2. 1つのPDFに複数の明細行がある場合は、全行を抽出する
3. 値が読み取れない場合は null を返す（推測で補完しない）
4. 合計行・小計行・ヘッダー行は含めない（明細行のみ）
5. 金額・数量は数値のみを返す（￥・円・個などの単位は除く）
6. 日付は必ずYYYY-MM-DD形式に統一する（令和・平成などの和暦は西暦に変換）
7. スペース・改行・全角半角の揺れは正規化する
8. 複数ページにわたる明細は全ページ分を結合して返す

【出力ルール】
必ずJSON形式のみで返す。説明文・前置き・コメントは一切含めない。
以下の構造を厳守する：

成功時：
{"success":true,"rows":[{"列名1":"値","列名2":"値"}],"warnings":[],"page_count":1}

失敗時：
{"success":false,"rows":[],"warnings":["抽出できる明細行が見つかりませんでした"],"page_count":1}
"""

AUTO_DETECT_SYSTEM_PROMPT = """あなたはPDF帳票からデータを自動検出・抽出する専門AIです。
以下のルールを厳密に守って動作してください。

【自動検出ルール】
1. PDFの内容を解析し、表・明細・データ行を自動的に識別する
2. データの種類に応じて適切な列名を日本語で自動命名する（例：品名、数量、単価、金額、日付、担当者など）
3. 1つのPDFに複数の明細行がある場合は、全行を抽出する
4. ヘッダー情報（発注番号、日付、取引先名など）も1行目として含める
5. 合計行・小計行は含めない（明細行のみ）
6. 値が読み取れない場合は null を返す（推測で補完しない）
7. 金額・数量は数値のみを返す（￥・円・個などの単位は除く）
8. 日付は必ずYYYY-MM-DD形式に統一する（令和・平成などの和暦は西暦に変換）
9. データが何も見つからない場合のみ失敗とする

【出力ルール】
必ずJSON形式のみで返す。説明文・前置き・コメントは一切含めない。
以下の構造を厳守する：

成功時：
{"success":true,"rows":[{"自動検出列名1":"値","自動検出列名2":"値"}],"warnings":[],"page_count":1}

失敗時：
{"success":false,"rows":[],"warnings":["PDFからデータを検出できませんでした"],"page_count":1}
"""


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


class ConversionService:
    def __init__(self):
        pass

    def _get_headers(self) -> dict:
        key = os.environ["ANTHROPIC_API_KEY"].strip()
        logger.info(f"Using Anthropic key: prefix={key[:20]} suffix={key[-10:]} len={len(key)}")
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def _call_anthropic(self, system: str, messages: list) -> str:
        headers = self._get_headers()
        payload = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            if r.status_code != 200:
                body = r.text[:500]
                logger.error(f"Anthropic API error: status={r.status_code} body={body}")
                raise ValueError(f"Anthropic {r.status_code}: {body}")
            data = r.json()
            return data["content"][0]["text"]

    # ─── PDFタイプ判定 ────────────────────────────────────────
    def detect_pdf_type(self, pdf_bytes: bytes) -> str:
        """テキストが抽出できればtext、できなければscan"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:2]:
                    text = page.extract_text()
                    if text and len(text.strip()) > 20:
                        return "text"
            return "scan"
        except Exception:
            return "scan"

    # ─── 変換メイン ───────────────────────────────────────────
    async def convert(
        self,
        pdf_bytes: bytes,
        pdf_type:  str,
        columns:   list[dict],
    ) -> dict[str, Any]:
        """PDFをClaudeで変換してdictを返す"""

        auto_detect = not columns

        if auto_detect:
            column_definitions = ""
        else:
            column_definitions = "\n".join(
                f"・{col['name']}：{col['description']}"
                for col in columns
            )

        if pdf_type == "text":
            return await self._convert_text_pdf(pdf_bytes, column_definitions, auto_detect)
        else:
            return await self._convert_scan_pdf(pdf_bytes, column_definitions, auto_detect)

    # ─── テキストPDF変換 ──────────────────────────────────────
    async def _convert_text_pdf(
        self, pdf_bytes: bytes, column_definitions: str, auto_detect: bool = False
    ) -> dict:
        text = self._extract_text(pdf_bytes)
        if not text.strip():
            logger.warning("Text extraction failed, falling back to scan mode")
            return await self._convert_scan_pdf(pdf_bytes, column_definitions, auto_detect)

        if auto_detect:
            prompt = f"""## PDFの内容
{text[:12000]}

上記のPDF内容を解析し、含まれるデータを自動検出して全行抽出してください。"""
            system = AUTO_DETECT_SYSTEM_PROMPT
        else:
            prompt = f"""## 抽出する列の定義
{column_definitions}

## PDFの内容
{text[:12000]}

上記のPDF内容から、指定された列のデータを全行抽出してください。"""
            system = SYSTEM_PROMPT

        text_content = await self._call_anthropic(system, [{"role": "user", "content": prompt}])
        return self._parse_response(text_content)

    # ─── スキャンPDF変換（Claude Vision） ────────────────────
    async def _convert_scan_pdf(
        self, pdf_bytes: bytes, column_definitions: str, auto_detect: bool = False
    ) -> dict:
        images = convert_from_bytes(pdf_bytes, dpi=200)

        content = []
        for img in images[:10]:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/png",
                    "data":       b64,
                },
            })

        if auto_detect:
            content.append({
                "type": "text",
                "text": "上記の画像（スキャンPDF）を解析し、含まれるデータを自動検出して全行抽出してください。",
            })
            system = AUTO_DETECT_SYSTEM_PROMPT
        else:
            content.append({
                "type": "text",
                "text": f"""## 抽出する列の定義
{column_definitions}

上記の画像（スキャンPDF）から、指定された列のデータを全行抽出してください。""",
            })
            system = SYSTEM_PROMPT

        text_content = await self._call_anthropic(system, [{"role": "user", "content": content}])
        return self._parse_response(text_content)

    # ─── テキスト抽出 ─────────────────────────────────────────
    def _extract_text(self, pdf_bytes: bytes) -> str:
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n--- 次のページ ---\n\n".join(pages)

    # ─── レスポンスパース + 検証 ──────────────────────────────
    def _parse_response(self, raw: str) -> dict:
        # JSONの前後にあるマークダウンコードブロックを除去
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {raw[:500]}")
            return {
                "success":    False,
                "rows":       [],
                "warnings":   ["AIの応答形式が不正でした。再試行してください。"],
                "page_count": 0,
            }

        # rowsの整合性チェック（必須）
        if result.get("success") and result.get("rows"):
            result["rows"] = self._normalize_rows(result["rows"])

        return result

    def _normalize_rows(self, rows: list[dict]) -> list[dict]:
        """nullと空文字の正規化、数値文字列の正規化"""
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean = {}
            for k, v in row.items():
                if v is None or v == "":
                    clean[k] = None
                elif isinstance(v, str):
                    # 全角数字を半角に変換
                    v = v.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                    # カンマ区切り数字のカンマ除去（金額用）
                    if v.replace(",", "").replace(".", "").isdigit():
                        v = v.replace(",", "")
                    clean[k] = v.strip()
                else:
                    clean[k] = v
            normalized.append(clean)
        return normalized
