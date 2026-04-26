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
from typing import Any

import anthropic
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


class ConversionService:
    def __init__(self, anthropic_api_key: str, supabase_url: str, supabase_key: str):
        self.client       = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

    # ─── PDF取得 ──────────────────────────────────────────────
    async def fetch_pdf(self, storage_path: str) -> bytes:
        """Supabase StorageからPDFを取得"""
        url = f"{self.supabase_url}/storage/v1/object/pdfs/{storage_path}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                url,
                headers={
                    "apikey":        self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                },
            )
            resp.raise_for_status()
            return resp.content

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

        column_definitions = "\n".join(
            f"・{col['name']}：{col['description']}"
            for col in columns
        )

        if pdf_type == "text":
            return await self._convert_text_pdf(pdf_bytes, column_definitions)
        else:
            return await self._convert_scan_pdf(pdf_bytes, column_definitions)

    # ─── テキストPDF変換 ──────────────────────────────────────
    async def _convert_text_pdf(
        self, pdf_bytes: bytes, column_definitions: str
    ) -> dict:
        # テキスト抽出
        text = self._extract_text(pdf_bytes)
        if not text.strip():
            # テキスト抽出失敗 → スキャンにフォールバック
            logger.warning("Text extraction failed, falling back to scan mode")
            return await self._convert_scan_pdf(pdf_bytes, column_definitions)

        prompt = f"""## 抽出する列の定義
{column_definitions}

## PDFの内容
{text[:12000]}

上記のPDF内容から、指定された列のデータを全行抽出してください。"""

        response = await self.client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 4096,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )

        return self._parse_response(response.content[0].text)

    # ─── スキャンPDF変換（Claude Vision） ────────────────────
    async def _convert_scan_pdf(
        self, pdf_bytes: bytes, column_definitions: str
    ) -> dict:
        # PDF → 画像変換
        images = convert_from_bytes(pdf_bytes, dpi=200)

        content = []
        for img in images[:10]:  # 最大10ページ
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

        content.append({
            "type": "text",
            "text": f"""## 抽出する列の定義
{column_definitions}

上記の画像（スキャンPDF）から、指定された列のデータを全行抽出してください。""",
        })

        response = await self.client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 4096,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": content}],
        )

        return self._parse_response(response.content[0].text)

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
