"""
FormFree - Python変換マイクロサービス
FastAPI + Claude API + pdfplumber
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from converter import ConversionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_SECRET     = os.environ["INTERNAL_API_SECRET"]
LARAVEL_URL    = os.environ.get("LARAVEL_URL", "http://laravel:8000")
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPA_SERVICE_KEY   = os.environ["SUPA_SERVICE_KEY"]

converter_service = ConversionService(
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"],
    supabase_url      = SUPABASE_URL,
    supabase_key      = SUPA_SERVICE_KEY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FormFree Python converter starting up")
    yield
    logger.info("FormFree Python converter shutting down")

app = FastAPI(title="FormFree Converter", lifespan=lifespan)


# ─── リクエストモデル ─────────────────────────────────────────
class ConvertRequest(BaseModel):
    job_id:           str
    company_id:       str
    pdf_storage_path: str
    pdf_type:         str = "text"   # text / scan
    columns:          list[dict]     # [{name, description}]
    csv_encoding:     str = "sjis"


# ─── エンドポイント ───────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "formfree-converter"}


@app.post("/convert")
async def convert(
    request:          ConvertRequest,
    background_tasks: BackgroundTasks,
    x_api_secret:     str = Header(None),
):
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # バックグラウンドで変換を開始（即座にレスポンスを返す）
    background_tasks.add_task(run_conversion, request)

    return {"accepted": True, "job_id": request.job_id}


# ─── バックグラウンド変換処理 ─────────────────────────────────
async def run_conversion(req: ConvertRequest):
    logger.info(f"Starting conversion for job {req.job_id}")

    try:
        # ① status → processing
        await update_job_status(req.job_id, "processing")

        # ② PDFをSupabase Storageから取得
        pdf_bytes = await converter_service.fetch_pdf(req.pdf_storage_path)

        # ③ PDFタイプを自動判定
        actual_type = converter_service.detect_pdf_type(pdf_bytes)

        # ④ Claude APIで変換
        result = await converter_service.convert(
            pdf_bytes = pdf_bytes,
            pdf_type  = actual_type,
            columns   = req.columns,
        )

        if not result["success"]:
            raise ValueError(result["warnings"][0] if result["warnings"] else "抽出できる行がありませんでした")

        # ⑤ conversion_rowsに保存
        rows_to_insert = [
            {
                "job_id":    req.job_id,
                "row_index": i,
                "data":      row,
                "is_edited": False,
            }
            for i, row in enumerate(result["rows"])
        ]

        if rows_to_insert:
            await insert_conversion_rows(rows_to_insert)

        # ⑥ jobをcompletedに更新
        await update_job_completed(
            job_id    = req.job_id,
            row_count = len(result["rows"]),
        )

        # ⑦ Laravelに完了通知
        await notify_laravel(req.job_id, "completed", len(result["rows"]))
        logger.info(f"Job {req.job_id} completed: {len(result['rows'])} rows")

    except Exception as e:
        logger.error(f"Job {req.job_id} failed: {e}", exc_info=True)
        await update_job_status(req.job_id, "failed", str(e))
        await notify_laravel(req.job_id, "failed", error=str(e))


# ─── Supabase操作 ─────────────────────────────────────────────
async def update_job_status(job_id: str, status: str, error: str = None):
    payload = {"status": status}
    if error:
        payload["error_message"] = error[:500]

    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/conversion_jobs?id=eq.{job_id}",
            headers={
                "apikey":        SUPA_SERVICE_KEY,
                "Authorization": f"Bearer {SUPA_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=payload,
        )


async def update_job_completed(job_id: str, row_count: int):
    from datetime import datetime, timezone
    payload = {
        "status":       "completed",
        "row_count":    row_count,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/conversion_jobs?id=eq.{job_id}",
            headers={
                "apikey":        SUPA_SERVICE_KEY,
                "Authorization": f"Bearer {SUPA_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=payload,
        )


async def insert_conversion_rows(rows: list[dict]):
    # rowsのdataをJSON文字列に変換
    for row in rows:
        row["data"] = json.dumps(row["data"], ensure_ascii=False)

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/conversion_rows",
            headers={
                "apikey":        SUPA_SERVICE_KEY,
                "Authorization": f"Bearer {SUPA_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=rows,
        )


async def notify_laravel(job_id: str, status: str,
                         row_count: int = None, error: str = None):
    """Laravelにジョブ完了を通知（メール送信のトリガー）"""
    payload = {"job_id": job_id, "status": status}
    if row_count is not None:
        payload["row_count"] = row_count
    if error:
        payload["error"] = error

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{LARAVEL_URL}/api/internal/job-completed",
                headers={"X-Api-Secret": API_SECRET},
                json=payload,
            )
    except Exception as e:
        logger.warning(f"Failed to notify Laravel: {e}")
