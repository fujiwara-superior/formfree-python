"""
FormFree - Python変換マイクロサービス
FastAPI + Claude API + pdfplumber
"""

import base64
import os
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from converter import ConversionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_SECRET   = os.environ["INTERNAL_API_SECRET"]
LARAVEL_URL  = os.environ.get("LARAVEL_URL", "http://laravel:8000")

converter_service = ConversionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FormFree Python converter starting up")
    yield
    logger.info("FormFree Python converter shutting down")

app = FastAPI(title="FormFree Converter", lifespan=lifespan)


# ─── リクエストモデル ─────────────────────────────────────────
class ConvertRequest(BaseModel):
    job_id:       str
    company_id:   str
    pdf_content:  str              # base64エンコードされたPDF
    pdf_type:     str = "text"     # text / scan
    columns:      list[dict]       # [{name, description}]
    csv_encoding: str = "sjis"


# ─── エンドポイント ───────────────────────────────────────────
@app.get("/health")
async def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key.startswith("sk-ant-"):
        raise HTTPException(status_code=500, detail="Invalid ANTHROPIC_API_KEY")
    return {"status": "ok", "service": "formfree-converter"}


@app.post("/convert")
async def convert(
    request:          ConvertRequest,
    background_tasks: BackgroundTasks,
    x_api_secret:     str = Header(None),
):
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    background_tasks.add_task(run_conversion, request, anthropic_key)

    return {"accepted": True, "job_id": request.job_id}


# ─── バックグラウンド変換処理 ─────────────────────────────────
async def run_conversion(req: ConvertRequest, anthropic_key: str):
    logger.info(f"Starting conversion for job {req.job_id}")

    try:
        pdf_bytes = base64.b64decode(req.pdf_content)
        actual_type = converter_service.detect_pdf_type(pdf_bytes)

        result = await converter_service.convert(
            pdf_bytes     = pdf_bytes,
            pdf_type      = actual_type,
            columns       = req.columns,
            anthropic_key = anthropic_key,
        )

        if not result["success"]:
            raise ValueError(result["warnings"][0] if result["warnings"] else "抽出できる行がありませんでした")

        await notify_laravel(req.job_id, "completed", rows=result["rows"])
        logger.info(f"Job {req.job_id} completed: {len(result['rows'])} rows")

    except Exception as e:
        logger.error(f"Job {req.job_id} failed: {e}", exc_info=True)
        await notify_laravel(req.job_id, "failed", error=str(e))


async def notify_laravel(job_id: str, status: str,
                         rows: list = None, error: str = None):
    """Laravelにジョブ完了を通知"""
    payload = {"job_id": job_id, "status": status}
    if rows is not None:
        payload["row_count"] = len(rows)
        payload["rows"] = rows
    if error:
        payload["error"] = error

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{LARAVEL_URL}/api/internal/job-completed",
                headers={"X-Api-Secret": API_SECRET, "Accept": "application/json"},
                json=payload,
            )
    except Exception as e:
        logger.warning(f"Failed to notify Laravel: {e}")
