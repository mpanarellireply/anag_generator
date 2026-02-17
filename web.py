import os
import shutil
import threading
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from src.logging_config import setup_logging
from src.orchestrator import Orchestrator

load_dotenv()
setup_logging()

app = FastAPI(title="PL/SQL Generator")

# In-memory job store: job_id -> {status, summary, output_dir, error, phases}
jobs: dict[str, dict] = {}


def _make_progress_callback(job_id: str):
    """Create a callback that updates the job's phase progress."""
    def callback(phase: str, status: str, elapsed: float | None = None):
        phases = jobs[job_id].setdefault("phases", {})
        phases[phase] = {"status": status, "elapsed": round(elapsed, 1) if elapsed is not None else None}
    return callback

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _run_pipeline(job_id: str, excel_path: str, params: dict):
    """Run the orchestrator pipeline in a background thread."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        output_dir = "output"
        debug_dir = f"jobs/{job_id}"
        cache_dir = "parser_cache"

        orchestrator = Orchestrator(
            openai_api_key=api_key,
            model=params.get("model", "gpt-4o-mini"),
            template_dir=".",
            output_dir=output_dir,
            cache_dir=cache_dir,
            debug_dir=debug_dir,
            progress_callback=_make_progress_callback(job_id),
        )

        example_sql_path = None
        if params.get("example_sql_name"):
            example_sql_path = str(UPLOAD_DIR / params["example_sql_name"])

        summary = orchestrator.run(
            excel_path,
            function_name=params.get("function") or None,
            force_parse=params.get("force_parse", False),
            vertical_code=params.get("vertical_code", 1),
            start_code=params.get("start_code", 1),
            example_sql=example_sql_path,
            max_refine=params.get("max_refine", 3),
            skip_logic=params.get("skip_logic", False),
            skip_review=params.get("skip_review", False),
            skip_refine=params.get("skip_refine", False),
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["summary"] = summary
        jobs[job_id]["output_files"] = orchestrator.saved_files
        jobs[job_id]["debug_dir"] = debug_dir

    except Exception:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = traceback.format_exc()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/run")
async def run_pipeline(
    excel_file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini"),
    function: str = Form(""),
    force_parse: bool = Form(False),
    vertical_code: str = Form(""),
    start_code: int = Form(1),
    max_refine: int = Form(3),
    skip_logic: bool = Form(False),
    skip_review: bool = Form(False),
    skip_refine: bool = Form(False),
    example_sql: UploadFile | None = File(None),
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    # Build job_id from function name (or excel filename) + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = function.strip() if function.strip() else Path(excel_file.filename).stem
    job_id = f"{base_name}_{timestamp}"
    excel_path = str(UPLOAD_DIR / f"{job_id}_{excel_file.filename}")
    with open(excel_path, "wb") as f:
        f.write(await excel_file.read())

    # Save optional example SQL
    example_sql_name = None
    if example_sql and example_sql.filename:
        example_sql_name = f"{job_id}_{example_sql.filename}"
        with open(str(UPLOAD_DIR / example_sql_name), "wb") as f:
            f.write(await example_sql.read())

    params = {
        "model": model,
        "function": function.strip(),
        "force_parse": force_parse,
        "vertical_code": vertical_code,
        "start_code": start_code,
        "max_refine": max_refine,
        "skip_logic": skip_logic,
        "skip_review": skip_review,
        "skip_refine": skip_refine,
        "example_sql_name": example_sql_name,
    }

    jobs[job_id] = {"status": "running", "summary": None, "output_files": None, "debug_dir": None, "error": None, "phases": {}}
    thread = threading.Thread(target=_run_pipeline, args=(job_id, excel_path, params), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "summary": job["summary"],
        "error": job["error"],
        "phases": job.get("phases", {}),
    }


@app.get("/download/debug/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    debug_dir = Path(job["debug_dir"])

    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        if debug_dir.exists():
            for f in debug_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f"{f.relative_to(debug_dir)}")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=output_{job_id}.zip"},
    )

@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    final_files = job["output_files"]
    print("-" * 80)
    print(final_files)
    print("-" * 80)

    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        for f in final_files:
            if os.path.exists(f):
                zf.write(f)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=output_{job_id}.zip"},
    )