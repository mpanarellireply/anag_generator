import os
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

# In-memory job store
jobs: dict[str, dict] = {}


def _make_progress_callback(job_id: str):
    """Create a callback that updates the job's per-function and global phase progress."""
    def callback(
        phase: str,
        status: str,
        elapsed: float | None = None,
        function_name: str | None = None,
        meta: dict | None = None,
    ):
        job = jobs[job_id]
        elapsed_rounded = round(elapsed, 1) if elapsed is not None else None

        # Handle sentinel pseudo-phases
        if phase == "__functions_discovered__":
            names = (meta or {}).get("function_names", [])
            job["function_names"] = names
            job["functions"] = {
                name: {"status": "pending", "error": None, "output_file": None, "phases": {}}
                for name in names
            }
            return

        if phase == "__function_done__":
            if function_name and function_name in job.get("functions", {}):
                job["functions"][function_name]["status"] = "done"
                output_path = (meta or {}).get("output_file")
                if output_path:
                    job["functions"][function_name]["output_file"] = output_path
                    job.setdefault("output_files", []).append(output_path)
            return

        if phase == "__function_error__":
            if function_name and function_name in job.get("functions", {}):
                job["functions"][function_name]["status"] = "error"
                job["functions"][function_name]["error"] = (meta or {}).get("error", "Unknown error")
            return

        # Normal phase update
        if function_name is None:
            # Global phase (Excel Parser, Parser Agent)
            job.setdefault("global_phases", {})[phase] = {
                "status": status, "elapsed": elapsed_rounded,
            }
        else:
            # Per-function phase
            funcs = job.setdefault("functions", {})
            func = funcs.setdefault(function_name, {
                "status": "running", "error": None, "output_file": None, "phases": {},
            })
            if func["status"] == "pending":
                func["status"] = "running"
            func["phases"][phase] = {"status": status, "elapsed": elapsed_rounded}

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

    jobs[job_id] = {
        "status": "running",
        "summary": None,
        "output_files": [],
        "debug_dir": None,
        "error": None,
        "function_names": [],
        "global_phases": {},
        "functions": {},
    }
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
        "function_names": job.get("function_names", []),
        "global_phases": job.get("global_phases", {}),
        "functions": job.get("functions", {}),
    }


@app.get("/download/debug/{job_id}")
async def download_debug(job_id: str):
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
        headers={"Content-Disposition": f"attachment; filename=debug_{job_id}.zip"},
    )


@app.get("/download/{job_id}/{function_name}")
async def download_function(job_id: str, function_name: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    func_info = job.get("functions", {}).get(function_name)
    if not func_info:
        raise HTTPException(status_code=404, detail="Function not found in this job")
    if func_info["status"] != "done" or not func_info.get("output_file"):
        raise HTTPException(status_code=400, detail="Function output not ready yet")

    file_path = func_info["output_file"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        content = f.read()

    return StreamingResponse(
        BytesIO(content),
        media_type="application/sql",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/download/{job_id}")
async def download_all(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    final_files = job.get("output_files", [])

    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        for f in final_files:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=output_{job_id}.zip"},
    )
