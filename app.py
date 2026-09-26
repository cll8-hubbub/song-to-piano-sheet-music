import shutil
import threading
import traceback
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline as p

APP_DIR = Path(__file__).parent
JOBS_DIR = APP_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI()
jobs: dict[str, dict] = {}


def job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    d.mkdir(exist_ok=True)
    return d


def run_pipeline(job_id: str, source_type: str, source_value: str):
    d = job_dir(job_id)
    try:
        jobs[job_id]["status"] = "acquiring audio"
        raw_wav = d / "input.wav"
        if source_type == "url":
            p.acquire_audio_from_url(source_value, raw_wav)
        else:
            src_path = Path(source_value)
            p.convert_to_wav(src_path, raw_wav)

        jobs[job_id]["status"] = "separating instruments"
        stem_wav = p.separate_stems(raw_wav, d)

        jobs[job_id]["status"] = "transcribing notes"
        midi_path = d / "transcribed.mid"
        p.transcribe_to_midi(stem_wav, midi_path)

        events = p.extract_note_events(midi_path)
        if not events:
            raise RuntimeError("No notes were detected in this audio.")

        jobs[job_id]["events"] = events

        jobs[job_id]["status"] = "generating difficulty previews"
        for difficulty in ("easy", "medium", "hard"):
            preview_score = p.arrange_for_piano(events, difficulty)
            preview_score.write("midi", fp=str(d / f"preview_{difficulty}.mid"))

        jobs[job_id]["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/api/search")
async def search(q: str):
    if not q.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    try:
        results = p.search_soundcloud(q.strip())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return {"results": results}


@app.post("/api/transcribe")
async def transcribe(
    source_url: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    job_id = str(uuid.uuid4())
    d = job_dir(job_id)
    jobs[job_id] = {"status": "queued"}

    if file is not None and file.filename:
        upload_path = d / f"upload_{file.filename}"
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source_type, source_value = "file", str(upload_path)
    elif source_url.strip():
        source_type, source_value = "url", source_url.strip()
    else:
        return JSONResponse({"error": "Provide a song link or an audio file."}, status_code=400)

    thread = threading.Thread(target=run_pipeline, args=(job_id, source_type, source_value))
    thread.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"status": j["status"], "error": j.get("error")}


@app.post("/api/render/{job_id}")
async def render(
    job_id: str,
    difficulty: str = Form(default="medium"),
    accidental_mode: str = Form(default="original"),
    show_note_names: bool = Form(default=False),
):
    j = jobs.get(job_id)
    if j is None or j.get("status") != "done":
        return JSONResponse({"error": "job not ready"}, status_code=400)

    events = j["events"]
    score = p.arrange_for_piano(events, difficulty)
    p.apply_accidental_mode(score, accidental_mode)
    p.apply_note_name_labels(score, show_note_names)
    score.makeNotation(inPlace=True)

    d = job_dir(job_id)
    xml_path = d / "render.musicxml"
    svg_path = d / "render.svg"
    pdf_path = d / "sheet.pdf"
    p.score_to_musicxml(score, xml_path)
    n_pages = p.render_musicxml_to_svg_and_pdf(xml_path, svg_path, pdf_path)

    return {
        "svg": svg_path.read_text(),
        "pages": n_pages,
        "pdf_url": f"api/download/{job_id}",
    }


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    pdf_path = job_dir(job_id) / "sheet.pdf"
    if not pdf_path.exists():
        return JSONResponse({"error": "not rendered yet"}, status_code=404)
    return FileResponse(pdf_path, filename="sheet_music.pdf", media_type="application/pdf")


@app.get("/api/preview/{job_id}/{difficulty}")
async def preview(job_id: str, difficulty: str):
    if difficulty not in ("easy", "medium", "hard"):
        return JSONResponse({"error": "invalid difficulty"}, status_code=400)
    midi_path = job_dir(job_id) / f"preview_{difficulty}.mid"
    if not midi_path.exists():
        return JSONResponse({"error": "not ready yet"}, status_code=404)
    return FileResponse(midi_path, media_type="audio/midi")


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
