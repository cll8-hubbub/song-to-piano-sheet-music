import json
import shutil
import threading
import traceback
import uuid
import warnings
from datetime import datetime
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


def save_job_meta(job_id: str):
    meta = dict(jobs[job_id].get("meta", {}))
    meta["job_id"] = job_id
    with open(job_dir(job_id) / "meta.json", "w") as f:
        json.dump(meta, f)


def ensure_job_loaded(job_id: str) -> bool:
    """True if the job is ready to render. Reconstructs from the saved MIDI
    (fast) rather than re-running the slow audio pipeline, so a library
    item opened after a server restart still works."""
    j = jobs.get(job_id)
    if j is not None and j.get("status") == "done":
        return True

    d = JOBS_DIR / job_id
    midi_path = d / "transcribed.mid"
    if not midi_path.exists():
        return False

    events = p.extract_note_events(midi_path)
    meta = {}
    meta_path = d / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    jobs[job_id] = {"status": "done", "events": events, "meta": meta}

    for difficulty in ("easy", "medium", "hard"):
        preview_path = d / f"preview_{difficulty}.mid"
        if not preview_path.exists():
            preview_score = p.arrange_for_piano(events, difficulty)
            preview_score.write("midi", fp=str(preview_path))

    return True


def run_pipeline(job_id: str, source_type: str, source_value: str):
    d = job_dir(job_id)
    try:
        jobs[job_id]["status"] = "acquiring audio"
        raw_wav = d / "input.wav"
        if source_type == "url":
            url_meta = p.acquire_audio_from_url(source_value, raw_wav)
            meta = jobs[job_id]["meta"]
            meta["title"] = meta.get("title") or url_meta.get("title")
            meta["uploader"] = meta.get("uploader") or url_meta.get("uploader")
            meta["webpage_url"] = meta.get("webpage_url") or url_meta.get("webpage_url")
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
        save_job_meta(job_id)
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


@app.get("/api/library")
async def library():
    entries = []
    for meta_path in JOBS_DIR.glob("*/meta.json"):
        try:
            with open(meta_path) as f:
                entries.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return {"library": entries}


@app.post("/api/transcribe")
async def transcribe(
    source_url: str = Form(default=""),
    source_title: str = Form(default=""),
    source_uploader: str = Form(default=""),
    source_webpage_url: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    job_id = str(uuid.uuid4())
    d = job_dir(job_id)

    if file is not None and file.filename:
        upload_path = d / f"upload_{file.filename}"
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source_type, source_value = "file", str(upload_path)
        default_title = Path(file.filename).stem
    elif source_url.strip():
        source_type, source_value = "url", source_url.strip()
        default_title = None
    else:
        return JSONResponse({"error": "Provide a song link or an audio file."}, status_code=400)

    jobs[job_id] = {
        "status": "queued",
        "meta": {
            "title": source_title.strip() or default_title,
            "uploader": source_uploader.strip() or None,
            "webpage_url": source_webpage_url.strip() or None,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    }

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
    label_key_signature: bool = Form(default=False),
    show_note_names: bool = Form(default=False),
):
    if not ensure_job_loaded(job_id):
        return JSONResponse({"error": "job not ready"}, status_code=400)
    j = jobs[job_id]

    events = j["events"]
    score = p.arrange_for_piano(events, difficulty)
    analyzed_key = p.detect_and_apply_key_signature(score)
    score.makeNotation(inPlace=True)
    p.apply_key_signature_labels(score, analyzed_key, label_key_signature)
    p.apply_note_name_labels(score, show_note_names)

    d = job_dir(job_id)
    xml_path = d / "render.musicxml"
    pdf_path = d / "sheet.pdf"
    p.score_to_musicxml(score, xml_path)

    meta = j.get("meta", {})
    title = meta.get("title") or "Untitled"
    source_url = meta.get("webpage_url")
    svgs = p.render_musicxml_to_svg_and_pdf(xml_path, pdf_path, title=title, difficulty=difficulty, source_url=source_url)

    return {
        "svgs": svgs,
        "pages": len(svgs),
        "pdf_url": f"api/download/{job_id}",
        "title": title,
        "difficulty": difficulty,
        "source_url": source_url,
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
    ensure_job_loaded(job_id)
    midi_path = job_dir(job_id) / f"preview_{difficulty}.mid"
    if not midi_path.exists():
        return JSONResponse({"error": "not ready yet"}, status_code=404)
    return FileResponse(midi_path, media_type="audio/midi")


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
