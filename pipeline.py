"""Audio -> piano sheet music pipeline.

Stages:
  1. acquire_audio      YouTube URL or uploaded file -> wav
  2. separate_stems     demucs, drop the drum stem (cleans up pitch detection)
  3. transcribe_to_midi basic-pitch, polyphonic audio -> MIDI notes
  4. arrange_for_piano   music21, split hands + quantize + simplify by difficulty
  5. apply_adjustments   force sharps/flats spelling, add note-name labels
  6. render_score        MusicXML -> SVG (preview) + PDF (download) via verovio/cairosvg
"""

import io
import subprocess
from pathlib import Path

import cairosvg
import imageio_ffmpeg
import verovio
from music21 import chord, clef, converter, key, meter, note, pitch, stream

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

SHARP_RESPELL = {"C-": "B", "D-": "C#", "E-": "D#", "F-": "E", "G-": "F#", "A-": "G#", "B-": "A#"}
FLAT_RESPELL = {"C#": "D-", "D#": "E-", "F#": "G-", "G#": "A-", "A#": "B-"}

DIFFICULTY_SETTINGS = {
    "easy": {
        "grid_divisor": 1,       # quarter-note grid
        "rh_max_notes": 1,       # melody only
        "lh_max_notes": 1,       # single bass note
        "lh_grid_divisor": 0.25, # left hand only changes once per measure (whole note)
    },
    "medium": {
        "grid_divisor": 2,       # eighth-note grid
        "rh_max_notes": 2,
        "lh_max_notes": 3,
        "lh_grid_divisor": 1,    # left hand on the quarter-note grid
    },
    "hard": {
        "grid_divisor": 4,       # sixteenth-note grid
        "rh_max_notes": 4,
        "lh_max_notes": 5,
        "lh_grid_divisor": 2,
    },
}

MIDDLE_C = 60


def search_soundcloud(query: str, limit: int = 8) -> list[dict]:
    cmd = [
        "yt-dlp", "--no-warnings", "--flat-playlist", "-J",
        f"scsearch{limit}:{query}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Search failed: {result.stderr[-500:]}")
    import json
    data = json.loads(result.stdout)
    return [
        {
            "title": e.get("title"),
            "uploader": e.get("uploader"),
            "duration": e.get("duration"),
            "url": e.get("url"),
        }
        for e in data.get("entries", [])
    ]


def acquire_audio_from_url(url: str, out_wav: Path) -> None:
    """Downloads any yt-dlp-supported URL's audio as wav (SoundCloud works
    reliably from server IPs; YouTube usually doesn't — see the bot-check
    message below). Raises RuntimeError with a user-actionable message on
    that known failure mode."""
    cmd = [
        "yt-dlp",
        "--ffmpeg-location", FFMPEG_PATH,
        "-x", "--audio-format", "wav",
        "-o", str(out_wav.with_suffix("")) + ".%(ext)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr
        if "Sign in to confirm" in stderr or "not a bot" in stderr:
            raise RuntimeError(
                "This host blocked the download (bot-check on server IPs, common for YouTube). "
                "Try searching SoundCloud instead, or run this on your own computer and upload the file:\n"
                f"  yt-dlp -x --audio-format wav \"{url}\""
            )
        raise RuntimeError(f"yt-dlp failed: {stderr[-800:]}")
    if not out_wav.exists():
        candidates = list(out_wav.parent.glob(out_wav.stem + ".*"))
        if candidates:
            candidates[0].rename(out_wav)
        else:
            raise RuntimeError("yt-dlp reported success but produced no audio file.")


def convert_to_wav(src_path: Path, out_wav: Path) -> None:
    cmd = [FFMPEG_PATH, "-y", "-i", str(src_path), "-ac", "1", "-ar", "44100", str(out_wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-800:]}")


def separate_stems(in_wav: Path, work_dir: Path) -> Path:
    """Runs demucs two-stems separation and returns the path to the
    'no_drums' stem (vocals+bass+other), which transcribes far more
    cleanly than a mix with percussion in it."""
    import demucs.separate

    out_dir = work_dir / "demucs_out"
    out_dir.mkdir(exist_ok=True)
    demucs.separate.main([
        "-n", "htdemucs",
        "--two-stems", "drums",
        "-o", str(out_dir),
        str(in_wav),
    ])
    track_name = in_wav.stem
    no_drums = out_dir / "htdemucs" / track_name / "no_drums.wav"
    if not no_drums.exists():
        raise RuntimeError("Demucs did not produce the expected stem output.")
    return no_drums


def transcribe_to_midi(audio_path: Path, out_midi: Path) -> None:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    _, midi_data, _ = predict(str(audio_path), model_or_model_path=ICASSP_2022_MODEL_PATH)
    midi_data.write(str(out_midi))


def _grid_size(divisor: float) -> float:
    """divisor >= 1 means that many subdivisions per quarter note
    (2 -> eighth notes). divisor < 1 means that many quarter notes per
    grid step (0.25 -> one grid step per whole note). Both cases reduce
    to the same reciprocal."""
    return 1.0 / divisor


def extract_note_events(midi_path: Path) -> list[tuple[float, float, int, int]]:
    """Returns (start_ql, end_ql, midi_pitch, velocity) in quarter-length
    units at the score's tempo, by parsing through music21 (which handles
    the tempo map for us)."""
    score = converter.parse(str(midi_path))
    events = []
    for n in score.flatten().notes:
        pitches = n.pitches if hasattr(n, "pitches") else [n.pitch]
        vel = getattr(n.volume, "velocity", None) or 80
        for p in pitches:
            events.append((float(n.offset), float(n.offset + n.duration.quarterLength), p.midi, vel))
    return events


def arrange_for_piano(events: list[tuple[float, float, int, int]], difficulty: str) -> stream.Score:
    settings = DIFFICULTY_SETTINGS[difficulty]
    rh_grid = _grid_size(settings["grid_divisor"])
    lh_grid = _grid_size(settings["lh_grid_divisor"])

    rh_events = [e for e in events if e[2] >= MIDDLE_C]
    lh_events = [e for e in events if e[2] < MIDDLE_C]

    rh_part = _bucket_and_build_part(rh_events, rh_grid, settings["rh_max_notes"], keep="highest")
    lh_part = _bucket_and_build_part(lh_events, lh_grid, settings["lh_max_notes"], keep="lowest")

    score = stream.Score()
    rh_part.insert(0, clef.TrebleClef())
    lh_part.insert(0, clef.BassClef())
    rh_part.insert(0, meter.TimeSignature("4/4"))
    lh_part.insert(0, meter.TimeSignature("4/4"))
    score.insert(0, rh_part)
    score.insert(0, lh_part)
    return score


def _bucket_and_build_part(events, grid: float, max_notes: int, keep: str) -> stream.Part:
    part = stream.Part()
    if not events:
        r = note.Rest(quarterLength=4.0)
        part.append(r)
        return part

    buckets: dict[float, list[tuple[int, int]]] = {}
    max_offset = 0.0
    for start, end, midi_pitch, vel in events:
        q_start = round(start / grid) * grid
        buckets.setdefault(q_start, []).append((midi_pitch, vel))
        max_offset = max(max_offset, q_start)

    sorted_offsets = sorted(buckets.keys())
    for i, off in enumerate(sorted_offsets):
        pitches_here = buckets[off]
        if keep == "highest":
            pitches_here = sorted(pitches_here, key=lambda pv: -pv[0])[:max_notes]
        else:
            pitches_here = sorted(pitches_here, key=lambda pv: pv[0])[:max_notes]

        next_off = sorted_offsets[i + 1] if i + 1 < len(sorted_offsets) else off + grid
        dur = max(next_off - off, grid)

        gap = off - (part.highestTime)
        if gap > 1e-6:
            part.append(note.Rest(quarterLength=gap))

        pitch_names = [pitch.Pitch(midi=p).nameWithOctave for p, _ in pitches_here]
        if len(pitch_names) == 1:
            el = note.Note(pitch_names[0], quarterLength=dur)
        else:
            el = chord.Chord(pitch_names, quarterLength=dur)
        part.append(el)

    return part


def apply_accidental_mode(score: stream.Score, mode: str) -> None:
    """mode: 'sharps', 'flats', or 'original'. Forces a consistent
    enharmonic spelling and pins the key signature to C so every
    accidental is explicit rather than implied by the key."""
    if mode not in ("sharps", "flats"):
        return
    target_table = SHARP_RESPELL if mode == "sharps" else FLAT_RESPELL

    for part in score.parts:
        part.insert(0, key.KeySignature(0))
        for el in part.flatten().notesAndRests:
            pitches_to_fix = el.pitches if isinstance(el, chord.Chord) else ([el.pitch] if isinstance(el, note.Note) else [])
            for p in pitches_to_fix:
                name = p.name
                if mode == "sharps" and name in target_table:
                    p.name = target_table[name]
                elif mode == "flats" and name in target_table:
                    p.name = target_table[name]
                if p.accidental is not None:
                    p.accidental.displayStatus = True


def apply_note_name_labels(score: stream.Score, show: bool) -> None:
    if not show:
        return
    for part in score.parts:
        for el in part.flatten().notes:
            if isinstance(el, chord.Chord):
                names = sorted({p.name for p in el.pitches})
                el.lyric = ",".join(names)
            elif isinstance(el, note.Note):
                el.lyric = el.pitch.name


def score_to_musicxml(score: stream.Score, out_path: Path) -> None:
    score.write("musicxml", fp=str(out_path))


def render_musicxml_to_svg_and_pdf(musicxml_path: Path, svg_out: Path, pdf_out: Path) -> int:
    tk = verovio.toolkit()
    tk.loadFile(str(musicxml_path))
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": True})
    n_pages = tk.getPageCount()

    svgs = []
    for i in range(1, n_pages + 1):
        svg = tk.renderToSVG(i)
        svgs.append(svg)
    svg_out.write_text(svgs[0])

    pdf_bytes_pages = [cairosvg.svg2pdf(bytestring=s.encode("utf-8")) for s in svgs]
    if len(pdf_bytes_pages) == 1:
        pdf_out.write_bytes(pdf_bytes_pages[0])
    else:
        _merge_pdfs(pdf_bytes_pages, pdf_out)
    return n_pages


def _merge_pdfs(pdf_byte_list, out_path: Path) -> None:
    from pypdf import PdfWriter, PdfReader

    writer = PdfWriter()
    for pdf_bytes in pdf_byte_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    with open(out_path, "wb") as f:
        writer.write(f)
