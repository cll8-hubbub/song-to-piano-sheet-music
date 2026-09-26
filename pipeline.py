"""Audio -> piano sheet music pipeline.

Stages:
  1. acquire_audio      SoundCloud/YouTube URL or uploaded file -> wav (+ metadata)
  2. separate_stems     demucs, drop the drum stem (cleans up pitch detection)
  3. transcribe_to_midi basic-pitch, polyphonic audio -> MIDI notes
  4. arrange_for_piano   music21, split hands + quantize + simplify by difficulty
  5. apply_adjustments   detect key signature, add note/key labels
  6. render_score        MusicXML -> SVG (preview) + labeled PDF (download) via verovio
"""

import io
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import verovio
from music21 import chord, clef, key, meter, note, pitch, stream, tempo
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as pdfcanvas
from svglib.svglib import svg2rlg

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

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
    data = json.loads(result.stdout)
    return [
        {
            "title": e.get("title"),
            "uploader": e.get("uploader"),
            "duration": e.get("duration"),
            "url": e.get("url"),
            "webpage_url": e.get("webpage_url"),
        }
        for e in data.get("entries", [])
    ]


def acquire_audio_from_url(url: str, out_wav: Path) -> dict:
    """Downloads any yt-dlp-supported URL's audio as wav (SoundCloud works
    reliably from server IPs; YouTube usually doesn't — see the bot-check
    message below). Raises RuntimeError with a user-actionable message on
    that known failure mode. Returns whatever title/uploader/webpage_url
    metadata yt-dlp could extract, for labeling the sheet music later."""
    info_result = subprocess.run(
        ["yt-dlp", "--no-warnings", "-J", url], capture_output=True, text=True
    )
    metadata = {"title": None, "uploader": None, "webpage_url": url}
    if info_result.returncode == 0:
        try:
            info = json.loads(info_result.stdout)
            metadata["title"] = info.get("title")
            metadata["uploader"] = info.get("uploader")
            metadata["webpage_url"] = info.get("webpage_url", url)
        except (json.JSONDecodeError, KeyError):
            pass

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
    return metadata


def estimate_tempo_from_audio(wav_path: Path) -> tuple[float, float]:
    """Beat-tracks the actual audio waveform -- kept from before drum
    removal, since percussive transients are the strongest tempo cue --
    rather than guessing from MIDI note patterns alone. More robust
    against the octave errors (mistaking double or half the real tempo)
    that simple note-onset heuristics are prone to.

    Returns (bpm, phase_seconds). phase_seconds is where the first real
    beat falls, modulo one beat period -- used to align the notation grid
    to the song's actual downbeat instead of assuming beat 1 lands at the
    very first instant of the audio file (it almost never does)."""
    import librosa

    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    except Exception:
        return 120.0, 0.0
    if bpm <= 0:
        return 120.0, 0.0
    # Nudge octave-doubled/halved estimates toward a typical song tempo range.
    while bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    period = 60.0 / bpm
    phase = float(beat_times[0]) % period if len(beat_times) else 0.0
    return bpm, phase


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


def extract_note_events(midi_path: Path, bpm: float, phase: float = 0.0) -> list[tuple[float, float, int, int]]:
    """Returns (start_ql, end_ql, midi_pitch, velocity) in quarter-length
    units, using the given real tempo (estimated from the actual audio via
    estimate_tempo_from_audio) to build the axis. Reads pretty_midi's raw
    note timing directly in seconds rather than going through music21's
    tempo-relative parsing: basic-pitch always tags its MIDI output with a
    flat placeholder 120bpm that has no relation to the actual song, so
    quantizing notation against that would misalign every note against the
    song's real beat.

    `phase` shifts the axis so beat 1 lands on a real detected beat instead
    of assuming it falls at the very first instant of the audio."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    seconds_to_ql = bpm / 60.0

    events = []
    for instrument in pm.instruments:
        for n in instrument.notes:
            start = max(0.0, n.start - phase)
            end = max(0.0, n.end - phase)
            events.append((start * seconds_to_ql, end * seconds_to_ql, n.pitch, n.velocity))
    return events


def _cluster_onsets(events: list[tuple[float, float, int, int]], tolerance_ql: float) -> list[tuple[float, float, int, int]]:
    """Notes meant to sound together rarely get reported with the exact
    same onset by the transcription model -- a few tens of milliseconds
    apart is normal detection jitter. Snap onsets within a small tolerance
    to the same time so a rounding boundary downstream can't split them
    into separate chords."""
    if not events:
        return events
    sorted_events = sorted(events, key=lambda e: e[0])
    clustered = []
    cluster_repr = sorted_events[0][0]
    prev_original = sorted_events[0][0]
    for start, end, midi_pitch, vel in sorted_events:
        if start - prev_original > tolerance_ql:
            cluster_repr = start
        prev_original = start
        clustered.append((cluster_repr, end, midi_pitch, vel))
    return clustered


def arrange_for_piano(events: list[tuple[float, float, int, int]], difficulty: str, bpm: float = 120.0) -> stream.Score:
    settings = DIFFICULTY_SETTINGS[difficulty]
    rh_grid = _grid_size(settings["grid_divisor"])
    lh_grid = _grid_size(settings["lh_grid_divisor"])

    onset_tolerance_ql = 0.07 * (bpm / 60.0)  # ~70ms, a typical "simultaneous" threshold
    events = _cluster_onsets(events, onset_tolerance_ql)

    rh_events = [e for e in events if e[2] >= MIDDLE_C]
    lh_events = [e for e in events if e[2] < MIDDLE_C]

    rh_part = _bucket_and_build_part(rh_events, rh_grid, settings["rh_max_notes"], keep="highest")
    lh_part = _bucket_and_build_part(lh_events, lh_grid, settings["lh_max_notes"], keep="lowest")

    score = stream.Score()
    rh_part.insert(0, clef.TrebleClef())
    lh_part.insert(0, clef.BassClef())
    rh_part.insert(0, meter.TimeSignature("4/4"))
    lh_part.insert(0, meter.TimeSignature("4/4"))
    rh_part.insert(0, tempo.MetronomeMark(number=round(bpm)))
    score.insert(0, rh_part)
    score.insert(0, lh_part)
    return score


def _bucket_and_build_part(events, grid: float, max_notes: int, keep: str) -> stream.Part:
    part = stream.Part()
    if not events:
        r = note.Rest(quarterLength=4.0)
        part.append(r)
        return part

    buckets: dict[float, list[tuple[int, int, float]]] = {}
    for start, end, midi_pitch, vel in events:
        q_start = round(start / grid) * grid
        buckets.setdefault(q_start, []).append((midi_pitch, vel, end))

    sorted_offsets = sorted(buckets.keys())
    for i, off in enumerate(sorted_offsets):
        pitches_here = buckets[off]
        if keep == "highest":
            pitches_here = sorted(pitches_here, key=lambda pv: -pv[0])[:max_notes]
        else:
            pitches_here = sorted(pitches_here, key=lambda pv: pv[0])[:max_notes]

        next_off = sorted_offsets[i + 1] if i + 1 < len(sorted_offsets) else off + grid
        # Duration comes from the note's real detected release time (quantized to
        # the grid), not just "until the next note starts" -- capped so it never
        # overlaps the next note, and floored to at least one grid unit. Rounding
        # is biased slightly toward extending rather than truncating: a note
        # detected as ending just shy of the next grid line is far more often a
        # note that really was held through it (piano decay/pedal makes offset
        # detection land a little early) than a note that really should cut off
        # a whole grid unit short, which is what plain round-to-nearest would do
        # and what produces spurious one-grid-unit rests.
        real_end = max(end for _, _, end in pitches_here)
        q_end = math.floor(real_end / grid + 0.65) * grid
        dur = max(min(q_end, next_off) - off, grid)

        gap = off - (part.highestTime)
        if gap > 1e-6:
            part.append(note.Rest(quarterLength=gap))

        pitch_names = [pitch.Pitch(midi=p).nameWithOctave for p, _, _ in pitches_here]
        velocity = max(v for _, v, _ in pitches_here)
        if len(pitch_names) == 1:
            el = note.Note(pitch_names[0], quarterLength=dur)
        else:
            el = chord.Chord(pitch_names, quarterLength=dur)
        el.volume.velocity = velocity
        part.append(el)

    return part


def detect_and_apply_key_signature(score: stream.Score) -> key.Key:
    """Analyzes the actual pitch content to guess the real key (e.g. D major,
    2 sharps) and writes a proper key signature onto both hands, instead of
    always reading as plain C. Call this before makeNotation() so accidental
    display is computed against the real key context."""
    try:
        analyzed_key = score.analyze("key")
    except Exception:
        analyzed_key = key.Key("C")
    for part in score.parts:
        part.insert(0, key.KeySignature(analyzed_key.sharps))
    return analyzed_key


def apply_key_signature_labels(score: stream.Score, analyzed_key: key.Key, show: bool) -> None:
    """Beginner aid: a real key signature only marks sharps/flats once, at
    the clef, the way an experienced reader expects. This optionally marks
    every affected note throughout the piece too (the way a beginner might
    pencil in a reminder), on top of the real key signature -- it doesn't
    invent extra sharps/flats, it just repeats the ones the key already
    implies everywhere they apply."""
    if not show or analyzed_key.sharps == 0:
        return
    altered_steps = {p.step for p in analyzed_key.alteredPitches}
    accidental_name = "sharp" if analyzed_key.sharps > 0 else "flat"

    for part in score.parts:
        for el in part.flatten().notes:
            pitches_here = el.pitches if isinstance(el, chord.Chord) else [el.pitch]
            for p in pitches_here:
                if p.step in altered_steps:
                    if p.accidental is None:
                        p.accidental = pitch.Accidental(accidental_name)
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


def render_musicxml_to_svg_and_pdf(
    musicxml_path: Path,
    pdf_out: Path,
    title: str = "Untitled",
    difficulty: str = "medium",
    source_url: str | None = None,
) -> list[str]:
    tk = verovio.toolkit()
    tk.loadFile(str(musicxml_path))
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "header": "none", "footer": "none"})
    n_pages = tk.getPageCount()

    svgs = [tk.renderToSVG(i) for i in range(1, n_pages + 1)]

    header_h, footer_h = 50, 24
    c = pdfcanvas.Canvas(str(pdf_out))
    for page_num, svg_str in enumerate(svgs, start=1):
        drawing = svg2rlg(io.BytesIO(svg_str.encode("utf-8")))
        page_w, page_h = drawing.width, drawing.height + header_h + footer_h
        c.setPageSize((page_w, page_h))
        renderPDF.draw(drawing, c, 0, footer_h)

        c.setFont("Helvetica-Bold", 13)
        c.drawString(20, page_h - 28, title or "Untitled")
        c.setFont("Helvetica", 9)
        c.drawString(20, page_h - 42, f"Difficulty: {difficulty.capitalize()}")

        if source_url:
            link_text = "Listen to the original"
            text_w = c.stringWidth(link_text, "Helvetica", 9)
            x0 = page_w - 20 - text_w
            c.setFillColorRGB(0.2, 0.2, 0.6)
            c.setFont("Helvetica", 9)
            c.drawString(x0, page_h - 42, link_text)
            c.linkURL(source_url, (x0, page_h - 46, x0 + text_w, page_h - 34), relative=0)
            c.setFillColorRGB(0, 0, 0)

        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 10, f"Page {page_num} of {n_pages}")
        c.showPage()
    c.save()
    return svgs
