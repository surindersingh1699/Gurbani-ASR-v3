"""Prepare pilot dataset: segment full recordings into training-ready chunks.

Uses faster-whisper to segment full-length kirtan MP3s into 1-30s FLAC chunks,
matches segments to canonical shabad lines from STTM database, and pushes
a structured HF dataset matching the gurbani-asr-v2-dataset schema.

Usage:
    python src/data/prepare_pilot.py segment    # Whisper segment + FLAC export
    python src/data/prepare_pilot.py push       # Push to HF Hub
    python src/data/prepare_pilot.py all        # Both steps
"""

import json
import logging
import os
import sys
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.data.sikhnet import ascii_to_unicode, lookup_shabad, load_catalog

os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, "prepare_pilot.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

SEGMENTS_DIR = "data/segments"
SEGMENTS_MANIFEST = "data/manifests/pilot_segments.json"
TARGET_SR = 16000
MIN_DURATION = 1.0
MAX_DURATION = 30.0


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file and convert to mono float32 at target sample rate."""
    import subprocess
    import tempfile

    # Use ffmpeg to convert to 16kHz mono WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_path],
            capture_output=True,
            check=True,
        )
        audio, sr = sf.read(tmp_path, dtype="float32")
        return audio
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def segment_recording(
    model: WhisperModel,
    audio_path: str,
    recording_id: str,
    catalog_entry: dict,
) -> list[dict]:
    """Segment a single recording using faster-whisper.

    Returns list of segment dicts ready for dataset creation.
    """
    log.info("Segmenting %s (%s)...", recording_id, catalog_entry.get("title", "")[:50])

    # Load audio
    try:
        audio = load_audio_mono(audio_path)
    except Exception as e:
        log.warning("Failed to load %s: %s", audio_path, e)
        return []

    total_duration = len(audio) / TARGET_SR
    log.info("  Duration: %.1f min, samples: %d", total_duration / 60, len(audio))

    # Run Whisper transcription with timestamps
    segments_iter, info = model.transcribe(
        audio,
        language="pa",
        beam_size=2,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
    )

    # Collect segments
    raw_segments = list(segments_iter)
    log.info("  Whisper found %d raw segments", len(raw_segments))

    # Merge short segments and split long ones into 1-30s chunks
    chunks = []
    for seg in raw_segments:
        start = seg.start
        end = seg.end
        text = seg.text.strip()
        duration = end - start

        if duration < MIN_DURATION:
            continue

        if duration <= MAX_DURATION:
            chunks.append({"start": start, "end": end, "text": text, "duration": duration})
        else:
            # Split long segments into MAX_DURATION chunks
            pos = start
            while pos < end:
                chunk_end = min(pos + MAX_DURATION, end)
                chunk_dur = chunk_end - pos
                if chunk_dur >= MIN_DURATION:
                    chunks.append({"start": pos, "end": chunk_end, "text": text, "duration": chunk_dur})
                pos = chunk_end

    log.info("  Produced %d chunks (1-30s)", len(chunks))

    # Build segment records
    shabad_lines = catalog_entry.get("shabad_lines", [])
    canonical_text = " ".join(shabad_lines)
    segment_records = []

    for i, chunk in enumerate(chunks):
        seg_id = f"{recording_id}_{i:04d}"

        # Extract audio chunk
        start_sample = int(chunk["start"] * TARGET_SR)
        end_sample = int(chunk["end"] * TARGET_SR)
        chunk_audio = audio[start_sample:end_sample]

        # Save as FLAC
        flac_path = os.path.join(SEGMENTS_DIR, f"{seg_id}.flac")
        os.makedirs(os.path.dirname(flac_path), exist_ok=True)
        sf.write(flac_path, chunk_audio, TARGET_SR, format="FLAC")

        segment_records.append({
            "audio_path": flac_path,
            "segment_id": seg_id,
            "recording_id": recording_id,
            "start": round(chunk["start"], 3),
            "end": round(chunk["end"], 3),
            "duration": round(chunk["duration"], 3),
            "canonical_transcription": canonical_text,
            "teacher_text": chunk["text"],
            "training_label": canonical_text,
            "label_source": "sttm_database",
            "match_score": 1.0,
            "skeleton_score": 1.0,
            "quality_score": 1.0,
            "decision": "accept",
            "ang": catalog_entry.get("ang", 0),
            "style_bucket": catalog_entry.get("style_bucket", "mixed"),
            "artist_name": catalog_entry.get("artist_name", ""),
            "source_url": catalog_entry.get("url", ""),
            "shabad_id": catalog_entry.get("shabad_id"),
            "raag": catalog_entry.get("raag", ""),
            "writer": catalog_entry.get("writer", ""),
            "phase": catalog_entry.get("phase", 4),
        })

    return segment_records


def cmd_segment():
    """Segment all downloaded pilot recordings."""
    catalog = load_catalog()
    if not catalog:
        log.error("No catalog found. Run pilot first.")
        return

    # Filter to downloaded files
    ready = [e for e in catalog if os.path.exists(e.get("local_path", ""))]
    log.info("Segmenting %d recordings...", len(ready))

    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    # Load faster-whisper model
    log.info("Loading Whisper model (small, CPU)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    log.info("Model loaded")

    all_segments = []
    for i, entry in enumerate(ready):
        log.info("[%d/%d] %s", i + 1, len(ready), entry.get("recording_id", ""))
        segments = segment_recording(
            model,
            entry["local_path"],
            entry["recording_id"],
            entry,
        )
        all_segments.extend(segments)
        log.info("  Total segments so far: %d", len(all_segments))

    # Save manifest
    os.makedirs(os.path.dirname(SEGMENTS_MANIFEST), exist_ok=True)
    with open(SEGMENTS_MANIFEST, "w") as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)

    log.info("=== SEGMENTATION COMPLETE ===")
    log.info("Total segments: %d", len(all_segments))
    total_dur = sum(s["duration"] for s in all_segments)
    log.info("Total duration: %.1f hours", total_dur / 3600)
    log.info("Avg segment: %.1f seconds", total_dur / len(all_segments) if all_segments else 0)

    return all_segments


def cmd_push():
    """Push segmented pilot to HF Hub as a proper dataset."""
    try:
        from datasets import Audio, Dataset, Features, Value
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    if not os.path.exists(SEGMENTS_MANIFEST):
        log.error("No segments manifest found. Run segment first.")
        return

    with open(SEGMENTS_MANIFEST) as f:
        segments = json.load(f)

    if not segments:
        log.error("No segments in manifest.")
        return

    # Verify audio files exist
    valid = [s for s in segments if os.path.exists(s["audio_path"])]
    log.info("Pushing %d segments to HF Hub (of %d total)", len(valid), len(segments))

    # Build dataset dict matching v2-dataset schema
    ds_dict = {
        "audio": [s["audio_path"] for s in valid],
        "segment_id": [s["segment_id"] for s in valid],
        "recording_id": [s["recording_id"] for s in valid],
        "start": [s["start"] for s in valid],
        "end": [s["end"] for s in valid],
        "duration": [s["duration"] for s in valid],
        "canonical_transcription": [s["canonical_transcription"] for s in valid],
        "teacher_text": [s["teacher_text"] for s in valid],
        "training_label": [s["training_label"] for s in valid],
        "label_source": [s["label_source"] for s in valid],
        "match_score": [s["match_score"] for s in valid],
        "skeleton_score": [s["skeleton_score"] for s in valid],
        "quality_score": [s["quality_score"] for s in valid],
        "decision": [s["decision"] for s in valid],
        "ang": [s["ang"] for s in valid],
        "style_bucket": [s["style_bucket"] for s in valid],
        "artist_name": [s["artist_name"] for s in valid],
        "source_url": [s["source_url"] for s in valid],
    }

    features = Features({
        "audio": Audio(sampling_rate=16000),
        "segment_id": Value("string"),
        "recording_id": Value("string"),
        "start": Value("float32"),
        "end": Value("float32"),
        "duration": Value("float32"),
        "canonical_transcription": Value("string"),
        "teacher_text": Value("string"),
        "training_label": Value("string"),
        "label_source": Value("string"),
        "match_score": Value("float32"),
        "skeleton_score": Value("float32"),
        "quality_score": Value("float32"),
        "decision": Value("string"),
        "ang": Value("int32"),
        "style_bucket": Value("string"),
        "artist_name": Value("string"),
        "source_url": Value("string"),
    })

    ds = Dataset.from_dict(ds_dict, features=features)

    repo_id = "surindersinghssj/gurbani-asr-v3-pilot"
    hf_token = os.environ.get("HF_TOKEN")

    log.info("Pushing to %s...", repo_id)
    ds.push_to_hub(
        repo_id,
        split="train",
        token=hf_token,
    )
    log.info("=== PUSH COMPLETE: %s ===", repo_id)
    log.info("Dataset: https://huggingface.co/datasets/%s", repo_id)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "segment":
        cmd_segment()
    elif cmd == "push":
        cmd_push()
    elif cmd == "all":
        cmd_segment()
        cmd_push()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
