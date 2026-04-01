#!/usr/bin/env python3
"""Whisper model comparison — pod-side script.

Self-contained script that runs on RunPod. Transcribes 4 kirtan tracks with
specified Whisper configs, aligns word timestamps to canonical STTM tuks,
cuts FLAC segments, and pushes results to HF Hub as named splits.

Usage:
    python3 scripts/whisper_model_comparison.py \
        --model Systran/faster-whisper-large-v3-turbo \
        --configs v3_turbo_base,v3_turbo_rep,v3_turbo_temp0,v3_turbo_beam5,v3_turbo_nocond \
        --hf-repo surindersinghssj/gurbani-asr-model-comparison
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
import requests
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─── Mool Mantar prompt ─────────────────────────────────────────────────────
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

TARGET_SR = 16000
SEGMENTS_DIR = "data/segments"
DB_PATH = "database.sqlite"

# ─── Config definitions ─────────────────────────────────────────────────────
COMMON_PARAMS = dict(
    language="pa",
    word_timestamps=True,
    vad_filter=False,
    initial_prompt=MOOL_MANTAR,
)

CONFIG_OVERRIDES = {
    "large_v2":        {"beam_size": 5, "condition_on_previous_text": True},
    "large_v3":        {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_base":   {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_rep":    {"beam_size": 5, "condition_on_previous_text": True, "repetition_penalty": 1.1},
    "v3_turbo_temp0":  {"beam_size": 5, "condition_on_previous_text": True, "temperature": 0},
    "v3_turbo_beam5":  {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_nocond": {"beam_size": 5, "condition_on_previous_text": False},
}

# ─── STTM ASCII → Unicode (embedded from sikhnet.py) ────────────────────────
_ASCII_REPLACEMENTS = [("<>", "\u0A74"), ("<", "\u0A74"), (">", "\u262C")]

_ASCII_MAP = {
    "a": "\u0A73", "A": "\u0A05", "e": "\u0A72",
    "s": "\u0A38", "S": "\u0A36", "h": "\u0A39", "H": "\u0A4D\u0A39",
    "k": "\u0A15", "K": "\u0A16", "g": "\u0A17", "G": "\u0A18", "|": "\u0A19",
    "c": "\u0A1A", "C": "\u0A1B", "j": "\u0A1C", "J": "\u0A1D", "\\": "\u0A1E",
    "t": "\u0A1F", "T": "\u0A20", "f": "\u0A21", "F": "\u0A22", "x": "\u0A23",
    "q": "\u0A24", "Q": "\u0A25", "d": "\u0A26", "D": "\u0A27", "n": "\u0A28",
    "p": "\u0A2A", "P": "\u0A2B", "b": "\u0A2C", "B": "\u0A2D", "m": "\u0A2E",
    "X": "\u0A2F", "r": "\u0A30", "l": "\u0A32", "L": "\u0A33", "v": "\u0A35",
    "V": "\u0A5C",
    "w": "\u0A3E", "W": "\u0A3E\u0A02", "i": "\u0A3F", "I": "\u0A40",
    "u": "\u0A41", "U": "\u0A42", "y": "\u0A47", "Y": "\u0A48",
    "o": "\u0A4B", "O": "\u0A4C", "E": "\u0A13",
    "M": "\u0A70", "N": "\u0A02", "`": "\u0A71", "~": "\u0A71", "@": "\u0A51",
    "z": "\u0A5B", "Z": "\u0A5A", "^": "\u0A59", "&": "\u0A5E",
    "R": "\u0A4D\u0A30",
    "0": "\u0A66", "1": "\u0A67", "2": "\u0A68", "3": "\u0A69", "4": "\u0A6A",
    "5": "\u0A6B", "6": "\u0A6C", "7": "\u0A6D", "8": "\u0A6E", "9": "\u0A6F",
    "[": "\u0964", "]": "\u0965",
    "\u00e6": "\u0A3C", "\u00a1": "\u0A74",
    "\u0192": "\u0A28\u0A42\u0A70", "\u0153": "\u0A4D\u0A24",
    "\u00cd": "\u0A4D\u0A35", "\u00cf": "\u0A75", "\u00d2": "\u0965",
    "\u00da": "\u0A03", "\u02c6": "\u0A02", "\u02dc": "\u0A4D\u0A28",
    "\u00a7": "\u0A4D\u0A39\u0A42", "\u00a4": "\u0A71",
    "\u00e7": "\u0A4D\u0A1A", "\u2020": "\u0A4D\u0A1F",
    "\u00fc": "\u0A41", "\u00ae": "\u0A4D\u0A30",
    "\u00b4": "\u0A75", "\u00a8": "\u0A42", "\u00b5": "\u0A70",
}

_ASCII_NULLIFY = set("\u00c6\u00d8\u00ff\u0152\u2030\u00d3\u00d4")

_UNICODE_SANITIZE = [
    ("\u0A73\u0A4B", "\u0A13"), ("\u0A05\u0A3E", "\u0A06"),
    ("\u0A72\u0A3F", "\u0A07"), ("\u0A72\u0A40", "\u0A08"),
    ("\u0A73\u0A41", "\u0A09"), ("\u0A73\u0A42", "\u0A0A"),
    ("\u0A72\u0A47", "\u0A0F"), ("\u0A05\u0A48", "\u0A10"),
    ("\u0A05\u0A4C", "\u0A14"),
]

_BASE_LETTERS = set(
    "\u0A15\u0A16\u0A17\u0A18\u0A19\u0A1A\u0A1B\u0A1C\u0A1D\u0A1E"
    "\u0A1F\u0A20\u0A21\u0A22\u0A23\u0A24\u0A25\u0A26\u0A27\u0A28"
    "\u0A2A\u0A2B\u0A2C\u0A2D\u0A2E\u0A2F\u0A30\u0A32\u0A33\u0A35"
    "\u0A36\u0A38\u0A39\u0A59\u0A5A\u0A5B\u0A5C\u0A5E"
    "\u0A05\u0A06\u0A07\u0A08\u0A09\u0A0A\u0A0F\u0A10\u0A13\u0A14"
    "\u0A72\u0A73\u0A74"
)
SIHARI = "\u0A3F"


def ascii_to_unicode(text: str) -> str:
    if not text:
        return text
    for ascii_seq, uni in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_seq, uni)
    result = []
    for ch in text:
        if ch in _ASCII_NULLIFY:
            continue
        result.append(_ASCII_MAP.get(ch, ch))
    text = "".join(result)
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] == SIHARI and chars[i + 1] in _BASE_LETTERS:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    text = "".join(chars)
    for raw, clean in _UNICODE_SANITIZE:
        text = text.replace(raw, clean)
    text = text.replace(";", "").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lookup_shabad(sttm_id: int, db_path: str = DB_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT s.id, sec.name_english, w.name_english, l.source_page
            FROM shabads s
            JOIN lines l ON l.shabad_id = s.id
            JOIN sections sec ON s.section_id = sec.id
            JOIN writers w ON s.writer_id = w.id
            WHERE s.sttm_id = ? ORDER BY l.order_id LIMIT 1""",
            (sttm_id,),
        )
        meta = cur.fetchone()
        if not meta:
            return None
        shabad_db_id, raag, writer, ang = meta
        cur.execute(
            "SELECT gurmukhi FROM lines WHERE shabad_id = ? ORDER BY order_id",
            (shabad_db_id,),
        )
        lines_ascii = [row[0] for row in cur.fetchall()]
        lines_unicode = [ascii_to_unicode(line) for line in lines_ascii]
        return {"shabad_lines": lines_unicode, "ang": ang, "raag": raag, "writer": writer}
    finally:
        conn.close()


# ─── Audio utilities ─────────────────────────────────────────────────────────

def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_path],
            capture_output=True, check=True,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
        return audio
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_track(url: str, output_path: str) -> bool:
    if os.path.exists(output_path):
        return True
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        resp = requests.get(url, allow_redirects=True, timeout=300,
                           headers={"User-Agent": "GurbaniASR/1.0"})
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        log.info("Downloaded %s (%.1f MB)", os.path.basename(output_path), len(resp.content) / 1e6)
        return True
    except Exception as e:
        log.error("Download failed %s: %s", url, e)
        return False


# ─── Gurmukhi text normalization for matching ────────────────────────────────

MATRAS = set("ਾਿੀੁੂੇੈੋੌੰੱ਼ਁ")


def normalize_gurmukhi(text: str) -> str:
    return "".join(c for c in text if c not in MATRAS).lower().strip()


def char_f1(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    if not overlap:
        return 0.0
    precision = len(overlap) / len(words_a)
    recall = len(overlap) / len(words_b)
    return 2 * precision * recall / (precision + recall)


# ─── Word-to-tuk alignment ──────────────────────────────────────────────────

def get_word_dicts(segments_iter) -> list[dict]:
    """Extract word-level dicts from faster-whisper segments iterator."""
    words = []
    for segment in segments_iter:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
    return words


def match_words_to_tuks(word_dicts: list[dict], canonical_lines: list[str],
                        min_score: float = 0.2) -> list[dict]:
    """Align Whisper word sequences to canonical tuk lines.

    Uses sequential scanning with matra-normalized word F1 scoring.
    Handles kirtan repetitions: same tuk may match multiple times.
    """
    if not word_dicts or not canonical_lines:
        return []

    segments = []
    word_ptr = 0

    for tuk_idx, tuk in enumerate(canonical_lines):
        tuk_norm = normalize_gurmukhi(tuk)
        tuk_words = tuk_norm.split()
        n_tuk = len(tuk_words)
        if n_tuk == 0:
            continue

        repetition = 0
        search_from = max(0, word_ptr - 3)

        # Look for this tuk (and repetitions) in the remaining words
        while search_from < len(word_dicts) - max(1, n_tuk // 2):
            best_score = 0.0
            best_start = search_from
            best_end = search_from + 1

            # Try windows of varying length around the tuk length
            min_win = max(1, n_tuk - 3)
            max_win = min(n_tuk * 3 + 2, len(word_dicts) - search_from)

            for win_size in range(min_win, max_win + 1):
                end = search_from + win_size
                if end > len(word_dicts):
                    break
                win_text = normalize_gurmukhi(
                    " ".join(word_dicts[i]["word"] for i in range(search_from, end))
                )
                score = char_f1(win_text, tuk_norm)
                if score > best_score:
                    best_score = score
                    best_start = search_from
                    best_end = end

            if best_score >= min_score:
                w_slice = word_dicts[best_start:best_end]
                segments.append({
                    "tuk_index": tuk_idx,
                    "canonical_line": tuk,
                    "whisper_text": " ".join(w["word"] for w in w_slice),
                    "start": w_slice[0]["start"],
                    "end": w_slice[-1]["end"],
                    "match_score": round(best_score, 4),
                    "avg_confidence": round(mean(w["probability"] for w in w_slice), 4),
                    "match_method": "word_f1",
                    "repetition": repetition,
                })
                search_from = best_end
                word_ptr = best_end
                repetition += 1
            else:
                # Try advancing search window
                search_from += max(1, n_tuk // 2)
                if search_from - word_ptr > n_tuk * 5:
                    break  # Give up on this tuk, move to next

    return segments


def get_whisper_text_for_segment(word_dicts: list[dict], start: float, end: float) -> str:
    return " ".join(
        w["word"] for w in word_dicts
        if w["start"] >= start - 0.05 and w["end"] <= end + 0.05
    )


# ─── Main pipeline ──────────────────────────────────────────────────────────

def transcribe_and_align(model, audio: np.ndarray, config_name: str,
                         catalog_entry: dict, recording_id: str) -> list[dict]:
    """Transcribe audio with given config, align to canonical tuks, cut segments."""
    params = {**COMMON_PARAMS, **CONFIG_OVERRIDES[config_name]}
    log.info("  Transcribing with config=%s params=%s", config_name, {k: v for k, v in params.items() if k != "initial_prompt"})

    segments_iter, info = model.transcribe(audio, **params)
    word_dicts = get_word_dicts(segments_iter)
    log.info("  Got %d words, language=%s prob=%.2f", len(word_dicts), info.language, info.language_probability)

    if not word_dicts:
        log.warning("  No words detected!")
        return []

    # Get canonical shabad lines
    shabad_lines = catalog_entry.get("shabad_lines", [])
    canonical_shabad = " ".join(shabad_lines)

    # Align words to tuks
    matched = match_words_to_tuks(word_dicts, shabad_lines)
    log.info("  Matched %d segments to %d canonical tuks", len(matched), len(shabad_lines))

    # Build segment records with audio
    segment_records = []
    seg_dir = os.path.join(SEGMENTS_DIR, config_name)
    os.makedirs(seg_dir, exist_ok=True)

    for i, seg in enumerate(matched):
        seg_id = f"{recording_id}_{i:04d}"
        start_sample = max(0, int(seg["start"] * TARGET_SR))
        end_sample = min(len(audio), int(seg["end"] * TARGET_SR))
        chunk_audio = audio[start_sample:end_sample]
        duration = len(chunk_audio) / TARGET_SR

        if duration < 0.5:
            continue

        flac_path = os.path.join(seg_dir, f"{seg_id}.flac")
        sf.write(flac_path, chunk_audio, TARGET_SR, format="FLAC")

        # Training text: the canonical line (what was actually sung)
        training_text = seg["canonical_line"]

        segment_records.append({
            "audio_path": flac_path,
            "whisper_text": seg["whisper_text"],
            "canonical_shabad": canonical_shabad,
            "canonical_line": seg["canonical_line"],
            "training_text": training_text,
            "config_name": config_name,
            "model_name": catalog_entry.get("_model_name", ""),
            "segment_id": seg_id,
            "recording_id": recording_id,
            "tuk_index": seg["tuk_index"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(duration, 3),
            "match_score": seg["match_score"],
            "avg_confidence": seg["avg_confidence"],
            "match_method": seg["match_method"],
            "repetition": seg["repetition"],
            "ang": catalog_entry.get("ang", 0),
            "raag": catalog_entry.get("raag", ""),
            "writer": catalog_entry.get("writer", ""),
            "style_bucket": catalog_entry.get("style_bucket", ""),
            "artist_name": catalog_entry.get("artist_name", ""),
            "source_url": catalog_entry.get("url", ""),
        })

    return segment_records


def push_to_hf(segments: list[dict], config_name: str, hf_repo: str):
    """Push segments as a named split to HF Hub with playable audio."""
    from datasets import Audio, Dataset, Features, Value

    if not segments:
        log.warning("No segments to push for %s", config_name)
        return

    ds_dict = {
        "whisper_text": [s["whisper_text"] for s in segments],
        "canonical_shabad": [s["canonical_shabad"] for s in segments],
        "canonical_line": [s["canonical_line"] for s in segments],
        "training_text": [s["training_text"] for s in segments],
        "audio": [s["audio_path"] for s in segments],
        "config_name": [s["config_name"] for s in segments],
        "model_name": [s["model_name"] for s in segments],
        "segment_id": [s["segment_id"] for s in segments],
        "recording_id": [s["recording_id"] for s in segments],
        "tuk_index": [s["tuk_index"] for s in segments],
        "start": [s["start"] for s in segments],
        "end": [s["end"] for s in segments],
        "duration": [s["duration"] for s in segments],
        "match_score": [s["match_score"] for s in segments],
        "avg_confidence": [s["avg_confidence"] for s in segments],
        "match_method": [s["match_method"] for s in segments],
        "repetition": [s["repetition"] for s in segments],
        "ang": [s["ang"] for s in segments],
        "raag": [s["raag"] for s in segments],
        "writer": [s["writer"] for s in segments],
        "style_bucket": [s["style_bucket"] for s in segments],
        "artist_name": [s["artist_name"] for s in segments],
        "source_url": [s["source_url"] for s in segments],
    }

    features = Features({
        "whisper_text": Value("string"),
        "canonical_shabad": Value("string"),
        "canonical_line": Value("string"),
        "training_text": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "config_name": Value("string"),
        "model_name": Value("string"),
        "segment_id": Value("string"),
        "recording_id": Value("string"),
        "tuk_index": Value("int32"),
        "start": Value("float32"),
        "end": Value("float32"),
        "duration": Value("float32"),
        "match_score": Value("float32"),
        "avg_confidence": Value("float32"),
        "match_method": Value("string"),
        "repetition": Value("int32"),
        "ang": Value("int32"),
        "raag": Value("string"),
        "writer": Value("string"),
        "style_bucket": Value("string"),
        "artist_name": Value("string"),
        "source_url": Value("string"),
    })

    ds = Dataset.from_dict(ds_dict, features=features)
    hf_token = os.environ.get("HF_TOKEN")

    log.info("Pushing %d segments to %s split=%s", len(segments), hf_repo, config_name)
    ds.push_to_hub(hf_repo, split=config_name, token=hf_token)
    log.info("Push complete for %s", config_name)


def write_report(segments: list[dict], config_name: str, model_name: str,
                 params: dict, catalog: list[dict]) -> dict:
    """Write per-config report JSON."""
    tracks = []
    for entry in catalog:
        rec_id = entry["recording_id"]
        track_segs = [s for s in segments if s["recording_id"] == rec_id]
        shabad_lines = entry.get("shabad_lines", [])

        matched_tuks = set(s["tuk_index"] for s in track_segs if s["repetition"] == 0)
        tracks.append({
            "recording_id": rec_id,
            "style_bucket": entry.get("style_bucket", ""),
            "artist_name": entry.get("artist_name", ""),
            "segments_produced": len(track_segs),
            "tuks_matched": len(matched_tuks),
            "tuks_total": len(shabad_lines),
            "tuk_coverage": round(len(matched_tuks) / max(1, len(shabad_lines)), 3),
            "mean_match_score": round(mean(s["match_score"] for s in track_segs), 4) if track_segs else 0,
            "median_match_score": round(median(s["match_score"] for s in track_segs), 4) if track_segs else 0,
            "mean_confidence": round(mean(s["avg_confidence"] for s in track_segs), 4) if track_segs else 0,
            "repetitions_captured": sum(1 for s in track_segs if s["repetition"] > 0),
            "total_duration_sec": round(sum(s["duration"] for s in track_segs), 1),
        })

    all_scores = [s["match_score"] for s in segments]
    all_conf = [s["avg_confidence"] for s in segments]
    all_tuks_matched = sum(t["tuks_matched"] for t in tracks)
    all_tuks_total = sum(t["tuks_total"] for t in tracks)

    report = {
        "config_name": config_name,
        "model_name": model_name,
        "params": {k: v for k, v in params.items() if k != "initial_prompt"},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "tracks": tracks,
        "aggregate": {
            "total_segments": len(segments),
            "mean_match_score": round(mean(all_scores), 4) if all_scores else 0,
            "median_match_score": round(median(all_scores), 4) if all_scores else 0,
            "mean_confidence": round(mean(all_conf), 4) if all_conf else 0,
            "tuk_coverage": round(all_tuks_matched / max(1, all_tuks_total), 3),
            "total_duration_sec": round(sum(s["duration"] for s in segments), 1),
        },
    }

    os.makedirs("logs", exist_ok=True)
    report_path = f"logs/{config_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Report written: %s", report_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Whisper model comparison pod script")
    parser.add_argument("--model", required=True, help="HF model ID (e.g. Systran/faster-whisper-large-v3-turbo)")
    parser.add_argument("--configs", required=True, help="Comma-separated config names to run")
    parser.add_argument("--hf-repo", default="surindersinghssj/gurbani-asr-model-comparison",
                       help="HF dataset repo to push results")
    parser.add_argument("--catalog", default="data/manifests/test_catalog.json",
                       help="Path to test catalog JSON")
    parser.add_argument("--db", default="database.sqlite", help="Path to STTM database")
    args = parser.parse_args()

    global DB_PATH
    DB_PATH = args.db

    config_names = [c.strip() for c in args.configs.split(",")]
    model_name = args.model.split("/")[-1]  # e.g. "faster-whisper-large-v3-turbo" → display name

    log.info("=== Whisper Model Comparison ===")
    log.info("Model: %s", args.model)
    log.info("Configs: %s", config_names)

    # Load catalog
    with open(args.catalog) as f:
        catalog = json.load(f)
    log.info("Loaded %d tracks from catalog", len(catalog))

    # Download audio
    log.info("Downloading audio...")
    audio_dir = "data/audio"
    for entry in catalog:
        rec_id = entry["recording_id"]
        audio_path = os.path.join(audio_dir, f"{rec_id}.mp3")
        entry["local_path"] = audio_path
        if not download_track(entry["url"], audio_path):
            log.error("Failed to download %s", entry["url"])
            continue

    # Pre-load all audio into memory
    log.info("Loading audio files...")
    audio_data = {}
    for entry in catalog:
        path = entry.get("local_path", "")
        if os.path.exists(path):
            try:
                audio_data[entry["recording_id"]] = load_audio_mono(path)
                dur = len(audio_data[entry["recording_id"]]) / TARGET_SR
                log.info("  %s: %.1f min", entry["recording_id"][:8], dur / 60)
            except Exception as e:
                log.error("  Failed to load %s: %s", path, e)

    # Load Whisper model once
    from faster_whisper import WhisperModel
    log.info("Loading model %s...", args.model)
    t0 = time.time()
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    log.info("Model loaded in %.1fs", time.time() - t0)

    # Run each config
    for config_name in config_names:
        log.info("\n" + "=" * 60)
        log.info("CONFIG: %s", config_name)
        log.info("=" * 60)

        params = {**COMMON_PARAMS, **CONFIG_OVERRIDES[config_name]}
        all_segments = []

        for entry in catalog:
            rec_id = entry["recording_id"]
            if rec_id not in audio_data:
                continue

            entry["_model_name"] = model_name
            log.info("Processing %s (%s)...", entry.get("title", "")[:40], entry.get("style_bucket", ""))

            segments = transcribe_and_align(
                model, audio_data[rec_id], config_name, entry, rec_id,
            )
            all_segments.extend(segments)
            log.info("  Segments: %d, total so far: %d", len(segments), len(all_segments))

        # Push to HF
        push_to_hf(all_segments, config_name, args.hf_repo)

        # Write report
        report = write_report(all_segments, config_name, model_name, params, catalog)
        log.info("Config %s: %d segments, match_score=%.3f, confidence=%.3f, coverage=%.1f%%",
                config_name, report["aggregate"]["total_segments"],
                report["aggregate"]["mean_match_score"],
                report["aggregate"]["mean_confidence"],
                report["aggregate"]["tuk_coverage"] * 100)

        # Write completion marker
        with open(f"logs/{config_name}_COMPLETE", "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())

    # Final marker
    with open("logs/COMPLETE", "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    log.info("\n=== ALL CONFIGS COMPLETE ===")


if __name__ == "__main__":
    main()
