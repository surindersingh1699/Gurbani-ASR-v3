"""Add audio slices to the gurbani-asr-model-comparison HF dataset.

For each split (config), downloads the source audio, slices by start/end
timestamps, and pushes a new version of the dataset with an Audio column.

Usage:
    python scripts/add_audio_to_comparison.py
    python scripts/add_audio_to_comparison.py --split large_v3   # single split
    python scripts/add_audio_to_comparison.py --dry-run           # no HF push
"""

import argparse
import io
import os
import subprocess
from pathlib import Path

import requests
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_REPO = "surindersinghssj/gurbani-asr-model-comparison"
PILOT_REPO = "surindersinghssj/gurbani-asr-v3-pilot-comparison"
TARGET_SR = 16000
ALL_SPLITS = [
    "large_v2", "large_v3",
    "v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
    "v3_turbo_beam5", "v3_turbo_nocond",
    "openai_large_v2", "openai_large_v3",
]

# ─── Audio helpers ────────────────────────────────────────────────────────────

def download_audio(url: str, cache_dir: Path) -> Path:
    """Download source audio to cache_dir, return path to 16kHz mono WAV."""
    import hashlib
    key = hashlib.md5(url.encode()).hexdigest()[:12]
    wav_path = cache_dir / f"{key}.wav"
    if wav_path.exists():
        print(f"  [cache] {url[:60]}")
        return wav_path

    print(f"  [download] {url[:60]}")
    resp = requests.get(url, allow_redirects=True, timeout=300,
                        headers={"User-Agent": "GurbaniASR/1.0"})
    resp.raise_for_status()

    mp3_path = cache_dir / f"{key}.mp3"
    mp3_path.write_bytes(resp.content)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(TARGET_SR),
         "-ac", "1", str(wav_path)],
        capture_output=True, check=True,
    )
    mp3_path.unlink()
    print(f"  [converted] {wav_path.name}")
    return wav_path


def slice_audio(wav_path: Path, start: float, end: float) -> dict:
    """Slice [start, end] seconds from wav_path. Returns HF Audio dict."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == TARGET_SR, f"Expected {TARGET_SR}Hz, got {sr}Hz"

    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    chunk = audio[s:e]

    # Encode to flac bytes (lossless, smaller than wav)
    buf = io.BytesIO()
    sf.write(buf, chunk, TARGET_SR, format="FLAC")
    return {"bytes": buf.getvalue(), "path": None}


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_split(split: str, cache_dir: Path, hf_token: str,
                hf_repo: str = HF_REPO) -> Dataset:
    """Load a split, add audio column, return updated Dataset."""
    print(f"\n{'='*50}")
    print(f"Split: {split}  (repo: {hf_repo})")

    try:
        ds = load_dataset(hf_repo, split=split, token=hf_token)
    except ValueError:
        # Split exists as parquet but not registered in dataset_info —
        # load directly from the parquet file.
        parquet_url = (
            f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
            f"/data/{split}-00000-of-00001.parquet"
        )
        print(f"  [parquet fallback] {parquet_url}")
        ds = load_dataset("parquet", data_files=parquet_url, split="train",
                          token=hf_token)
    print(f"  Loaded {len(ds)} rows")

    # Download all unique source URLs
    unique_urls = list(set(ds["source_url"]))
    print(f"  Downloading {len(unique_urls)} source audio files...")
    wav_cache: dict[str, Path] = {}
    for url in unique_urls:
        try:
            wav_cache[url] = download_audio(url, cache_dir)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}")

    # Slice each row
    print(f"  Slicing {len(ds)} segments...")
    audio_col = []
    skipped = 0
    for row in ds:
        url = row["source_url"]
        if url not in wav_cache:
            audio_col.append({"bytes": b"", "path": None})
            skipped += 1
            continue
        try:
            audio_dict = slice_audio(wav_cache[url], row["start"], row["end"])
            audio_col.append(audio_dict)
        except Exception as e:
            print(f"  WARN slice error seg {row['segment_id']}: {e}")
            audio_col.append({"bytes": b"", "path": None})
            skipped += 1

    if skipped:
        print(f"  WARNING: {skipped} segments have empty audio")

    # Build new dataset with audio column.
    # Store FLAC bytes as plain binary first, then cast_column to Audio to
    # avoid encode_example which requires torch/torchcodec (datasets v4.7+).
    data = ds.to_dict()
    data["audio"] = [item["bytes"] for item in audio_col]

    features = Features({
        "config_name": Value("string"),
        "model_name": Value("string"),
        "recording_id": Value("string"),
        "segment_id": Value("string"),
        "tuk_index": Value("int32"),
        "canonical_line": Value("string"),
        "whisper_text": Value("string"),
        "start": Value("float32"),
        "end": Value("float32"),
        "duration": Value("float32"),
        "match_score": Value("float32"),
        "avg_confidence": Value("float32"),
        "repetition": Value("int32"),
        "ang": Value("int32"),
        "raag": Value("string"),
        "writer": Value("string"),
        "style_bucket": Value("string"),
        "artist_name": Value("string"),
        "source_url": Value("string"),
        "audio": Value("binary"),
    })

    new_ds = Dataset.from_dict(data, features=features)
    # cast_column updates Arrow metadata only — no re-encoding, no torch needed.
    # decode=False prevents torchcodec from being invoked when accessing rows
    # locally (datasets v4.7+). HF viewer decodes FLAC server-side regardless.
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=TARGET_SR, decode=False))
    print(f"  Built dataset: {len(new_ds)} rows with audio")

    # Spot-check: decode=False so row access returns {"bytes": ..., "path": ...}
    row = new_ds[0]
    audio_bytes = len(row["audio"]["bytes"]) if row["audio"]["bytes"] else 0
    print(f"  Spot-check row 0: {row['canonical_line'][:40]} | "
          f"dur={row['duration']:.2f}s | audio_bytes={audio_bytes}")

    return new_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="", help="Single split to process")
    parser.add_argument("--repo", default="",
                        help=f"HF dataset repo to read/write "
                             f"(default: {HF_REPO}, "
                             f"pilot: {PILOT_REPO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download and slice but don't push to HF")
    args = parser.parse_args()

    hf_repo = args.repo or HF_REPO

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        raise ValueError("HF_TOKEN not set — run with --dry-run or set HF_TOKEN")

    cache_dir = Path("data/audio_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Audio cache: {cache_dir.resolve()}")
    print(f"Repo: {hf_repo}")

    splits_to_process = [args.split] if args.split else ALL_SPLITS

    # Build all splits first (so we can push as a consistent DatasetDict)
    built: dict[str, Dataset] = {}
    for split in splits_to_process:
        built[split] = build_split(split, cache_dir, hf_token, hf_repo=hf_repo)

    if args.dry_run:
        print(f"\n[dry-run] All {len(built)} splits built successfully. Skipping push.")
        print(f"Done. Dataset: https://huggingface.co/datasets/{hf_repo}")
        return

    # When processing all splits, use DatasetDict.push_to_hub so HF sees a
    # consistent schema across all splits in a single operation — avoids the
    # "features don't match existing splits" error from partial pushes.
    if not args.split:
        print(f"\nPushing all {len(built)} splits as DatasetDict...")
        dd = DatasetDict(built)
        dd.push_to_hub(hf_repo, token=hf_token)
        print(f"  All splits pushed.")
    else:
        # Single-split mode: push individually (schema already consistent if
        # the other splits on HF already have the audio column)
        split = args.split
        print(f"\nPushing split={split}...")
        built[split].push_to_hub(hf_repo, split=split, token=hf_token)
        print(f"  {split} pushed.")

    print(f"\nDone. Dataset: https://huggingface.co/datasets/{hf_repo}")


if __name__ == "__main__":
    main()
