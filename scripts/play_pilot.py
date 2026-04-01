#!/usr/bin/env python3
"""Browse and play audio segments from the pilot comparison dataset.

Loads surindersinghssj/gurbani-asr-v3-pilot-comparison from HF Hub,
lets you browse segments by style bucket, and plays each clip with
the matching canonical Gurbani line.

Usage:
    python3 scripts/play_pilot.py                         # interactive browser
    python3 scripts/play_pilot.py --split faster_whisper  # specific split
    python3 scripts/play_pilot.py --random 10             # play 10 random segments
    python3 scripts/play_pilot.py --list                  # list splits and counts
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HF_REPO = "surindersinghssj/gurbani-asr-v3-pilot-comparison"
HF_TOKEN = os.environ.get("HF_TOKEN")


def decode_audio(raw: dict) -> tuple[np.ndarray, int]:
    """Decode HF audio struct {bytes, path} → (array, sample_rate).

    Works without torch — reads raw bytes via soundfile.
    """
    audio_bytes = raw.get("bytes")
    if audio_bytes:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # stereo → mono
        return arr, sr
    raise ValueError("No audio bytes in row")


def play_audio(arr: np.ndarray, sr: int):
    """Play audio array using macOS afplay (no sounddevice needed)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, arr, sr)
        subprocess.run(["afplay", tmp_path], check=True)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def format_duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def get_raw_rows(ds) -> list[dict]:
    """Extract rows from dataset without triggering torch audio decoding."""
    pa = ds._data.to_pydict()
    n = len(pa["segment_id"])
    rows = []
    for i in range(n):
        row = {k: pa[k][i] for k in pa}
        rows.append(row)
    return rows


def print_segment(row: dict, index: int, total: int):
    dur = row.get("duration", 0) or 0
    print(f"\n{'─' * 64}")
    print(f"  [{index}/{total}]  {row.get('artist_name', '')}  •  {row.get('style_bucket', '')}  •  {row.get('pipeline', '')}")
    print(f"  Segment: {row.get('segment_id', '')}  •  Duration: {format_duration(dur)}")
    print(f"  Match: {row.get('match_score', 0):.3f}  •  Confidence: {row.get('avg_confidence', 0):.3f}")
    print(f"\n  Gurbani:")
    print(f"  {row.get('training_label', '')}")
    print(f"{'─' * 64}")


def run_interactive(rows: list[dict], split_name: str):
    total = len(rows)
    print(f"\nLoaded {total} segments from split '{split_name}'")
    print("Controls: Enter=play, n=next, p=prev, q=quit")

    i = 0
    while 0 <= i < total:
        row = rows[i]
        print_segment(row, i + 1, total)

        try:
            cmd = input("  [Enter=play / n=next / p=prev / q=quit] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if cmd == "q":
            break
        elif cmd == "p":
            i = max(0, i - 1)
            continue
        elif cmd == "n":
            i += 1
            continue
        elif cmd in ("", "play"):
            audio_raw = row.get("audio")
            if audio_raw:
                try:
                    arr, sr = decode_audio(audio_raw)
                    print(f"  ▶ Playing {format_duration(len(arr)/sr)}...")
                    play_audio(arr, sr)
                except KeyboardInterrupt:
                    print("  (stopped)")
                except Exception as e:
                    print(f"  Playback error: {e}")
            else:
                print("  No audio in this row")
            i += 1
        else:
            i += 1

    print(f"\nDone.")


def run_random(rows: list[dict], n: int):
    import random
    sample = random.sample(rows, min(n, len(rows)))
    total = len(sample)
    print(f"\nPlaying {total} random segments\n")

    for i, row in enumerate(sample):
        print_segment(row, i + 1, total)
        audio_raw = row.get("audio")
        if audio_raw:
            try:
                arr, sr = decode_audio(audio_raw)
                print(f"  ▶ Playing {format_duration(len(arr)/sr)}...")
                play_audio(arr, sr)
            except KeyboardInterrupt:
                print("  (skipped)")
                try:
                    if input("  Continue? [Enter/q] > ").strip().lower() == "q":
                        break
                except (EOFError, KeyboardInterrupt):
                    break
            except Exception as e:
                print(f"  Playback error: {e}")
        else:
            print("  (no audio)")

    print(f"\nDone.")


def list_splits():
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    try:
        files = list(api.list_repo_files(HF_REPO, repo_type="dataset"))
        splits = set()
        for f in files:
            if f.endswith(".parquet"):
                parts = Path(f).parts
                if len(parts) >= 2 and parts[0] not in ("data", "."):
                    splits.add(parts[0])
        if not splits:
            # Flat layout: check parquet file names
            for f in files:
                if f.endswith(".parquet"):
                    # e.g. "faster_whisper-train-00000-of-00001.parquet"
                    name = Path(f).stem.split("-")[0]
                    splits.add(name)

        print(f"\nSplits in {HF_REPO}:")
        for s in sorted(splits):
            try:
                ds = load_dataset(HF_REPO, split=s, token=HF_TOKEN)
                pa = ds._data
                print(f"  {s}: {pa.num_rows} rows")
            except Exception:
                print(f"  {s}")
    except Exception as e:
        print(f"Could not list splits: {e}")


def main():
    parser = argparse.ArgumentParser(description="Browse and play Gurbani pilot segments")
    parser.add_argument("--split", default="faster_whisper",
                        help="HF dataset split name (default: faster_whisper)")
    parser.add_argument("--random", type=int, metavar="N",
                        help="Play N random segments")
    parser.add_argument("--list", action="store_true",
                        help="List available splits and exit")
    args = parser.parse_args()

    if args.list:
        list_splits()
        return

    print(f"Loading {HF_REPO} split='{args.split}'...")
    try:
        ds = load_dataset(HF_REPO, split=args.split, token=HF_TOKEN)
    except Exception as e:
        print(f"Error: {e}")
        print(f"\nRun with --list to see available splits.")
        sys.exit(1)

    rows = get_raw_rows(ds)

    if args.random:
        run_random(rows, args.random)
    else:
        run_interactive(rows, args.split)


if __name__ == "__main__":
    main()
