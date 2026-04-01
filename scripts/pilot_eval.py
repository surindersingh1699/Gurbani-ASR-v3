"""Quick qualitative eval after pilot training.

Loads the pilot Phase 2 checkpoint, transcribes a few samples from the
sehaj path dataset, and prints predicted vs canonical text side by side.

Usage:
    python scripts/pilot_eval.py
    python scripts/pilot_eval.py --checkpoint checkpoints/phase2 --n 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pilot qualitative eval")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--dataset", type=str, default="surindersinghssj/gurbani-asr",
                        help="HF dataset repo for eval samples")
    args = parser.parse_args()

    from datasets import load_dataset
    from faster_whisper import WhisperModel

    log.info("Loading model from %s ...", args.checkpoint)
    model = WhisperModel(args.checkpoint, device="auto", compute_type="float16")

    log.info("Loading %d samples from %s ...", args.n, args.dataset)
    ds = load_dataset(args.dataset, split="train").shuffle(seed=99).select(range(args.n))

    correct = 0
    for i, row in enumerate(ds):
        audio = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]
        label = row.get("transcription") or row.get("sentence") or ""

        segments, info = model.transcribe(audio, language="pa", beam_size=2)
        pred = " ".join(s.text for s in segments).strip()

        match = "==" if pred == label else "!="
        if pred == label:
            correct += 1

        print(f"\n--- Sample {i+1}/{args.n} ---")
        print(f"LABEL: {label}")
        print(f"PRED:  {pred}")
        print(f"MATCH: {match}")

    print(f"\n--- Exact match: {correct}/{args.n} ---")


if __name__ == "__main__":
    main()
