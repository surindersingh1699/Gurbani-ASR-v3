"""Surt — LoRA fine-tuning loop for all training phases.

Implements the 5-phase curriculum training plan from docs/TRAINING.md:
  Phase 1: Text LM priming (decoder-only, perplexity metric)
  Phase 2: Clean audio — sehaj path (Seq2Seq, WER metric)
  Phase 3: Studio kirtan — hazoori + puratan (Seq2Seq + augmentation)
  Phase 4: Live kirtan — AKJ, taksali, mixed (Seq2Seq + heavy augmentation)
  Phase 5: Quantisation only (no training — see quantise.py)

Usage:
    python src/model/train.py --phase 1
    python src/model/train.py --phase 2 --resume
    python src/model/train.py --phase 1 --dry-run   # show config without training
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# Maximum OOM retries — halve batch each time
MAX_OOM_RETRIES = 2


# ─── Model + LoRA setup ────────────────────────────────────────────────────

def load_model_and_tokenizer(phase: int, config, resume: bool = False):
    """Load whisper-small with LoRA adapter for the given phase.

    Phase 1: loads base model from HF Hub.
    Phase 2+: loads previous phase checkpoint from HF Hub.
    If resume=True: loads current phase checkpoint (for continuing interrupted runs).
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(config.BASE_MODEL)

    if resume:
        # Resume from current phase checkpoint
        hub_path = f"{config.HF_MODEL_REPO}-phase{phase}"
        log.info("Resuming from %s", hub_path)
        model = WhisperForConditionalGeneration.from_pretrained(hub_path)
    elif phase > 1:
        # Load previous phase checkpoint
        prev_hub = f"{config.HF_MODEL_REPO}-phase{phase - 1}"
        log.info("Loading phase %d checkpoint: %s", phase - 1, prev_hub)
        model = WhisperForConditionalGeneration.from_pretrained(prev_hub)
    else:
        # Phase 1: start from base
        log.info("Loading base model: %s", config.BASE_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(config.BASE_MODEL)

    # Apply LoRA
    targets = config.lora_targets_for_phase(phase)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Phase %d — LoRA targets: %s", phase, targets)
    log.info("Trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}",
             100 * trainable / total)

    return model, processor


# ─── Dataset loading ────────────────────────────────────────────────────────

def load_dataset_for_phase(phase: int, processor, config):
    """Load and prepare dataset for the given phase.

    Phase 1: text-only from STTM corpus (HF_TEXT_CORPUS_REPO).
    Phase 2-4: audio+text from HF Hub (HF_DATASET_REPO), filtered by phase.
    Returns (train_dataset, eval_dataset).
    """
    if phase == 1:
        return _load_phase1_text(processor, config)
    else:
        return _load_audio_phase(phase, processor, config)


def _load_phase1_text(processor, config):
    """Phase 1: text-only dataset for decoder LM priming."""
    from datasets import load_dataset

    repo = getattr(config, "HF_TEXT_CORPUS_REPO", config.HF_DATASET_REPO)
    ds = load_dataset(repo, split="train", streaming=False)

    # Filter to SGGS only (source_id=1) if the column exists
    if "source_id" in ds.column_names:
        ds = ds.filter(lambda x: x["source_id"] == 1)

    # Pilot subset
    subset_size = getattr(config, "PILOT_SUBSET_SIZE", None)
    if subset_size:
        ds = ds.select(range(min(subset_size, len(ds))))

    ds = ds.shuffle(seed=config.SHUFFLE_SEED)

    # 85/15 train/eval split
    split = ds.train_test_split(test_size=0.15, seed=config.SHUFFLE_SEED)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = processor.tokenizer
    # Detect text column name
    text_col = "sentence" if "sentence" in train_ds.column_names else "transcription"

    def tokenize(batch):
        labels = tokenizer(
            batch[text_col],
            padding="max_length",
            max_length=448,
            truncation=True,
        )
        batch["labels"] = labels["input_ids"]
        return batch

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    log.info("Phase 1: %d train, %d eval text samples (from %s)", len(train_ds), len(eval_ds), repo)
    return train_ds, eval_ds


def _load_audio_phase(phase: int, processor, config):
    """Phases 2-4: audio+text dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        config.HF_DATASET_REPO,
        split="train",
        streaming=False,
    )

    # Filter by phase only if the column exists
    if "phase" in ds.column_names:
        ds = ds.filter(lambda x: x.get("phase") == phase)

    # Pilot subset
    subset_size = getattr(config, "PILOT_SUBSET_SIZE", None)
    if subset_size:
        ds = ds.select(range(min(subset_size, len(ds))))

    ds = ds.shuffle(seed=config.SHUFFLE_SEED)

    split = ds.train_test_split(test_size=0.15, seed=config.SHUFFLE_SEED)
    train_ds = split["train"]
    eval_ds = split["test"]

    log.info("Phase %d: %d train, %d eval audio samples", phase, len(train_ds), len(eval_ds))
    return train_ds, eval_ds


# ─── Data collator ──────────────────────────────────────────────────────────

class DataCollatorSpeechSeq2Seq:
    """Collates audio+text batches for Whisper Seq2Seq training.

    Handles:
    - Audio → mel spectrogram via WhisperFeatureExtractor
    - Text → token IDs via WhisperTokenizer
    - Padding for both inputs and labels
    - Optional online augmentation (Phases 3-4)
    """

    def __init__(self, processor, phase: int = 2):
        self.processor = processor
        self.phase = phase
        self.augment = self._build_augmentation(phase)

    def _build_augmentation(self, phase: int):
        """Build online augmentation pipeline for Phases 3-4."""
        if phase < 3:
            return None
        try:
            from audiomentations import Compose, PitchShift
            augmentations = [PitchShift(min_semitones=-2, max_semitones=2, p=0.4)]

            if phase >= 4:
                from audiomentations import AddBackgroundNoise, RoomSimulator
                # Only add noise/reverb if sample files exist
                harmonium_path = Path("data/augmentation/harmonium_samples")
                if harmonium_path.exists():
                    augmentations.insert(0, AddBackgroundNoise(
                        sounds_path=str(harmonium_path),
                        min_snr_db=5, max_snr_db=20, p=0.6,
                    ))
                augmentations.append(RoomSimulator(
                    min_size_x=8, max_size_x=40,
                    min_size_y=8, max_size_y=40,
                    min_size_z=3, max_size_z=12,
                    p=0.5,
                ))

            return Compose(augmentations)
        except ImportError:
            log.warning("audiomentations not installed — skipping augmentation")
            return None

    def __call__(self, features):
        import numpy as np

        audio_arrays = []
        labels = []

        for f in features:
            audio = f["audio"]["array"]
            sr = f["audio"]["sampling_rate"]

            # Resample to 16kHz if needed
            if sr != 16000:
                import torchaudio
                waveform = torch.tensor(audio).unsqueeze(0)
                audio = torchaudio.functional.resample(waveform, sr, 16000).squeeze(0).numpy()

            # Apply augmentation
            if self.augment is not None:
                audio = self.augment(samples=audio.astype(np.float32), sample_rate=16000)

            audio_arrays.append(audio)
            labels.append(f.get("transcription") or f.get("sentence") or "")

        # Extract mel features
        inputs = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt",
        )

        # Tokenize labels
        label_features = self.processor.tokenizer(
            labels, padding=True, return_tensors="pt", truncation=True, max_length=448,
        )

        # Replace padding token id with -100 for loss masking
        label_ids = label_features["input_ids"]
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        batch = {
            "input_features": inputs.input_features,
            "labels": label_ids,
        }
        return batch


class DataCollatorTextOnly:
    """Collates text-only batches for Phase 1 decoder LM priming.

    Creates dummy mel spectrograms (zeros) since the encoder still runs
    but we only care about decoder language modelling.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Stack pre-tokenized labels
        labels = torch.tensor([f["labels"] for f in features])

        # Dummy mel input (zeros) — encoder will produce embeddings but
        # the decoder's self-attention learning is what matters
        batch_size = len(features)
        dummy_mel = torch.zeros(batch_size, 80, 3000)

        # Mask padding in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": dummy_mel,
            "labels": labels,
        }


# ─── Metrics ────────────────────────────────────────────────────────────────

def build_compute_metrics(processor, phase: int):
    """Return a compute_metrics function for the Trainer."""
    import numpy as np

    if phase == 1:
        # Phase 1: perplexity (computed from loss by Trainer automatically)
        return None

    def compute_metrics(pred):
        import jiwer

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer * 100}  # percentage

    return compute_metrics


# ─── Trainer builder ────────────────────────────────────────────────────────

def build_trainer(model, processor, train_ds, eval_ds, phase: int, config,
                  output_dir: str, checkpoint_path: str = None):
    """Build a Seq2SeqTrainer with all config merged."""
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    phase_config = config.TRAINING_CONFIG[phase]
    if phase_config is None:
        raise ValueError(f"Phase {phase} has no training config (quantisation only)")

    # Merge phase-specific + common training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=phase_config["batch_size"],
        learning_rate=phase_config["learning_rate"],
        warmup_steps=phase_config["warmup_steps"],
        max_steps=phase_config["max_steps"],
        eval_steps=phase_config["eval_steps"],
        save_steps=phase_config["save_steps"],
        predict_with_generate=(phase > 1),
        generation_max_length=448,
        hub_model_id=f"{config.HF_MODEL_REPO}-phase{phase}",
        hub_strategy="every_save",
        metric_for_best_model="loss" if phase == 1 else "wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        **config.COMMON_TRAINING_ARGS,
    )

    # Select collator
    if phase == 1:
        collator = DataCollatorTextOnly(processor)
    else:
        collator = DataCollatorSpeechSeq2Seq(processor, phase=phase)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor, phase),
        processing_class=processor.feature_extractor,
    )

    return trainer


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Surt LoRA fine-tuning")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Training phase (1-4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from current phase checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--config", type=str, default="config",
                        help="Config module name (default: config, use config_pilot for pilot)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    config = importlib.import_module(args.config)

    phase = args.phase
    output_dir = args.output_dir or f"checkpoints/phase{phase}"

    log.info("=" * 60)
    log.info("SURT TRAINING — Phase %d", phase)
    log.info("Base model: %s", config.BASE_MODEL)
    log.info("LoRA rank: %d, alpha: %d", config.LORA_RANK, config.LORA_ALPHA)
    log.info("LoRA targets: %s", config.lora_targets_for_phase(phase))
    log.info("Training config: %s", config.TRAINING_CONFIG[phase])
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    if args.dry_run:
        log.info("DRY RUN — exiting without training")
        return

    # Load model
    model, processor = load_model_and_tokenizer(phase, config, resume=args.resume)

    # Load dataset
    train_ds, eval_ds = load_dataset_for_phase(phase, processor, config)

    # Build trainer
    checkpoint_path = output_dir if args.resume and Path(output_dir).exists() else None
    trainer = build_trainer(model, processor, train_ds, eval_ds, phase, config,
                            output_dir, checkpoint_path)

    # Train with OOM auto-recovery
    for attempt in range(MAX_OOM_RETRIES + 1):
        try:
            trainer.train(resume_from_checkpoint=checkpoint_path)
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and attempt < MAX_OOM_RETRIES:
                torch.cuda.empty_cache()
                old_bs = trainer.args.per_device_train_batch_size
                new_bs = max(1, old_bs // 2)
                new_ga = trainer.args.gradient_accumulation_steps * 2
                log.warning("OOM! Reducing batch %d→%d, accumulation %d→%d (retry %d/%d)",
                           old_bs, new_bs,
                           trainer.args.gradient_accumulation_steps, new_ga,
                           attempt + 1, MAX_OOM_RETRIES)
                trainer.args.per_device_train_batch_size = new_bs
                trainer.args.gradient_accumulation_steps = new_ga
                # Rebuild trainer with new args
                trainer = build_trainer(model, processor, train_ds, eval_ds, phase, config,
                                       output_dir, checkpoint_path)
            else:
                raise

    # Push final checkpoint to HF Hub
    log.info("Pushing final checkpoint to HF Hub...")
    trainer.push_to_hub(commit_message=f"Phase {phase} training complete")

    # Log final metrics
    metrics = trainer.evaluate()
    log.info("=" * 60)
    log.info("PHASE %d COMPLETE", phase)
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)

    wer_target = config.WER_TARGETS.get(phase)
    if wer_target and "eval_wer" in metrics:
        wer = metrics["eval_wer"]
        status = "PASS" if wer < wer_target else "FAIL"
        log.info("  WER: %.1f%% (target: <%.1f%%) — %s", wer, wer_target, status)

    log.info("=" * 60)


if __name__ == "__main__":
    main()
