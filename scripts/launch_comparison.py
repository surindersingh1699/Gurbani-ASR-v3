#!/usr/bin/env python3
"""Launch Whisper model comparison on 3 parallel RunPod GPU pods.

Self-contained approach: uploads tarball to HF Hub, pods auto-download and run.
No SSH needed. Monitors completion via HF Hub dataset splits.

Usage:
    python3 scripts/launch_comparison.py                    # Full pipeline
    python3 scripts/launch_comparison.py --upload-only      # Upload tarball only
    python3 scripts/launch_comparison.py --monitor-only     # Monitor existing pods only
    python3 scripts/launch_comparison.py --keep-pods        # Don't terminate pods after
"""

import argparse
import base64
import json
import logging
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d  %(asctime)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
HF_REPO = "surindersinghssj/gurbani-asr-model-comparison"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RUNPOD_API = "https://api.runpod.io/graphql"
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_DIR / "data" / "manifests" / "test_catalog.json"
ANALYSIS_DIR = PROJECT_DIR / "data" / "analysis"
TEMPLATE_IDS_FILE = PROJECT_DIR / "data" / "manifests" / "runpod_template_ids.json"
POD_STATE_FILE = PROJECT_DIR / "data" / "manifests" / "pod_state.json"

# GPU preference order
GPU_TYPES = [
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA L4",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 4060 Ti",
    "NVIDIA RTX 4000 SFF Ada Generation",
]

MODELS = [
    {"key": "large-v2", "repo": "Systran/faster-whisper-large-v2",
     "configs": ["large_v2"]},
    {"key": "large-v3", "repo": "Systran/faster-whisper-large-v3",
     "configs": ["large_v3"]},
    {"key": "large-v3-turbo", "repo": "Systran/faster-whisper-large-v3-turbo",
     "configs": ["v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
                  "v3_turbo_beam5", "v3_turbo_nocond"]},
]

ALL_CONFIGS = ["large_v2", "large_v3", "v3_turbo_base",
               "v3_turbo_rep", "v3_turbo_temp0", "v3_turbo_beam5", "v3_turbo_nocond"]

# ─── Bootstrap script (base64-encoded to avoid escaping issues) ──────────────

BOOTSTRAP_SCRIPT = r"""#!/bin/bash
set -e
echo "=== [$(date)] Installing dependencies ==="
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
pip install -q "faster-whisper>=1.1.0" scipy numpy soundfile requests datasets huggingface-hub tqdm 2>/dev/null

echo "=== [$(date)] Downloading tarball from HF Hub ==="
python3 << 'PYEOF'
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download(
    repo_id=os.environ["HF_REPO"],
    filename="surt_comparison.tar.gz",
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN", None),
)
shutil.copy(p, "/workspace/surt_comparison.tar.gz")
print(f"Downloaded tarball to /workspace/surt_comparison.tar.gz")
PYEOF

echo "=== [$(date)] Extracting tarball ==="
cd /workspace && tar xzf surt_comparison.tar.gz
ls -la /workspace/scripts/ /workspace/database.sqlite /workspace/data/manifests/

echo "=== [$(date)] Starting comparison (MODEL=$MODEL, CONFIGS=$CONFIGS) ==="
cd /workspace
nohup python3 scripts/whisper_model_comparison.py \
    --model "$MODEL" \
    --configs "$CONFIGS" \
    --hf-repo "$HF_REPO" \
    > /workspace/run.log 2>&1 &

echo "=== [$(date)] Bootstrap complete — comparison PID=$! running in background ==="
"""


# ─── GraphQL helper ──────────────────────────────────────────────────────────

def _graphql(query: str, variables: dict = None) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(RUNPOD_API, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        log.error("GraphQL errors: %s", data["errors"])
    return data


# ─── Template management ────────────────────────────────────────────────────

def get_existing_templates() -> dict:
    result = _graphql("query { myself { podTemplates { id name } } }")
    templates = result.get("data", {}).get("myself", {}).get("podTemplates", [])
    return {t["name"]: t["id"] for t in templates}


def delete_template(template_id: str):
    return _graphql(
        """mutation ($id: String!) { deleteTemplate(templateId: $id) }""",
        {"id": template_id},
    )


def create_auto_run_template() -> str:
    """Create a single auto-run template. Pods pass MODEL, CONFIGS, HF_REPO, HF_TOKEN via env."""
    existing = get_existing_templates()

    # Clean up old templates
    for old_name in ["surt-cmp-v2", "surt-cmp-v3", "surt-cmp-turbo", "surt-cmp-auto"]:
        if old_name in existing:
            log.info("Deleting old template: %s (%s)", old_name, existing[old_name])
            try:
                delete_template(existing[old_name])
            except Exception as e:
                log.warning("  Failed: %s", e)

    # Encode bootstrap script as base64 to avoid shell/GraphQL escaping issues
    b64 = base64.b64encode(BOOTSTRAP_SCRIPT.encode()).decode()
    startup = f"echo {b64} | base64 -d > /tmp/bootstrap.sh && bash /tmp/bootstrap.sh"

    result = _graphql(
        """mutation SaveTemplate($input: SaveTemplateInput!) {
            saveTemplate(input: $input) { id name }
        }""",
        {"input": {
            "name": "surt-cmp-auto",
            "imageName": DOCKER_IMAGE,
            "containerDiskInGb": 30,
            "volumeInGb": 0,
            "dockerArgs": "",
            "env": [],
            "startScript": startup,
            "isPublic": False,
        }},
    )

    saved = result.get("data", {}).get("saveTemplate", {})
    template_id = saved.get("id", "")
    log.info("Created auto-run template: %s (%s)", saved.get("name"), template_id)

    # Save template ID
    os.makedirs(TEMPLATE_IDS_FILE.parent, exist_ok=True)
    with open(TEMPLATE_IDS_FILE, "w") as f:
        json.dump({"surt-cmp-auto": template_id}, f, indent=2)

    return template_id


# ─── Tarball creation + HF upload ────────────────────────────────────────────

def create_tarball() -> str:
    tarball_path = str(PROJECT_DIR / "surt_comparison.tar.gz")
    log.info("Creating tarball: %s", tarball_path)

    include = [
        "scripts/whisper_model_comparison.py",
        "data/manifests/test_catalog.json",
        "database.sqlite",
    ]

    with tarfile.open(tarball_path, "w:gz") as tar:
        for rel_path in include:
            full_path = PROJECT_DIR / rel_path
            if full_path.exists():
                tar.add(str(full_path), arcname=rel_path)
                size_mb = full_path.stat().st_size / 1e6
                log.info("  Added %s (%.1f MB)", rel_path, size_mb)
            else:
                log.warning("  Missing: %s", rel_path)

    total_mb = os.path.getsize(tarball_path) / 1e6
    log.info("Tarball created: %.1f MB", total_mb)
    return tarball_path


def upload_tarball_to_hf(tarball_path: str):
    """Upload tarball to HF Hub dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Create repo if needed
    try:
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
        log.info("HF repo ready: %s", HF_REPO)
    except Exception as e:
        log.info("HF repo: %s", e)

    # Upload tarball
    log.info("Uploading tarball to HF Hub (%s)...", HF_REPO)
    api.upload_file(
        path_or_fileobj=tarball_path,
        path_in_repo="surt_comparison.tar.gz",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    size_mb = os.path.getsize(tarball_path) / 1e6
    log.info("Uploaded %.1f MB to %s", size_mb, HF_REPO)


# ─── Pod management ─────────────────────────────────────────────────────────

def find_gpu_type() -> str:
    try:
        gpus = runpod.get_gpus()
        available = {g["id"] for g in gpus if g.get("communityCloud") or g.get("secureCloud")}
        for gpu in GPU_TYPES:
            if gpu in available:
                return gpu
    except Exception:
        pass
    return GPU_TYPES[0]


def create_pod_for_model(model_info: dict, template_id: str) -> dict:
    gpu_type = find_gpu_type()
    configs_str = ",".join(model_info["configs"])
    log.info("Creating pod for %s (configs=%s, gpu=%s)", model_info["key"], configs_str, gpu_type)

    env = {
        "HF_TOKEN": HF_TOKEN,
        "HF_REPO": HF_REPO,
        "MODEL": model_info["repo"],
        "CONFIGS": configs_str,
    }

    pod = runpod.create_pod(
        name=f"surt-cmp-{model_info['key']}",
        image_name=DOCKER_IMAGE,
        gpu_type_id=gpu_type,
        template_id=template_id,
        cloud_type="ALL",
        container_disk_in_gb=30,
        volume_in_gb=0,
        start_ssh=True,
        env=env,
    )

    log.info("Pod created: %s (id=%s)", model_info["key"], pod.get("id"))
    return pod


def wait_for_pod_ready(pod_id: str, timeout: int = 600) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "")
        runtime = pod.get("runtime")
        if status == "RUNNING" and runtime:
            log.info("Pod %s RUNNING (uptime=%ds)",
                     pod_id[:8], runtime.get("uptimeInSeconds", 0))
            return pod
        log.info("Pod %s status=%s, waiting...", pod_id[:8], status)
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not start within {timeout}s")


# ─── HF Hub monitoring ──────────────────────────────────────────────────────

def get_hf_splits() -> list[str]:
    """Check which splits exist in the HF dataset."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.dataset_info(HF_REPO)
        # Dataset card config has splits info
        # Check by listing parquet files
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
        splits = set()
        for f in files:
            # HF datasets store as data/<split>-00000-of-00001-*.parquet
            # or <split>/train-*.parquet etc.
            for config in ALL_CONFIGS:
                if config in f and f.endswith(".parquet"):
                    splits.add(config)
        return sorted(splits)
    except Exception:
        return []


def monitor_completion(pod_ids: dict, poll_interval: int = 60, timeout: int = 5400) -> list[str]:
    """Poll HF Hub for completed splits. Returns list of completed config names."""
    log.info("Monitoring HF Hub for completed splits (poll every %ds, timeout %ds)...",
             poll_interval, timeout)
    start = time.time()
    completed = set()

    while time.time() - start < timeout:
        splits = get_hf_splits()
        new = set(splits) - completed
        if new:
            for s in sorted(new):
                log.info("  NEW split detected: %s", s)
            completed = set(splits)

        # Check pod health
        for key, pid in pod_ids.items():
            try:
                pod = runpod.get_pod(pid)
                status = pod.get("desiredStatus", "?")
                uptime = pod.get("runtime", {}).get("uptimeInSeconds", 0)
                if status != "RUNNING":
                    log.warning("  Pod %s (%s) status=%s — may have crashed!", key, pid[:8], status)
            except Exception:
                pass

        print_progress(completed, len(ALL_CONFIGS))

        if len(completed) >= len(ALL_CONFIGS):
            log.info("All %d splits complete!", len(ALL_CONFIGS))
            return sorted(completed)

        time.sleep(poll_interval)

    log.warning("Timeout after %ds. Completed %d/%d splits: %s",
                timeout, len(completed), len(ALL_CONFIGS), sorted(completed))
    return sorted(completed)


# ─── Progress display ────────────────────────────────────────────────────────

def print_progress(completed: set, total: int):
    print(f"\n=== Progress: {len(completed)}/{total} configs complete ===")
    for config in ALL_CONFIGS:
        status = "DONE" if config in completed else "..."
        print(f"  {config:<20s}  {status}")
    print()


# ─── Analysis generation ────────────────────────────────────────────────────

def download_reports_from_hf() -> dict:
    """Download dataset from HF Hub and build report-like structures."""
    from datasets import load_dataset

    reports = {}
    for config_name in ALL_CONFIGS:
        try:
            ds = load_dataset(HF_REPO, split=config_name, token=HF_TOKEN)
            if len(ds) == 0:
                continue

            # Build per-track stats
            tracks_by_rec = {}
            for row in ds:
                rec_id = row["recording_id"]
                if rec_id not in tracks_by_rec:
                    tracks_by_rec[rec_id] = {
                        "recording_id": rec_id,
                        "style_bucket": row.get("style_bucket", ""),
                        "artist_name": row.get("artist_name", ""),
                        "segments": [],
                    }
                tracks_by_rec[rec_id]["segments"].append(row)

            tracks = []
            for rec_id, info in tracks_by_rec.items():
                segs = info["segments"]
                matched_tuks = set(
                    s["tuk_index"] for s in segs
                    if s.get("repetition", 0) == 0
                )
                # Count total tuks from canonical_shabad
                # Estimate: split canonical_shabad by ॥ (double danda)
                canonical = segs[0].get("canonical_shabad", "")
                tuk_count = max(1, canonical.count("॥"))

                tracks.append({
                    "recording_id": rec_id,
                    "style_bucket": info["style_bucket"],
                    "artist_name": info["artist_name"],
                    "segments_produced": len(segs),
                    "tuks_matched": len(matched_tuks),
                    "tuks_total": tuk_count,
                    "tuk_coverage": round(len(matched_tuks) / max(1, tuk_count), 3),
                    "mean_match_score": round(mean(s["match_score"] for s in segs), 4),
                    "median_match_score": round(median(s["match_score"] for s in segs), 4),
                    "mean_confidence": round(mean(s["avg_confidence"] for s in segs), 4),
                    "repetitions_captured": sum(1 for s in segs if s.get("repetition", 0) > 0),
                    "total_duration_sec": round(sum(s["duration"] for s in segs), 1),
                })

            all_scores = [s["match_score"] for row in tracks_by_rec.values() for s in row["segments"]]
            all_conf = [s["avg_confidence"] for row in tracks_by_rec.values() for s in row["segments"]]
            all_tuks_matched = sum(t["tuks_matched"] for t in tracks)
            all_tuks_total = sum(t["tuks_total"] for t in tracks)

            model_name = ds[0].get("model_name", config_name) if len(ds) > 0 else config_name

            reports[config_name] = {
                "config_name": config_name,
                "model_name": model_name,
                "tracks": tracks,
                "aggregate": {
                    "total_segments": len(ds),
                    "mean_match_score": round(mean(all_scores), 4) if all_scores else 0,
                    "median_match_score": round(median(all_scores), 4) if all_scores else 0,
                    "mean_confidence": round(mean(all_conf), 4) if all_conf else 0,
                    "tuk_coverage": round(all_tuks_matched / max(1, all_tuks_total), 3),
                    "total_duration_sec": round(sum(s["duration"]
                        for row in tracks_by_rec.values() for s in row["segments"]), 1),
                },
            }
            log.info("Downloaded split %s: %d segments, match=%.3f",
                     config_name, len(ds), reports[config_name]["aggregate"]["mean_match_score"])

        except Exception as e:
            log.warning("Failed to download split %s: %s", config_name, e)

    return reports


def generate_summary(reports: dict) -> dict:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_configs": len(reports),
        "configs": {},
        "rankings": {},
        "recommendations": [],
    }

    for name, report in sorted(reports.items()):
        agg = report["aggregate"]
        summary["configs"][name] = {
            "model": report["model_name"],
            "total_segments": agg["total_segments"],
            "mean_match_score": agg["mean_match_score"],
            "median_match_score": agg["median_match_score"],
            "mean_confidence": agg["mean_confidence"],
            "tuk_coverage": agg["tuk_coverage"],
            "total_duration_sec": agg["total_duration_sec"],
        }

    by_match = sorted(reports.items(), key=lambda x: x[1]["aggregate"]["mean_match_score"], reverse=True)
    by_conf = sorted(reports.items(), key=lambda x: x[1]["aggregate"]["mean_confidence"], reverse=True)
    by_coverage = sorted(reports.items(), key=lambda x: x[1]["aggregate"]["tuk_coverage"], reverse=True)
    by_segments = sorted(reports.items(), key=lambda x: x[1]["aggregate"]["total_segments"], reverse=True)

    summary["rankings"] = {
        "by_match_score": [n for n, _ in by_match],
        "by_confidence": [n for n, _ in by_conf],
        "by_tuk_coverage": [n for n, _ in by_coverage],
        "by_segments": [n for n, _ in by_segments],
    }

    model_configs = ["large_v2", "large_v3", "v3_turbo_base"]
    model_reports = {k: v for k, v in reports.items() if k in model_configs}
    if model_reports:
        best = max(model_reports.items(), key=lambda x: x[1]["aggregate"]["mean_match_score"])
        summary["group_a_best"] = best[0]

    turbo_configs = ["v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
                     "v3_turbo_beam5", "v3_turbo_nocond"]
    turbo_reports = {k: v for k, v in reports.items() if k in turbo_configs}
    if turbo_reports:
        best = max(turbo_reports.items(), key=lambda x: x[1]["aggregate"]["mean_match_score"])
        summary["group_b_best"] = best[0]
        base_score = turbo_reports.get("v3_turbo_base", {}).get("aggregate", {}).get("mean_match_score", 0)
        for name, report in turbo_reports.items():
            if name == "v3_turbo_base":
                continue
            delta = report["aggregate"]["mean_match_score"] - base_score
            if abs(delta) > 0.03:
                direction = "improves" if delta > 0 else "degrades"
                summary["recommendations"].append(
                    f"{name} {direction} match_score by {delta:+.3f} vs turbo baseline"
                )

    return summary


def generate_analysis_markdown(reports: dict, summary: dict) -> str:
    lines = [
        "# Whisper Model Comparison — Kirtan Analysis",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Setup",
        "",
        "- **Models**: large-v2, large-v3, large-v3-turbo (+ 4 turbo decoding variations)",
        "- **Prompt**: Mool Mantar (all configs)",
        "- **VAD**: Off (all configs)",
        "- **Audio**: 4 hazoori kirtan tracks, ~40 minutes total",
        "- **Alignment**: Word-level timestamps → matra-normalized F1 → canonical tuk matching",
        "",
        "## Results Summary",
        "",
        "| Config | Model | Segments | Match Score | Confidence | Tuk Coverage |",
        "|--------|-------|----------|-------------|------------|--------------|",
    ]

    for name in ALL_CONFIGS:
        if name in summary["configs"]:
            c = summary["configs"][name]
            lines.append(
                f"| {name} | {c['model']} | {c['total_segments']} | "
                f"{c['mean_match_score']:.3f} | {c['mean_confidence']:.3f} | "
                f"{c['tuk_coverage']*100:.1f}% |"
            )

    lines.extend([
        "",
        "## Group A: Model Comparison (v2 vs v3 vs turbo)",
        "",
    ])

    for name in ["large_v2", "large_v3", "v3_turbo_base"]:
        if name in reports:
            agg = reports[name]["aggregate"]
            lines.append(f"- **{name}**: match={agg['mean_match_score']:.3f}, "
                        f"conf={agg['mean_confidence']:.3f}, "
                        f"coverage={agg['tuk_coverage']*100:.1f}%")

    if "group_a_best" in summary:
        lines.extend(["", f"**Best model**: {summary['group_a_best']}"])

    lines.extend(["", "## Group B: Turbo Decoding Variations", ""])

    turbo_configs = ["v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
                     "v3_turbo_beam5", "v3_turbo_nocond"]
    base_score = reports.get("v3_turbo_base", {}).get("aggregate", {}).get("mean_match_score", 0)

    for name in turbo_configs:
        if name in reports:
            agg = reports[name]["aggregate"]
            delta = agg["mean_match_score"] - base_score if name != "v3_turbo_base" else 0
            delta_str = f" ({delta:+.3f})" if name != "v3_turbo_base" else " (baseline)"
            lines.append(f"- **{name}**: match={agg['mean_match_score']:.3f}{delta_str}, "
                        f"conf={agg['mean_confidence']:.3f}, "
                        f"coverage={agg['tuk_coverage']*100:.1f}%")

    if "group_b_best" in summary:
        lines.extend(["", f"**Best turbo config**: {summary['group_b_best']}"])

    lines.extend(["", "## Per-Track Breakdown", ""])

    # Per-track for each config
    for name in ALL_CONFIGS:
        if name not in reports:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Track | Style | Segments | Match Score | Confidence | Coverage |")
        lines.append("|-------|-------|----------|-------------|------------|----------|")
        for track in reports[name].get("tracks", []):
            lines.append(
                f"| {track['recording_id'][:8]} | {track['style_bucket']} | "
                f"{track['segments_produced']} | {track['mean_match_score']:.3f} | "
                f"{track['mean_confidence']:.3f} | {track['tuk_coverage']*100:.1f}% |"
            )
        lines.append("")

    if summary.get("recommendations"):
        lines.extend(["## Recommendations", ""])
        for rec in summary["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    lines.extend([
        "## Rankings",
        "",
        f"**By match score**: {' > '.join(summary['rankings'].get('by_match_score', []))}",
        "",
        f"**By confidence**: {' > '.join(summary['rankings'].get('by_confidence', []))}",
        "",
        f"**By tuk coverage**: {' > '.join(summary['rankings'].get('by_tuk_coverage', []))}",
        "",
        f"**By segments produced**: {' > '.join(summary['rankings'].get('by_segments', []))}",
        "",
        "## Dataset",
        "",
        f"Results: [{HF_REPO}](https://huggingface.co/datasets/{HF_REPO})",
        "",
        "7 splits: " + ", ".join(ALL_CONFIGS),
    ])

    return "\n".join(lines)


# ─── Main orchestration ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch Whisper model comparison")
    parser.add_argument("--upload-only", action="store_true", help="Upload tarball only")
    parser.add_argument("--monitor-only", action="store_true", help="Monitor existing pods only")
    parser.add_argument("--analyze-only", action="store_true", help="Download results and analyze")
    parser.add_argument("--keep-pods", action="store_true", help="Don't terminate pods")
    args = parser.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    if args.analyze_only:
        log.info("=== Downloading results and generating analysis ===")
        reports = download_reports_from_hf()
        if not reports:
            log.error("No splits found in HF dataset!")
            sys.exit(1)
        summary = generate_summary(reports)
        summary_path = ANALYSIS_DIR / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        md = generate_analysis_markdown(reports, summary)
        md_path = ANALYSIS_DIR / "COMPARISON_ANALYSIS.md"
        with open(md_path, "w") as f:
            f.write(md)
        log.info("Analysis: %s", md_path)
        log.info("Summary: %s", summary_path)
        return

    if args.monitor_only:
        if POD_STATE_FILE.exists():
            with open(POD_STATE_FILE) as f:
                pod_ids = json.load(f)
        else:
            log.error("No pod state file found. Run full pipeline first.")
            sys.exit(1)
        completed = monitor_completion(pod_ids, poll_interval=45, timeout=5400)
        log.info("Completed splits: %s", completed)
        return

    # ─── Full pipeline ───────────────────────────────────────────────────
    log.info("=== Whisper Model Comparison Pipeline (Self-Contained) ===")

    # Step 1: Create tarball
    if not CATALOG_PATH.exists():
        log.error("Catalog not found: %s. Run track selection first.", CATALOG_PATH)
        sys.exit(1)

    tarball_path = create_tarball()

    # Step 2: Upload tarball to HF Hub
    upload_tarball_to_hf(tarball_path)

    if args.upload_only:
        log.info("Upload complete. Run again without --upload-only to launch pods.")
        return

    # Step 3: Create auto-run template
    template_id = create_auto_run_template()
    log.info("Template ID: %s", template_id)

    # Step 4: Create 3 pods
    pod_ids = {}
    for model_info in MODELS:
        try:
            pod = create_pod_for_model(model_info, template_id)
            pod_ids[model_info["key"]] = pod["id"]
        except Exception as e:
            log.error("Failed to create pod for %s: %s", model_info["key"], e)

    if not pod_ids:
        log.error("No pods created!")
        sys.exit(1)

    # Save pod state for --monitor-only
    with open(POD_STATE_FILE, "w") as f:
        json.dump(pod_ids, f, indent=2)
    log.info("Created %d pods: %s", len(pod_ids), pod_ids)

    # Step 5: Wait for pods to be RUNNING
    for key, pid in pod_ids.items():
        try:
            wait_for_pod_ready(pid)
        except TimeoutError as e:
            log.error("Pod %s: %s", key, e)

    log.info("All pods RUNNING. Bootstrap scripts will auto-download tarball + run comparison.")
    log.info("Monitoring HF Hub for completed splits...")

    # Step 6: Monitor via HF Hub polling
    completed = monitor_completion(pod_ids, poll_interval=45, timeout=5400)

    # Step 7: Download results and analyze
    log.info("Downloading results from HF Hub...")
    reports = download_reports_from_hf()

    if reports:
        summary = generate_summary(reports)
        summary_path = ANALYSIS_DIR / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Summary: %s", summary_path)

        md = generate_analysis_markdown(reports, summary)
        md_path = ANALYSIS_DIR / "COMPARISON_ANALYSIS.md"
        with open(md_path, "w") as f:
            f.write(md)
        log.info("Analysis: %s", md_path)

        # Print final results
        for name in ALL_CONFIGS:
            if name in reports:
                agg = reports[name]["aggregate"]
                print(f"  {name:<20s}  segs={agg['total_segments']:>4d}  "
                      f"match={agg['mean_match_score']:.3f}  "
                      f"conf={agg['mean_confidence']:.3f}  "
                      f"cov={agg['tuk_coverage']*100:.1f}%")

        print(f"\nAnalysis:  {md_path}")
        print(f"Dataset:   https://huggingface.co/datasets/{HF_REPO}")
    else:
        log.warning("No results downloaded! Check pod logs.")

    # Step 8: Terminate pods
    if not args.keep_pods:
        log.info("Terminating pods...")
        for key, pid in pod_ids.items():
            try:
                runpod.terminate_pod(pid)
                log.info("  Terminated %s (%s)", key, pid[:8])
            except Exception as e:
                log.warning("  Failed: %s: %s", key, e)
    else:
        log.info("Keeping pods alive (--keep-pods)")
        for key, pid in pod_ids.items():
            log.info("  %s: %s", key, pid)


if __name__ == "__main__":
    main()
