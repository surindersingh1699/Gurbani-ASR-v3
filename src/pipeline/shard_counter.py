"""Shard counter server — runs on Hetzner to coordinate parallel GPU uploads.

Each isolated RunPod agent calls /next to claim a unique shard index before
uploading its parquet file to HF Hub. This prevents the random-hash naming
problem that occurs when multiple agents call push_to_hub() independently.

Usage (on Hetzner):
    python src/pipeline/shard_counter.py

Or via infisical:
    infisical run --env=prod -- python src/pipeline/shard_counter.py

Client usage (on RunPod agents):
    from src.pipeline.shard_counter import claim_shard, mark_done
    idx = claim_shard("http://138.199.174.101:9112")
    # ... upload as data/train-{idx:05d}.parquet ...
    mark_done("http://138.199.174.101:9112", idx)
"""

import argparse
import json
import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, request

log = logging.getLogger("shard-counter")

# ── Server ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_lock = threading.Lock()

# Persistent state file so counter survives restarts
STATE_FILE = Path("data/shard_counter_state.json")


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"counter": 0, "claimed": {}, "completed": []}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


_state = _load_state()


@app.route("/next")
def next_index():
    """Claim the next shard index. Returns {"index": N}."""
    agent = request.args.get("agent", "unknown")
    with _lock:
        idx = _state["counter"]
        _state["counter"] += 1
        _state["claimed"][str(idx)] = agent
        _save_state(_state)
    log.info("Shard %d claimed by %s", idx, agent)
    return jsonify({"index": idx})


@app.route("/done/<int:idx>", methods=["POST"])
def mark_done(idx):
    """Mark a shard as successfully uploaded."""
    with _lock:
        if idx not in _state["completed"]:
            _state["completed"].append(idx)
        _save_state(_state)
    log.info("Shard %d upload confirmed", idx)
    return jsonify({
        "completed": len(_state["completed"]),
        "claimed": _state["counter"],
    })


@app.route("/status")
def status():
    """Current counter state — claimed vs completed shards."""
    with _lock:
        missing = [
            i for i in range(_state["counter"])
            if i not in _state["completed"]
        ]
        return jsonify({
            "counter": _state["counter"],
            "claimed": _state["claimed"],
            "completed": sorted(_state["completed"]),
            "missing": missing,
            "total_claimed": _state["counter"],
            "total_completed": len(_state["completed"]),
        })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset counter to 0. Use only when starting a fresh upload batch."""
    with _lock:
        _state["counter"] = 0
        _state["claimed"] = {}
        _state["completed"] = []
        _save_state(_state)
    log.info("Counter reset")
    return jsonify({"status": "reset"})


# ── Client helpers (import these on RunPod agents) ──────────────────────────

def claim_shard(counter_url: str, agent_id: str = "unknown") -> int:
    """Claim the next shard index from the counter server.

    Args:
        counter_url: Base URL of the counter server (e.g. http://138.199.174.101:9112)
        agent_id: Identifier for this agent (for tracking)

    Returns:
        The claimed shard index (int).
    """
    import requests

    resp = requests.get(f"{counter_url}/next", params={"agent": agent_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()["index"]


def claim_shard_safe(counter_url: str, agent_id: str = "unknown", fallback_file: str = ".shard_fallback") -> int:
    """Claim shard index with local file fallback if server is unreachable."""
    try:
        return claim_shard(counter_url, agent_id)
    except Exception:
        log.warning("Shard counter unreachable — using local fallback")
        import filelock
        lock = filelock.FileLock(f"{fallback_file}.lock")
        with lock:
            path = Path(fallback_file)
            idx = int(path.read_text().strip()) if path.exists() else 10000  # high offset to avoid collisions
            path.write_text(str(idx + 1))
        return idx


def report_done(counter_url: str, idx: int) -> dict:
    """Report that a shard was successfully uploaded.

    Args:
        counter_url: Base URL of the counter server.
        idx: The shard index that was uploaded.

    Returns:
        Server response with completed/claimed counts.
    """
    import requests

    resp = requests.post(f"{counter_url}/done/{idx}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def upload_shard(dataset, repo_id: str, shard_idx: int, hf_token: str,
                 counter_url: str = None, agent_id: str = "unknown"):
    """Save dataset as parquet and upload with correct sequential name.

    Args:
        dataset: A datasets.Dataset object to upload.
        repo_id: HF Hub repo ID (e.g. surindersinghssj/gurbani-asr-dataset).
        shard_idx: The shard index (from claim_shard).
        hf_token: HuggingFace write token.
        counter_url: If provided, reports done after upload.
        agent_id: Identifier for this agent.
    """
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    filename = f"train-{shard_idx:05d}.parquet"

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / filename
        dataset.to_parquet(str(local_path))

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{filename}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload shard {shard_idx:05d} from {agent_id}",
        )

    log.info("Uploaded %s to %s", filename, repo_id)

    if counter_url:
        report_done(counter_url, shard_idx)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Shard counter server")
    parser.add_argument("--port", type=int, default=9112)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reset", action="store_true", help="Reset counter before starting")
    args = parser.parse_args()

    if args.reset:
        _state["counter"] = 0
        _state["claimed"] = {}
        _state["completed"] = []
        _save_state(_state)
        log.info("Counter reset on startup")

    app.run(host=args.host, port=args.port)
