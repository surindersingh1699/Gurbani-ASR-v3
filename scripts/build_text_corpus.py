"""Build Phase 1 text corpus from STTM database.sqlite.

Extracts all Gurbani lines, converts STTM ASCII → Unicode Gurmukhi,
and pushes to HF Hub as a text-only dataset for decoder LM priming.

Usage:
    python scripts/build_text_corpus.py              # push to HF Hub
    python scripts/build_text_corpus.py --dry-run    # preview without pushing
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unicode_convert import ascii_to_unicode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "database.sqlite"

# Source names from the STTM database
SOURCE_NAMES = {
    1: "Sri Guru Granth Sahib Ji",
    2: "Sri Dasam Granth",
    3: "Vaaran Bhai Gurdas Ji",
    4: "Kabit Savaiye Bhai Gurdas Ji",
    5: "Ghazals Bhai Nand Lal Ji",
    6: "Zindagi Nama Bhai Nand Lal Ji",
    7: "Ganj Nama Bhai Nand Lal Ji",
    8: "Jot Bigas Bhai Nand Lal Ji",
    9: "Ardaas",
    11: "Sarabloh Granth",
}

HF_REPO = os.environ.get("HF_TEXT_CORPUS_REPO", "surindersinghssj/gurbani-asr-text")


def extract_lines(db_path: Path) -> list[dict]:
    """Extract all lines from database, convert to Unicode, return as dicts."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT l.gurmukhi, l.source_page, l.order_id, sh.source_id
        FROM lines l
        JOIN shabads sh ON l.shabad_id = sh.id
        ORDER BY l.order_id
    """).fetchall()
    conn.close()

    records = []
    skipped = 0
    for row in rows:
        unicode_text = ascii_to_unicode(row["gurmukhi"])
        if not unicode_text or len(unicode_text.strip()) < 2:
            skipped += 1
            continue
        records.append({
            "sentence": unicode_text,
            "source_id": row["source_id"],
            "source_name": SOURCE_NAMES.get(row["source_id"], "Unknown"),
            "source_page": row["source_page"],
            "order_id": row["order_id"],
        })

    log.info("Extracted %d lines (%d skipped as empty)", len(records), skipped)
    return records


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1 text corpus from STTM database")
    parser.add_argument("--dry-run", action="store_true", help="Preview without pushing to HF")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to database.sqlite")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    records = extract_lines(db_path)

    # Print stats per source
    from collections import Counter
    source_counts = Counter(r["source_id"] for r in records)
    for sid, count in sorted(source_counts.items()):
        name = SOURCE_NAMES.get(sid, "Unknown")
        log.info("  Source %d (%s): %d lines", sid, name, count)

    # Show a few samples
    log.info("Sample lines:")
    for r in records[:5]:
        log.info("  [%s p.%d] %s", r["source_name"][:20], r["source_page"], r["sentence"][:80])

    if args.dry_run:
        log.info("DRY RUN — would push %d rows to %s", len(records), HF_REPO)
        return

    # Build HF dataset and push
    from datasets import Dataset

    ds = Dataset.from_list(records)
    log.info("Dataset: %s", ds)
    log.info("Pushing to %s ...", HF_REPO)
    ds.push_to_hub(HF_REPO, private=False)
    log.info("Done — %d rows pushed to %s", len(ds), HF_REPO)


if __name__ == "__main__":
    main()
