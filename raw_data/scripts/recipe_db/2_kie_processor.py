#!/usr/bin/env python3
"""
2_kie_processor.py
==================
Step 2 of the RECIPE-DB pipeline.

What it does:
  1. Reads all OCR .txt files from {split}/ocr/ that have no matching .json in labels/
  2. Calls Gemini (thinking model) for structured KIE extraction per config.py schema
  3. Validates JSON, falls back to json_repair, strict schema enforcement
  4. Saves .json to {split}/labels/{stem}.json
  5. Tracks daily API budget (250 RPD guard) with file-based state
  6. Checkpoints every 10 calls, pushes to Oxen when daily limit approached

Rate Limits (gemini-3.1-pro-preview Tier 1):
  25 RPM / 250 RPD → use 20 RPM / 240 RPD as safe margin
  At 20 RPM (3s delay): 875 total → ~4.4 days minimum
  Strategy: run daily, script auto-stops before hitting limit

Usage:
  python 2_kie_processor.py --mode validate   # 3 OCR files only
  python 2_kie_processor.py --mode run        # full pipeline, budget-aware
  python 2_kie_processor.py --mode run --split train  # one split only
  python 2_kie_processor.py --status          # show budget status
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(cfg.PROJECT_ROOT / ".env")

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

# ─── Logging ──────────────────────────────────────────────────────────────────
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOGS_DIR / "2_kie_processor.log", mode="a"),
    ],
)
logger = logging.getLogger("kie_processor")

# ─── Checkpoint & Budget ──────────────────────────────────────────────────────
cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = cfg.CHECKPOINTS_DIR / "kie_processor.json"

def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {
        "processed": [],
        "failed":    [],
        "budget": {}   # {"YYYY-MM-DD": count}
    }

def save_checkpoint(ckpt: Dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))

def get_today_count(ckpt: Dict) -> int:
    today = str(date.today())
    return ckpt["budget"].get(today, 0)

def increment_budget(ckpt: Dict) -> None:
    today = str(date.today())
    ckpt["budget"][today] = ckpt["budget"].get(today, 0) + 1

def budget_remaining(ckpt: Dict) -> int:
    return cfg.KIE_RPD_LIMIT - get_today_count(ckpt)

# ─── Schema Utilities ─────────────────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base (in place). Only known keys from base."""
    for key, val in override.items():
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        elif isinstance(val, list) and val:
            base[key] = val
        elif isinstance(val, str) and val:
            base[key] = val
        # None / empty → keep base default


def _clean_empty_arrays(merged: Dict) -> None:
    """Remove all-empty placeholder dicts from array fields."""
    checks = {
        "store_contacts":  (merged.get("info", {}), "store_contacts", "value"),
        "items":           (merged, "items", "item_name"),
        "returned_items":  (merged, "returned_items", "item_name"),
        "discounts":       (merged.get("payment", {}), "discounts", "amount"),
        "taxes":           (merged.get("payment", {}), "taxes", "amount"),
        "additional_charges": (merged.get("payment", {}), "additional_charges", "amount"),
    }
    for _field, (container, key, primary) in checks.items():
        if isinstance(container, dict) and isinstance(container.get(key), list):
            container[key] = [
                item for item in container[key]
                if isinstance(item, dict) and item.get(primary, "").strip()
            ]


def _ensure_arrays(merged: Dict) -> None:
    """Guarantee all array fields are lists (never None or string)."""
    root_arrays = ["items", "returned_items"]
    info_arrays = ["store_contacts"]
    pay_arrays  = ["discounts", "taxes", "additional_charges"]

    for f in root_arrays:
        if not isinstance(merged.get(f), list):
            merged[f] = []
    for f in info_arrays:
        if not isinstance(merged.get("info", {}).get(f), list):
            merged.setdefault("info", {})[f] = []
    for f in pay_arrays:
        if not isinstance(merged.get("payment", {}).get(f), list):
            merged.setdefault("payment", {})[f] = []


def validate_and_merge(ai_result: Dict) -> Dict:
    """
    Merge AI result into clean schema defaults.
    Ensures: correct structure, no extra keys, no broken arrays.
    """
    merged = copy.deepcopy(cfg.EXTRACTION_SCHEMA)
    _deep_merge(merged, ai_result)
    _ensure_arrays(merged)
    _clean_empty_arrays(merged)
    return merged

# ─── JSON Parsing ─────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> Dict:
    """
    3-layer JSON parsing:
      Layer 1: json.loads (should succeed with response_format=json_object)
      Layer 2: json_repair fallback
      Layer 3: raise with diagnostics
    """
    content = raw.strip()

    # Strip markdown code fences (defensive)
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Find first `{` (strip any leaked preamble from thinking models)
    brace_idx = content.find("{")
    if brace_idx > 0:
        logger.debug(f"Stripped {brace_idx} chars of leading preamble")
        content = content[brace_idx:]

    # Layer 1: standard parse
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(content)
        return obj
    except json.JSONDecodeError as e1:
        logger.warning(f"json.loads failed: {e1} — trying json_repair")

    # Layer 2: json_repair
    if JSON_REPAIR_AVAILABLE:
        try:
            result = repair_json(content, return_objects=True)
            if isinstance(result, dict):
                logger.info("json_repair: recovered")
                return result
        except Exception as e2:
            logger.warning(f"json_repair also failed: {e2}")
    else:
        logger.warning("json_repair not installed — pip install json-repair")

    raise json.JSONDecodeError(
        f"All parse layers failed. First 400 chars: {content[:400]}",
        content, 0
    )

# ─── Gemini API Call ──────────────────────────────────────────────────────────

def call_gemini(client: OpenAI, prompt: str, stem: str) -> str:
    """
    Call Gemini via OpenAI-compatible API.
    response_format=json_object → API-level JSON constraint (prevents malformed output).
    """
    MAX_RETRIES = 3
    BACKOFF     = [10, 30, 60]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise receipt information extractor. "
                            "Extract ONLY information explicitly visible in the OCR text. "
                            "Never calculate, infer, or assume any values. "
                            "Output strictly valid JSON matching the provided schema exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        except Exception as e:
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            logger.warning(
                f"[{stem}] Gemini attempt {attempt+1}/3 failed: {e}. Retry in {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(f"[{stem}] Gemini failed after {MAX_RETRIES} attempts")

# ─── Per-File Worker ──────────────────────────────────────────────────────────

def process_ocr_file(
    client: OpenAI,
    txt_path: Path,
    split: str,
    stem: str,
) -> bool:
    """
    Process one OCR .txt file → extract KIE JSON → save to labels/.
    Returns True on success, False on failure.
    """
    label_dst = cfg.labels_dir(split) / f"{stem}.json"

    # Skip if already done
    if label_dst.exists():
        logger.debug(f"[{stem}] Already has label, skipping")
        return True

    # Read OCR text
    try:
        ocr_text = txt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error(f"[{stem}] Cannot read OCR file: {e}")
        return False

    if not ocr_text:
        logger.warning(f"[{stem}] Empty OCR text, skipping")
        return False

    # Build prompt and call Gemini
    try:
        prompt   = cfg.build_kie_prompt(ocr_text)
        raw_resp = call_gemini(client, prompt, stem)
    except Exception as e:
        logger.error(f"[{stem}] API call failed: {e}")
        return False

    # Parse JSON
    try:
        ai_result = parse_json_response(raw_resp)
    except json.JSONDecodeError as e:
        logger.error(f"[{stem}] JSON parse failed: {e}")
        return False

    # Merge + validate schema
    try:
        merged = validate_and_merge(ai_result)
    except Exception as e:
        logger.error(f"[{stem}] Schema merge failed: {e}")
        return False

    # Save label
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    n_items   = len(merged.get("items", []))
    store     = merged.get("info", {}).get("store_name", "")[:30]
    total     = merged.get("payment", {}).get("grand_total", "")
    logger.info(f"[{stem}] ✓ {split} | items={n_items} | store='{store}' | total={total}")
    return True

# ─── Work Plan ────────────────────────────────────────────────────────────────

def collect_pending(mode: str, split_filter: Optional[str]) -> List[tuple]:
    """
    Return list of (txt_path, split, stem) for all unprocessed OCR files.
    Files with existing .json labels are skipped.
    """
    ckpt = load_checkpoint()
    processed_set = set(ckpt["processed"])

    splits = ["train", "test"]
    if split_filter:
        splits = [split_filter]

    pending: List[tuple] = []
    for split in splits:
        ocr_path = cfg.ocr_dir(split)
        if not ocr_path.exists():
            logger.warning(f"OCR dir not found: {ocr_path} — run step 1 first")
            continue

        for txt in sorted(ocr_path.glob("*.txt")):
            stem = txt.stem
            label = cfg.labels_dir(split) / f"{stem}.json"

            if stem in processed_set or label.exists():
                continue
            pending.append((txt, split, stem))

    if mode == "validate":
        pending = pending[:3]
        logger.info(f"VALIDATE mode: processing first {len(pending)} OCR files only")

    return pending, ckpt

# ─── Oxen Push ────────────────────────────────────────────────────────────────

def oxen_push(commit_msg: str) -> None:
    try:
        import oxen
        from oxen.auth import config_auth

        if cfg.OXEN_AUTH_TOKEN:
            config_auth(cfg.OXEN_AUTH_TOKEN)

        repo = oxen.Repo(str(cfg.PROJECT_ROOT))
        repo.add(str(cfg.RECIPE_DB_DIR))
        repo.commit(commit_msg)
        repo.push()
        logger.info(f"Oxen push: '{commit_msg}'")
    except Exception as e:
        logger.warning(f"Oxen push skipped (non-fatal): {e}")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def show_status() -> None:
    """Print current budget + completion status."""
    ckpt = load_checkpoint()
    today = str(date.today())
    used  = ckpt["budget"].get(today, 0)
    total_done = len(ckpt["processed"])
    total_fail = len(ckpt["failed"])

    print(f"\n{'=' * 50}")
    print(f"  RECIPE-DB KIE Processor Status")
    print(f"{'=' * 50}")
    print(f"  Today ({today}):")
    print(f"    API calls used  : {used} / {cfg.KIE_RPD_LIMIT}")
    print(f"    Remaining today : {cfg.KIE_RPD_LIMIT - used}")
    print(f"  All time:")
    print(f"    Processed       : {total_done}")
    print(f"    Failed          : {total_fail}")
    print(f"{'=' * 50}\n")


def run_pipeline(mode: str, split_filter: Optional[str]) -> None:
    """Main pipeline orchestrator with daily budget guard."""

    logger.info("=" * 65)
    logger.info(f"  RECIPE-DB KIE Processor — Mode: {mode.upper()}")
    logger.info("=" * 65)
    logger.info(f"  LLM model     : {cfg.LLM_MODEL}")
    logger.info(f"  LLM endpoint  : {cfg.LLM_ENDPOINT}")
    logger.info(f"  Temperature   : {cfg.LLM_TEMPERATURE}")
    logger.info(f"  Rate limit    : {cfg.KIE_RPM_LIMIT} RPM / {cfg.KIE_RPD_LIMIT} RPD")
    logger.info(f"  Delay         : {cfg.KIE_DELAY_SECONDS:.1f}s between requests")
    logger.info(f"  Prompt Rep    : {'ENABLED' if cfg.PROMPT_REPETITION else 'DISABLED (thinking mode)'}")
    logger.info(f"  json_repair   : {'available ✓' if JSON_REPAIR_AVAILABLE else 'NOT installed'}")
    logger.info("=" * 65)

    if not cfg.LLM_API_KEY:
        logger.error("LLM_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_ENDPOINT)

    pending, ckpt = collect_pending(mode, split_filter)
    if not pending:
        logger.info("Nothing to process — all labels already exist.")
        show_status()
        return

    remaining_today = budget_remaining(ckpt)
    logger.info(
        f"\n  Pending: {len(pending)} files | "
        f"Budget today: {get_today_count(ckpt)} used / {remaining_today} remaining"
    )

    if remaining_today <= 0 and mode == "run":
        logger.warning(
            f"Daily RPD limit ({cfg.KIE_RPD_LIMIT}) reached for today. "
            f"Run again tomorrow or switch to gemini-3-flash (10k RPD) in .env."
        )
        show_status()
        return

    if mode == "run":
        # Limit this run to remaining budget
        pending = pending[:remaining_today]
        logger.info(f"  Capped to {len(pending)} files (daily budget remaining)")

    success_count = 0
    fail_count    = 0
    CHECKPOINT_EVERY = 10  # Checkpoint frequently due to daily budget value
    OXEN_PUSH_EVERY  = 50  # Push to Oxen periodically

    last_request_time = 0.0

    for i, (txt_path, split, stem) in enumerate(pending, 1):
        logger.info(f"\n[{i}/{len(pending)}] [{split}] {stem}")

        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < cfg.KIE_DELAY_SECONDS and i > 1:
            time.sleep(cfg.KIE_DELAY_SECONDS - elapsed)

        ok = process_ocr_file(client, txt_path, split, stem)
        last_request_time = time.time()

        if ok:
            success_count += 1
            ckpt["processed"].append(stem)
            increment_budget(ckpt)
        else:
            fail_count += 1
            ckpt["failed"].append(stem)

        # Checkpoint
        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt)
            logger.info(
                f"  ── Checkpoint [{i}/{len(pending)}] "
                f"success={success_count} fail={fail_count} "
                f"budget_used={get_today_count(ckpt)}"
            )

        # Periodic Oxen push (every 50 to avoid too many small pushes)
        if i % OXEN_PUSH_EVERY == 0 and mode == "run":
            oxen_push(
                f"feat(recipe-db): add KIE labels batch — "
                f"{success_count} labels ({i}/{len(pending)})"
            )

        # Stop if budget exhausted mid-run (safety guard)
        if budget_remaining(ckpt) <= 0 and mode == "run":
            logger.warning(
                "Daily RPD limit reached mid-run. "
                "Progress saved. Run again tomorrow."
            )
            break

    # Final checkpoint
    save_checkpoint(ckpt)

    # Validate mode: print sample outputs
    if mode == "validate":
        logger.info(f"\n{'=' * 65}")
        logger.info("  VALIDATE OUTPUT — Sample KIE Results")
        logger.info(f"{'=' * 65}")
        for txt_path, split, stem in pending[:3]:
            label = cfg.labels_dir(split) / f"{stem}.json"
            if label.exists():
                data   = json.loads(label.read_text())
                info   = data.get("info", {})
                pay    = data.get("payment", {})
                items  = data.get("items", [])
                logger.info(f"\n── [{split}] {stem}")
                logger.info(f"   store_name   : {info.get('store_name', '')}")
                logger.info(f"   payment_date : {info.get('payment_date', '')}")
                logger.info(f"   grand_total  : {pay.get('grand_total', '')}")
                logger.info(f"   items count  : {len(items)}")

    # Summary
    logger.info(f"\n{'=' * 65}")
    logger.info("  KIE PROCESSING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Success    : {success_count}")
    logger.info(f"  Failed     : {fail_count}")
    logger.info(f"  Budget used today: {get_today_count(ckpt)} / {cfg.KIE_RPD_LIMIT}")
    logger.info("=" * 65)

    # Final Oxen push
    if mode == "run" and success_count > 0:
        oxen_push(
            f"feat(recipe-db): KIE labels — "
            f"{success_count} labels added "
            f"({len(ckpt['processed'])} total)"
        )

    show_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Extract structured KIE JSON from OCR text using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check remaining daily budget:
  python 2_kie_processor.py --status

  # Test with 3 files:
  python 2_kie_processor.py --mode validate

  # Full run (stops at daily RPD limit, resume next day):
  python 2_kie_processor.py --mode run

  # Train split only:
  python 2_kie_processor.py --mode run --split train

  # If Gemini Pro limit hit, switch to flash in .env:
  # LLM_MODEL=gemini-2.5-flash (10k RPD, high thinking)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "run"],
        default="validate",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default=None,
        help="Process one split only",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show budget status and exit",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    run_pipeline(args.mode, args.split)


if __name__ == "__main__":
    main()