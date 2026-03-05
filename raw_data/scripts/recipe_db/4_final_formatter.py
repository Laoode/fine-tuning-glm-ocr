#!/usr/bin/env python3
"""
4_final_formatter.py
====================
Step 4 (final) of the RECIPE-DB pipeline.

What it does:
  1. Reads verified labels/*.json + images/*.jpg for each split (train/test)
  2. Validates each image is readable (skips corrupt/unreadable)
  3. Converts to LLaMA-Factory ShareGPT format per GLM-OCR fine-tuning spec
  4. Writes: LLaMA-Factory/data/recipe_db_train.json
             LLaMA-Factory/data/recipe_db_test.json
  5. Writes: LLaMA-Factory/data/dataset_info.json (patch/merge)
  6. Generates: dataset statistics report
  7. Pushes everything to Oxen

LLaMA-Factory ShareGPT format (per GLM-OCR fine-tuning guide):
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>请按下列JSON格式输出图中信息:\n{schema}"
      },
      {
        "role": "assistant",
        "content": "{extracted_json_string}"
      }
    ],
    "images": ["recipe_db/train/images/{stem}.jpg"]
  }

Image paths are relative to LLaMA-Factory/data/ directory.

Usage:
  python 4_final_formatter.py --mode validate   # 5 samples, inspect output
  python 4_final_formatter.py --mode run        # full run
  python 4_final_formatter.py --mode stats      # statistics only
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from PIL import Image, UnidentifiedImageError

# ─── Logging ──────────────────────────────────────────────────────────────────
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOGS_DIR / "4_final_formatter.log", mode="a"),
    ],
)
logger = logging.getLogger("final_formatter")

# ─── Output Paths ─────────────────────────────────────────────────────────────
# LLaMA-Factory/data/ is the root for all data
LLAMA_DATA_DIR = cfg.PROJECT_ROOT / "LLaMA-Factory" / "data"
DATASET_INFO   = LLAMA_DATA_DIR / "dataset_info.json"

def llama_dataset_file(split: str) -> Path:
    return LLAMA_DATA_DIR / f"recipe_db_{split}.json"

# Image paths in LLaMA-Factory are relative to LLaMA-Factory/data/
def llama_image_rel_path(split: str, stem: str) -> str:
    return f"recipe_db/{split}/images/{stem}.jpg"

# ─── Image Validation ─────────────────────────────────────────────────────────

def is_image_readable(img_path: Path) -> bool:
    """
    Return True if image can be opened and decoded by PIL.
    Silently returns False for corrupt, truncated, or unsupported files.
    """
    try:
        with Image.open(img_path) as im:
            im.verify()   # Catches truncated/corrupt files
        # Reopen after verify (verify() leaves file in unknown state)
        with Image.open(img_path) as im:
            im.load()     # Force full decode
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False

# ─── Schema Helpers ───────────────────────────────────────────────────────────

def _ensure_all_schema_keys(label: Dict) -> Dict:
    """
    Ensure label dict has all keys from EXTRACTION_SCHEMA.
    Missing keys are filled with schema defaults.
    Does NOT remove extra keys (user may have added metadata).
    """
    import copy

    def _fill(base: Dict, schema: Dict) -> Dict:
        result = copy.deepcopy(base)
        for k, v in schema.items():
            if k not in result:
                result[k] = copy.deepcopy(v)
            elif isinstance(v, dict) and isinstance(result[k], dict):
                result[k] = _fill(result[k], v)
        return result

    return _fill(label, cfg.EXTRACTION_SCHEMA)


def _clean_empty_arrays(label: Dict) -> Dict:
    """Remove all-empty placeholder dicts from array fields."""
    import copy
    label = copy.deepcopy(label)

    # (container_path, field_name, primary_key_for_emptiness_check)
    checks = [
        (label.get("info", {}),    "store_contacts",    "value"),
        (label,                    "items",              "item_name"),
        (label,                    "returned_items",     "item_name"),
        (label.get("payment", {}), "discounts",          "amount"),
        (label.get("payment", {}), "taxes",              "amount"),
        (label.get("payment", {}), "additional_charges", "amount"),
    ]
    for container, field, primary in checks:
        if isinstance(container, dict) and isinstance(container.get(field), list):
            container[field] = [
                item for item in container[field]
                if isinstance(item, dict) and item.get(primary, "").strip()
            ]
    return label

# ─── Sample Collection ────────────────────────────────────────────────────────

def collect_samples(split: str) -> List[Tuple[str, Path, Path]]:
    """
    Return (stem, image_path, label_path) for all valid pairs in a split.
    Skips: missing image, corrupt image, missing label, invalid JSON.
    """
    lbl_dir = cfg.labels_dir(split)
    img_dir = cfg.images_dir(split)

    if not lbl_dir.exists():
        logger.warning(f"Labels dir not found: {lbl_dir} — run steps 1-3 first")
        return []

    samples     = []
    skip_no_img = 0
    skip_corrupt = 0
    skip_bad_json = 0

    for lbl_path in sorted(lbl_dir.glob("*.json")):
        stem     = lbl_path.stem
        img_path = img_dir / f"{stem}.jpg"

        # Check image exists
        if not img_path.exists():
            logger.warning(f"[{stem}] No image, skipping")
            skip_no_img += 1
            continue

        # Check image is readable (not corrupt)
        if not is_image_readable(img_path):
            logger.warning(f"[{stem}] Corrupt/unreadable image, skipping")
            skip_corrupt += 1
            continue

        # Check label JSON is parseable
        try:
            json.loads(lbl_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"[{stem}] Invalid label JSON: {e}, skipping")
            skip_bad_json += 1
            continue

        samples.append((stem, img_path, lbl_path))

    logger.info(
        f"[{split}] {len(samples)} valid samples | "
        f"skipped: no_img={skip_no_img} corrupt={skip_corrupt} bad_json={skip_bad_json}"
    )
    return samples

# ─── Format Conversion ────────────────────────────────────────────────────────

def to_sharegpt_sample(split: str, stem: str, lbl_path: Path) -> Optional[Dict]:
    """
    Convert one label JSON to LLaMA-Factory ShareGPT format.

    User prompt:   "<image>请按下列JSON格式输出图中信息:\n{schema}"
    Assistant:     compact JSON string of extracted data
    Image path:    relative to LLaMA-Factory/data/ (e.g., "recipe_db/train/images/cord_00001.jpg")
    """
    try:
        label = json.loads(lbl_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[{stem}] Cannot read label: {e}")
        return None

    # Ensure all schema keys present + clean empty placeholders
    label = _ensure_all_schema_keys(label)
    label = _clean_empty_arrays(label)

    # Build user prompt and assistant response
    user_prompt   = cfg.build_glm_finetune_user_prompt()
    assistant_str = json.dumps(label, ensure_ascii=False, separators=(",", ":"))

    return {
        "messages": [
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_str},
        ],
        "images": [llama_image_rel_path(split, stem)],
    }

# ─── Dataset Info JSON ────────────────────────────────────────────────────────

def build_dataset_info_entry(split: str) -> Dict:
    """
    Build dataset_info.json entry for one split.
    File path is relative to LLaMA-Factory/data/.
    """
    return {
        f"recipe_db_{split}": {
            "file_name":  f"recipe_db_{split}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images":   "images",
            },
            "tags": {
                "role_tag":    "role",
                "content_tag": "content",
                "user_tag":    "user",
                "assistant_tag": "assistant",
            },
        }
    }


def patch_dataset_info(new_entries: Dict) -> None:
    """
    Merge new_entries into existing dataset_info.json.
    Creates the file if it doesn't exist.
    """
    existing = {}
    if DATASET_INFO.exists():
        try:
            existing = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Existing dataset_info.json is invalid — overwriting")

    existing.update(new_entries)
    DATASET_INFO.parent.mkdir(parents=True, exist_ok=True)
    DATASET_INFO.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"dataset_info.json updated → {DATASET_INFO}")

# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(split: str, samples: List[Dict]) -> Dict:
    """Compute dataset statistics for a split."""
    n_items       = 0
    n_with_items  = 0
    n_has_total   = 0
    source_counts: Dict[str, int] = {}

    for sample in samples:
        # Decode assistant content to get the label
        try:
            label = json.loads(sample["messages"][1]["content"])
        except Exception:
            continue

        items = label.get("items", [])
        n_items     += len(items)
        n_with_items += 1 if items else 0
        n_has_total  += 1 if label.get("payment", {}).get("grand_total") else 0

        # Source from image path stem prefix
        img_path = sample["images"][0]
        stem   = Path(img_path).stem
        prefix = stem.split("_")[0] if "_" in stem else "unknown"
        source_counts[prefix] = source_counts.get(prefix, 0) + 1

    return {
        "total_samples":   len(samples),
        "n_with_items":    n_with_items,
        "n_has_total":     n_has_total,
        "avg_items":       round(n_items / len(samples), 2) if samples else 0,
        "by_source":       source_counts,
    }


def print_stats(split: str, stats: Dict) -> None:
    logger.info(f"\n  [{split.upper()}] Statistics:")
    logger.info(f"    Total samples  : {stats['total_samples']}")
    logger.info(f"    With items     : {stats['n_with_items']}")
    logger.info(f"    Has grand_total: {stats['n_has_total']}")
    logger.info(f"    Avg items/rcpt : {stats['avg_items']}")
    logger.info(f"    By source      : {stats['by_source']}")

# ─── Oxen Push ────────────────────────────────────────────────────────────────

def oxen_push(commit_msg: str) -> None:
    try:
        import oxen
        from oxen.auth import config_auth

        if cfg.OXEN_AUTH_TOKEN:
            config_auth(cfg.OXEN_AUTH_TOKEN)

        repo = oxen.Repo(str(cfg.PROJECT_ROOT))
        # Stage the LLaMA-Factory data directory and the recipe_db labels
        repo.add(str(LLAMA_DATA_DIR))
        repo.add(str(cfg.RECIPE_DB_DIR))
        repo.commit(commit_msg)
        repo.push()
        logger.info(f"Oxen push: '{commit_msg}'")
    except Exception as e:
        logger.warning(f"Oxen push skipped (non-fatal): {e}")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_stats_only() -> None:
    """Print stats for current state of dataset without writing output files."""
    for split in ["train", "test"]:
        samples_meta = collect_samples(split)
        if not samples_meta:
            continue
        samples = []
        for stem, img_path, lbl_path in samples_meta:
            s = to_sharegpt_sample(split, stem, lbl_path)
            if s:
                samples.append(s)
        stats = compute_stats(split, samples)
        print_stats(split, stats)


def run_pipeline(mode: str) -> None:
    """Main formatter pipeline."""
    logger.info("=" * 65)
    logger.info(f"  RECIPE-DB Final Formatter — Mode: {mode.upper()}")
    logger.info("=" * 65)
    logger.info(f"  Output dir : {LLAMA_DATA_DIR}")
    logger.info(f"  Schema keys: {list(cfg.EXTRACTION_SCHEMA.keys())}")
    logger.info("=" * 65)

    LLAMA_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_results:  Dict[str, List[Dict]] = {"train": [], "test": []}
    dataset_info_entries: Dict = {}

    for split in ["train", "test"]:
        logger.info(f"\n{'─'*65}")
        logger.info(f"  Processing split: {split.upper()}")
        logger.info(f"{'─'*65}")

        samples_meta = collect_samples(split)
        if not samples_meta:
            logger.info(f"[{split}] No samples found — skipping")
            continue

        # Validate mode: limit to first 5 across all splits combined
        if mode == "validate":
            samples_meta = samples_meta[:3] if split == "train" else samples_meta[:2]
            logger.info(f"VALIDATE mode: capped to {len(samples_meta)} samples")

        formatted    = []
        skip_convert = 0

        for stem, img_path, lbl_path in samples_meta:
            sample = to_sharegpt_sample(split, stem, lbl_path)
            if sample is None:
                skip_convert += 1
                continue
            formatted.append(sample)
            logger.debug(f"[{stem}] ✓ formatted")

        logger.info(
            f"[{split}] Formatted: {len(formatted)} | "
            f"Conversion errors: {skip_convert}"
        )

        if not formatted:
            logger.warning(f"[{split}] No formatted samples — skipping output")
            continue

        all_results[split] = formatted

        # Write output JSON
        out_file = llama_dataset_file(split)
        out_file.write_text(
            json.dumps(formatted, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[{split}] Saved {len(formatted)} samples → {out_file}")

        # Statistics
        stats = compute_stats(split, formatted)
        print_stats(split, stats)

        # Build dataset_info entry
        dataset_info_entries.update(build_dataset_info_entry(split))

    # Patch dataset_info.json
    if dataset_info_entries:
        patch_dataset_info(dataset_info_entries)

    # Validate mode: print sample outputs
    if mode == "validate":
        logger.info(f"\n{'='*65}")
        logger.info("  VALIDATE OUTPUT — Sample LLaMA-Factory Records")
        logger.info(f"{'='*65}")
        for split in ["train", "test"]:
            for i, sample in enumerate(all_results[split][:2], 1):
                try:
                    label = json.loads(sample["messages"][1]["content"])
                    info  = label.get("info", {})
                    pay   = label.get("payment", {})
                    items = label.get("items", [])
                    logger.info(f"\n── [{split.upper()} {i}] {sample['images'][0]}")
                    logger.info(f"   store_name   : {info.get('store_name', '')}")
                    logger.info(f"   payment_date : {info.get('payment_date', '')}")
                    logger.info(f"   grand_total  : {pay.get('grand_total', '')}")
                    logger.info(f"   items count  : {len(items)}")
                    logger.info(f"   user_prompt  : {sample['messages'][0]['content'][:80]}...")
                except Exception as e:
                    logger.warning(f"   Cannot decode sample: {e}")

    # Final summary
    total_train = len(all_results["train"])
    total_test  = len(all_results["test"])

    logger.info(f"\n{'='*65}")
    logger.info("  FORMATTING COMPLETE")
    logger.info(f"{'='*65}")
    logger.info(f"  Train samples : {total_train}")
    logger.info(f"  Test samples  : {total_test}")
    logger.info(f"  Total         : {total_train + total_test}")
    logger.info(f"  Files:")
    for split in ["train", "test"]:
        if all_results[split]:
            logger.info(f"    {llama_dataset_file(split)}")
    logger.info(f"    {DATASET_INFO}")
    logger.info(f"\n  Next step — LoRA fine-tuning:")
    logger.info(f"    cd LLaMA-Factory")
    logger.info(f"    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \\")
    logger.info(f"      llamafactory-cli train config/glm_ocr_lora_sft.yaml")
    logger.info("=" * 65)

    # Oxen push (run mode only)
    if mode == "run" and (total_train + total_test) > 0:
        oxen_push(
            f"feat(recipe-db): final LLaMA-Factory dataset — "
            f"train={total_train} test={total_test}"
        )

# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Convert verified KIE labels to LLaMA-Factory format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect 5 sample outputs first:
  python 4_final_formatter.py --mode validate

  # Full run:
  python 4_final_formatter.py --mode run

  # Just print dataset statistics:
  python 4_final_formatter.py --mode stats

Output:
  LLaMA-Factory/data/recipe_db_train.json
  LLaMA-Factory/data/recipe_db_test.json
  LLaMA-Factory/data/dataset_info.json   (patched/merged)

Then register in LLaMA-Factory and train:
  # dataset_info.json is auto-patched, just update your YAML:
  dataset: recipe_db_train

  DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \\
    llamafactory-cli train LLaMA-Factory/config/glm_ocr_lora_sft.yaml
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "run", "stats"],
        default="validate",
        help="validate: 5 samples | run: full pipeline | stats: statistics only",
    )
    args = parser.parse_args()

    if args.mode == "stats":
        run_stats_only()
    else:
        run_pipeline(args.mode)


if __name__ == "__main__":
    main()