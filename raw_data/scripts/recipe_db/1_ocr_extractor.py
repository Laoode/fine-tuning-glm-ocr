#!/usr/bin/env python3
"""
1_ocr_extractor.py
==================
Step 1 of the RECIPE-DB pipeline.

What it does:
  1. Collects images from all SOURCE_CONFIGS (with per-source limits)
  2. Validates images (skips corrupt), converts all to JPEG
  3. Splits 95% train / 5% test per source (sorted order, deterministic)
  4. Copies images → LLaMA-Factory/data/recipe_db/{split}/images/{prefix}_{stem}.jpg
  5. Runs GLM-OCR via vLLM concurrently (ThreadPoolExecutor) for GPU saturation
  6. Saves raw OCR text → {split}/ocr/{prefix}_{stem}.txt
  7. Checkpoints every 50 images, pushes to Oxen on completion

Usage:
  python 1_ocr_extractor.py --mode validate   # 3 images from first source only
  python 1_ocr_extractor.py --mode run        # full pipeline, checkpoint-resumable
  python 1_ocr_extractor.py --mode run --source cord  # single source only
"""

import argparse
import base64
import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Setup path so config.py is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

load_dotenv(cfg.PROJECT_ROOT / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOGS_DIR / "1_ocr_extractor.log", mode="a"),
    ],
)
logger = logging.getLogger("ocr_extractor")

# ─── Checkpoint ───────────────────────────────────────────────────────────────
cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = cfg.CHECKPOINTS_DIR / "ocr_extractor.json"

def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"processed": []}

def save_checkpoint(data: Dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))

# ─── Image Utilities ──────────────────────────────────────────────────────────

def collect_source_images(src: Dict) -> List[Path]:
    """Collect image paths from a source, sorted deterministically."""
    src_path: Path = src["path"]
    if not src_path.exists():
        logger.warning(f"[{src['name']}] Source path not found: {src_path}")
        return []

    images: List[Path] = []
    for ext_pattern in src["exts"]:
        images.extend(src_path.glob(ext_pattern))

    # Deduplicate (different patterns can match same file on case-insensitive FS)
    seen = set()
    unique: List[Path] = []
    for p in sorted(images):
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            unique.append(p)

    unique.sort()  # Deterministic order → consistent train/test split

    limit = src.get("limit")
    if limit:
        unique = unique[:limit]

    logger.info(f"[{src['name']}] Found {len(unique)} images (limit={limit})")
    return unique


def split_train_test(images: List[Path], test_ratio: float) -> Tuple[List[Path], List[Path]]:
    """Deterministic 95/5 split. Test = last N files (sorted order)."""
    n_test = max(1, math.ceil(len(images) * test_ratio))
    train = images[:-n_test]
    test  = images[-n_test:]
    return train, test


def output_stem(prefix: str, src_path: Path) -> str:
    """Build unique output filename stem: {prefix}_{original_stem}"""
    return f"{prefix}_{src_path.stem}"


def load_and_convert_image(img_path: Path) -> Optional[bytes]:
    """
    Load image, convert to RGB JPEG bytes.
    Returns None if image is corrupt/unreadable.
    """
    try:
        # Handle HEIF/HEIC formats
        if img_path.suffix.lower() in (".heic", ".heif"):
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                logger.warning(f"pillow-heif not installed, skipping {img_path.name}")
                return None

        with Image.open(img_path) as im:
            im.verify()  # Check for corruption

        with Image.open(img_path) as im:
            # Convert to RGB (handles RGBA, palette, etc.)
            rgb = im.convert("RGB")
            # Cap resolution at 4096px longest side to avoid GPU OOM
            max_px = 4096
            if max(rgb.size) > max_px:
                ratio = max_px / max(rgb.size)
                new_size = (int(rgb.width * ratio), int(rgb.height * ratio))
                rgb = rgb.resize(new_size, Image.LANCZOS)

            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=92)
            return buf.getvalue()

    except (UnidentifiedImageError, Exception) as e:
        logger.warning(f"Corrupt/unreadable image, skipping: {img_path.name} — {e}")
        return None


def encode_image_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("utf-8")

# ─── GLM-OCR Call ─────────────────────────────────────────────────────────────

def run_glm_ocr(client: OpenAI, img_b64: str, stem: str) -> Optional[str]:
    """
    Call GLM-OCR via vLLM OpenAI-compatible API.
    Returns raw OCR text or None on failure.

    Uses base64 data URI — works reliably regardless of vLLM local file path config.
    Prompt: "Text Recognition:" — GLM-OCR predefined task for raw text extraction.
    """
    MAX_RETRIES = 3
    BACKOFF = [5, 15, 30]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=cfg.GLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": cfg.GLM_OCR_PROMPT
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            logger.warning(f"[{stem}] GLM-OCR attempt {attempt+1}/3 failed: {e}. Retry in {wait}s")
            time.sleep(wait)

    logger.error(f"[{stem}] GLM-OCR failed after {MAX_RETRIES} attempts")
    return None

# ─── Per-Image Worker ─────────────────────────────────────────────────────────

def process_image(
    client: OpenAI,
    src_path: Path,
    prefix: str,
    split: str,
) -> Optional[str]:
    """
    Full pipeline for one image:
      1. Load + convert to JPEG
      2. Copy to images/
      3. Run GLM-OCR
      4. Save .txt to ocr/
    Returns the output stem (e.g., "cord_00001") on success, None on failure.
    """
    stem = output_stem(prefix, src_path)

    # Destination paths
    img_dst  = cfg.images_dir(split) / f"{stem}.jpg"
    txt_dst  = cfg.ocr_dir(split)    / f"{stem}.txt"

    # Skip if already processed (both files exist)
    if img_dst.exists() and txt_dst.exists():
        logger.debug(f"[{stem}] Already done, skipping")
        return stem

    # Step 1: Load + convert image
    jpeg_bytes = load_and_convert_image(src_path)
    if jpeg_bytes is None:
        return None

    # Step 2: Copy image to destination
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    img_dst.write_bytes(jpeg_bytes)

    # Step 3: Run GLM-OCR
    img_b64  = encode_image_b64(jpeg_bytes)
    ocr_text = run_glm_ocr(client, img_b64, stem)
    if ocr_text is None:
        img_dst.unlink(missing_ok=True)  # Clean up on failure
        return None

    # Step 4: Save OCR text
    txt_dst.parent.mkdir(parents=True, exist_ok=True)
    txt_dst.write_text(ocr_text, encoding="utf-8")

    logger.info(f"[{stem}] ✓ {split} | {len(ocr_text)} chars OCR")
    return stem

# ─── Oxen Versioning ──────────────────────────────────────────────────────────

def oxen_push(commit_msg: str) -> None:
    """Push recipe_db data to Oxen remote. Gracefully skips if oxen not configured."""
    try:
        import oxen
        from oxen.auth import config_auth

        if cfg.OXEN_AUTH_TOKEN:
            config_auth(cfg.OXEN_AUTH_TOKEN)

        repo = oxen.Repo(str(cfg.PROJECT_ROOT))
        # Stage only the recipe_db directory (not the whole project)
        repo.add(str(cfg.RECIPE_DB_DIR))
        repo.commit(commit_msg)
        repo.push()
        logger.info(f"Oxen push: '{commit_msg}'")

    except Exception as e:
        logger.warning(f"Oxen push skipped (non-fatal): {e}")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def build_work_plan(mode: str, source_filter: Optional[str]) -> List[Tuple[Path, str, str]]:
    """
    Returns list of (src_path, prefix, split) for all images to process.
    """
    ckpt = load_checkpoint()
    processed_set = set(ckpt["processed"])

    plan: List[Tuple[Path, str, str]] = []  # (src_path, prefix, split)

    configs = cfg.SOURCE_CONFIGS
    if source_filter:
        configs = [c for c in configs if c["name"] == source_filter]
        if not configs:
            logger.error(f"Unknown source filter: {source_filter}")
            sys.exit(1)

    for src in configs:
        images = collect_source_images(src)
        if not images:
            continue

        train_imgs, test_imgs = split_train_test(images, src["test_ratio"])
        logger.info(
            f"[{src['name']}] Split: {len(train_imgs)} train / {len(test_imgs)} test"
        )

        # Create all output directories
        for split in ("train", "test"):
            cfg.images_dir(split).mkdir(parents=True, exist_ok=True)
            cfg.ocr_dir(split).mkdir(parents=True, exist_ok=True)
            cfg.labels_dir(split).mkdir(parents=True, exist_ok=True)

        for img_path in train_imgs:
            stem = output_stem(src["prefix"], img_path)
            if stem not in processed_set:
                plan.append((img_path, src["prefix"], "train"))

        for img_path in test_imgs:
            stem = output_stem(src["prefix"], img_path)
            if stem not in processed_set:
                plan.append((img_path, src["prefix"], "test"))

    if mode == "validate":
        plan = plan[:3]
        logger.info(f"VALIDATE mode: processing first {len(plan)} images only")

    logger.info(f"Work plan: {len(plan)} images to process")
    return plan, ckpt


def run_pipeline(mode: str, source_filter: Optional[str]) -> None:
    """Main pipeline orchestrator."""

    logger.info("=" * 65)
    logger.info(f"  RECIPE-DB OCR Extractor — Mode: {mode.upper()}")
    logger.info("=" * 65)
    logger.info(f"  GLM endpoint   : {cfg.GLM_ENDPOINT}")
    logger.info(f"  GLM model      : {cfg.GLM_MODEL}")
    logger.info(f"  Max workers    : {cfg.GLM_MAX_WORKERS}")
    logger.info(f"  Output dir     : {cfg.RECIPE_DB_DIR}")
    logger.info("=" * 65)

    # Validate GLM-OCR server is reachable
    try:
        glm_client = OpenAI(api_key="EMPTY", base_url=cfg.GLM_ENDPOINT, timeout=300)
        models = glm_client.models.list()
        logger.info(f"GLM-OCR server OK — models: {[m.id for m in models.data]}")
    except Exception as e:
        logger.error(f"Cannot reach GLM-OCR server at {cfg.GLM_ENDPOINT}: {e}")
        logger.error("Start the server: vllm serve zai-org/GLM-OCR --port 8000 ...")
        sys.exit(1)

    plan, ckpt = build_work_plan(mode, source_filter)
    if not plan:
        logger.info("Nothing to process — all images already done.")
        return

    success_count = 0
    fail_count    = 0
    processed_this_run: List[str] = []

    CHECKPOINT_EVERY = 50

    with ThreadPoolExecutor(max_workers=cfg.GLM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_image, glm_client, src_path, prefix, split): (src_path, prefix, split)
            for src_path, prefix, split in plan
        }

        for i, future in enumerate(as_completed(futures), 1):
            src_path, prefix, split = futures[future]
            stem = output_stem(prefix, src_path)

            try:
                result = future.result()
            except Exception as e:
                logger.error(f"[{stem}] Unexpected error: {e}")
                result = None

            if result:
                success_count += 1
                ckpt["processed"].append(result)
                processed_this_run.append(result)
            else:
                fail_count += 1

            # Checkpoint periodically
            if i % CHECKPOINT_EVERY == 0:
                save_checkpoint(ckpt)
                logger.info(
                    f"  ── Checkpoint [{i}/{len(plan)}] "
                    f"success={success_count} fail={fail_count}"
                )

    # Final checkpoint
    save_checkpoint(ckpt)

    # Validate mode: print sample outputs
    if mode == "validate":
        logger.info(f"\n{'=' * 65}")
        logger.info("  VALIDATE OUTPUT — Sample OCR Results")
        logger.info(f"{'=' * 65}")
        for stem in processed_this_run[:3]:
            # Find the txt file
            for split in ("train", "test"):
                txt = cfg.ocr_dir(split) / f"{stem}.txt"
                if txt.exists():
                    text = txt.read_text(encoding="utf-8")
                    logger.info(f"\n── [{split}] {stem}")
                    logger.info(f"   First 300 chars: {text[:300]}")
                    break

    # Summary
    logger.info(f"\n{'=' * 65}")
    logger.info("  OCR EXTRACTION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Processed : {success_count}")
    logger.info(f"  Failed    : {fail_count}")
    logger.info(f"  Total done: {len(ckpt['processed'])}")
    logger.info("=" * 65)

    # Oxen push (run mode only)
    if mode == "run" and success_count > 0:
        oxen_push(
            f"feat(recipe-db): add OCR results — {success_count} images "
            f"({len(ckpt['processed'])} total)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Extract OCR text from receipt images using GLM-OCR via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 3 images first:
  python 1_ocr_extractor.py --mode validate

  # Full run (checkpoint-resumable, safe to interrupt):
  python 1_ocr_extractor.py --mode run

  # Single source only:
  python 1_ocr_extractor.py --mode run --source cord

Prerequisites:
  vllm serve zai-org/GLM-OCR --port 8000 \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \\
    --allowed-local-media-path /
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "run"],
        default="validate",
        help="validate=3 images only | run=full pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter to a single source name (e.g., cord, pint, sroe)",
    )
    args = parser.parse_args()
    run_pipeline(args.mode, args.source)


if __name__ == "__main__":
    main()