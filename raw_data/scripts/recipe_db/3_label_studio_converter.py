#!/usr/bin/env python3
"""
3_label_studio_converter.py
============================
Step 3 of the RECIPE-DB pipeline.

Purpose: Manual verification of Gemini KIE extraction results before final formatting.

What it does:
  EXPORT mode (before review):
    1. Reads labels/*.json + images/*.jpg per split
    2. Generates Label Studio tasks.json (importable into Label Studio)
       - Images referenced via local file serving (relative to PROJECT_ROOT)
       - Tasks exported WITHOUT pre-filled annotations so they appear in labeling queue
       - JSON pre-populated via predictions (editable by reviewer)
    3. Generates labeling_config.xml (paste into Label Studio project)
    4. Generates VERIFY_REPORT.md summarizing what needs checking

  IMPORT mode (after review):
    1. Reads Label Studio export JSON
    2. Extracts corrected annotations
    3. Overwrites labels/*.json with verified/corrected versions
    4. Reports what was changed

Label Studio setup (one-time):
  pip install label-studio

  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/fine-tuning-glm-ocr \\
  label-studio start --port 8081

  Then in Label Studio:
    1. Create new project
    2. Settings → Cloud Storage → Add Source Storage → Local Files
       Absolute local path: /path/to/fine-tuning-glm-ocr
    3. Settings → Labeling Interface → paste labeling_config.xml content
    4. Import → upload tasks.json
    5. Review each task (image + OCR + JSON)
    6. Export → JSON → save as label_studio_export.json

Usage:
  # Export tasks for Label Studio
  python 3_label_studio_converter.py --mode export

  # Export specific split only
  python 3_label_studio_converter.py --mode export --split train

  # After reviewing in Label Studio, import corrections:
  python 3_label_studio_converter.py --mode import --file label_studio_export.json

  # Generate verification report only (no Label Studio needed):
  python 3_label_studio_converter.py --mode report
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ─── Logging ──────────────────────────────────────────────────────────────────
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOGS_DIR / "3_label_studio_converter.log", mode="a"),
    ],
)
logger = logging.getLogger("ls_converter")

# ─── Output Paths ─────────────────────────────────────────────────────────────
LS_DIR = cfg.PROJECT_ROOT / "label_studio"
LS_DIR.mkdir(parents=True, exist_ok=True)

TASKS_FILE          = LS_DIR / "tasks.json"
LABELING_CONFIG_XML = LS_DIR / "labeling_config.xml"
VERIFY_REPORT_MD    = LS_DIR / "VERIFY_REPORT.md"

# ─── Label Studio Config XML ──────────────────────────────────────────────────
# Layout: image on top, OCR text and JSON editor side by side below.
# User edits the JSON textarea to correct extraction errors.
#
# NOTE: OCR text is pre-populated via $ocr_text in data (read-only display).
#       JSON is pre-populated via predictions (editable by reviewer).
LABELING_CONFIG = """<View>
  <Style>
    .main { display: flex; flex-direction: column; gap: 12px; }
    .row  { display: flex; gap: 12px; }
    .col  { flex: 1; }
    label { font-weight: bold; font-size: 13px; color: #555; }
    .meta { font-size: 12px; color: #888; padding: 4px 0; }
  </Style>

  <View className="main">

    <!-- Header: filename and source info -->
    <View>
      <Text name="meta_info" value="$meta_info" />
    </View>

    <!-- Receipt image -->
    <View>
      <Header value="Receipt Image" />
      <Image name="image" value="$image" maxWidth="700px" />
    </View>

    <!-- OCR text + JSON editor side by side -->
    <View className="row">

      <!-- Left: raw OCR text (read-only reference) -->
      <View className="col">
        <Header value="OCR Text (GLM-OCR output — read only)" />
        <Text name="ocr_display" value="$ocr_text" />
      </View>

      <!-- Right: editable KIE JSON -->
      <View className="col">
        <Header value="Extracted JSON (edit to correct)" />
        <TextArea
          name="label_json"
          toName="image"
          placeholder="Extracted JSON"
          rows="30"
          maxSubmissions="1"
          editable="true"
          perRegion="false"
        />
      </View>

    </View>

    <!-- Review verdict -->
    <View>
      <Header value="Review verdict" />
      <Choices name="verdict" toName="image" choice="single-radio">
        <Choice value="correct"     alias="correct"     />
        <Choice value="corrected"   alias="corrected"   />
        <Choice value="skip"        alias="skip"        />
      </Choices>
    </View>

  </View>
</View>"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_pairs(split: str) -> List[Tuple[str, Path, Path, Path]]:
    """
    Return list of (stem, image_path, ocr_path, label_path) for a split.
    Only returns pairs where BOTH image and label exist.
    OCR text is optional (included if exists).
    """
    pairs = []
    lbl_dir = cfg.labels_dir(split)
    img_dir = cfg.images_dir(split)
    ocr_dir_ = cfg.ocr_dir(split)

    if not lbl_dir.exists():
        logger.warning(f"Labels dir not found: {lbl_dir} — run step 2 first")
        return pairs

    for lbl_path in sorted(lbl_dir.glob("*.json")):
        stem      = lbl_path.stem
        img_path  = img_dir  / f"{stem}.jpg"
        ocr_path  = ocr_dir_ / f"{stem}.txt"

        if not img_path.exists():
            logger.warning(f"[{stem}] Image not found, skipping")
            continue

        pairs.append((stem, img_path, ocr_path, lbl_path))

    return pairs


def label_to_ls_task(
    task_id: int,
    stem: str,
    split: str,
    img_path: Path,
    ocr_path: Path,
    lbl_path: Path,
) -> Dict:
    """
    Build one Label Studio task dict.

    Key design decisions:
      - Image served via Label Studio local file serving
        Path is RELATIVE to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
        e.g. if DOCUMENT_ROOT=/teamspace/studios/this_studio/fine-tuning-glm-ocr
        then path = "LLaMA-Factory/data/recipe_db/train/images/cord_GAMBAR_100.jpg"
      - NO pre-filled annotations → task appears in labeling queue as "unlabeled"
      - JSON pre-populated via `predictions` (editable, auto-loaded in UI)
      - OCR text in `data.ocr_text` for the read-only Text display
    """
    # Local file serving: path relative to PROJECT_ROOT (= DOCUMENT_ROOT)
    try:
        rel_path = img_path.relative_to(cfg.PROJECT_ROOT)
    except ValueError:
        # Fallback: use absolute path (less reliable)
        rel_path = img_path
        logger.warning(f"[{stem}] Image not under PROJECT_ROOT, using absolute path")

    image_url = f"/data/local-files/?d={rel_path.as_posix()}"

    # OCR text
    ocr_text = ""
    if ocr_path.exists():
        ocr_text = ocr_path.read_text(encoding="utf-8").strip()

    # Label JSON
    label_data = {}
    try:
        label_data = json.loads(lbl_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[{stem}] Cannot read label JSON: {e}")

    label_json_str = json.dumps(label_data, ensure_ascii=False, indent=2)

    # Parse prefix from stem to identify source dataset
    source = stem.split("_")[0] if "_" in stem else "unknown"

    return {
        "id": task_id,
        "data": {
            "image":     image_url,
            "meta_info": f"Split: {split} | Source: {source} | File: {stem}",
            "ocr_text":  ocr_text,
            "stem":      stem,
            "split":     split,
        },
        # NO "annotations" key → Label Studio treats this as unlabeled
        # Pre-populate JSON via predictions so reviewer sees it pre-filled
        "predictions": [
            {
                "result": [
                    {
                        "id":        f"json_{task_id}",
                        "type":      "textarea",
                        "from_name": "label_json",
                        "to_name":   "image",
                        "value":     {"text": [label_json_str]}
                    },
                    {
                        "id":        f"verdict_{task_id}",
                        "type":      "choices",
                        "from_name": "verdict",
                        "to_name":   "image",
                        "value":     {"choices": ["correct"]}
                    },
                ]
            }
        ],
    }

# ─── Schema Validation ────────────────────────────────────────────────────────

def _check_schema_keys(data: Dict, schema: Dict, path: str = "") -> List[str]:
    """Recursively check that all schema keys are present in data."""
    errors = []
    for key, val in schema.items():
        full_path = f"{path}.{key}" if path else key
        if key not in data:
            errors.append(f"Missing key: {full_path}")
        elif isinstance(val, dict) and isinstance(data.get(key), dict):
            errors.extend(_check_schema_keys(data[key], val, full_path))
    return errors


def validate_label(label: Dict) -> List[str]:
    """Return list of structural issues found in a label dict."""
    issues = []

    # Check top-level keys
    for key in ["info", "items", "returned_items", "payment"]:
        if key not in label:
            issues.append(f"Missing top-level key: {key}")

    # Check info subkeys
    schema_info = cfg.EXTRACTION_SCHEMA.get("info", {})
    for k in schema_info:
        if k not in label.get("info", {}):
            issues.append(f"Missing info.{k}")

    # Check payment subkeys
    schema_pay = cfg.EXTRACTION_SCHEMA.get("payment", {})
    for k in schema_pay:
        if k not in label.get("payment", {}):
            issues.append(f"Missing payment.{k}")

    # Check array types
    for field in ["items", "returned_items"]:
        if not isinstance(label.get(field), list):
            issues.append(f"Field '{field}' is not a list")

    for field in ["store_contacts", ]:
        if not isinstance(label.get("info", {}).get(field), list):
            issues.append(f"Field 'info.{field}' is not a list")

    for field in ["discounts", "taxes", "additional_charges"]:
        if not isinstance(label.get("payment", {}).get(field), list):
            issues.append(f"Field 'payment.{field}' is not a list")

    return issues

# ─── Export Mode ──────────────────────────────────────────────────────────────

def run_export(split_filter: Optional[str]) -> None:
    """Export all labels as Label Studio tasks.json."""
    splits = ["train", "test"]
    if split_filter:
        splits = [split_filter]

    all_tasks = []
    task_id   = 1
    stats     = {"total": 0, "with_issues": 0, "missing_ocr": 0}
    issues_log: List[Dict] = []

    for split in splits:
        pairs = collect_pairs(split)
        logger.info(f"[{split}] {len(pairs)} label-image pairs found")

        for stem, img_path, ocr_path, lbl_path in pairs:
            task = label_to_ls_task(task_id, stem, split, img_path, ocr_path, lbl_path)
            all_tasks.append(task)

            stats["total"] += 1
            if not ocr_path.exists():
                stats["missing_ocr"] += 1

            # Validate schema
            try:
                label = json.loads(lbl_path.read_text(encoding="utf-8"))
                issues = validate_label(label)
                if issues:
                    stats["with_issues"] += 1
                    issues_log.append({"stem": stem, "split": split, "issues": issues})
            except Exception as e:
                stats["with_issues"] += 1
                issues_log.append({"stem": stem, "split": split, "issues": [f"Parse error: {e}"]})

            task_id += 1

    # Save tasks.json
    TASKS_FILE.write_text(json.dumps(all_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"\nExported {len(all_tasks)} tasks → {TASKS_FILE}")

    # Save labeling_config.xml
    LABELING_CONFIG_XML.write_text(LABELING_CONFIG, encoding="utf-8")
    logger.info(f"Labeling config  → {LABELING_CONFIG_XML}")

    # Generate VERIFY_REPORT.md
    _generate_report(stats, issues_log)

    # Print setup instructions
    print(f"""
{'='*65}
  Label Studio Export Complete
{'='*65}
  Tasks exported : {stats['total']}
  Schema issues  : {stats['with_issues']}
  Missing OCR    : {stats['missing_ocr']}

  Files:
    {TASKS_FILE}
    {LABELING_CONFIG_XML}
    {VERIFY_REPORT_MD}

  Setup Label Studio:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={cfg.PROJECT_ROOT} \\
    label-studio start --port 8081

    NOTE: DOCUMENT_ROOT must be set to PROJECT_ROOT:
      {cfg.PROJECT_ROOT}

    Image paths in tasks.json are RELATIVE to this root, e.g.:
      /data/local-files/?d=LLaMA-Factory/data/recipe_db/train/images/xxx.jpg

  In Label Studio browser:
    1. Create new project → give it a name
    2. Settings → Cloud Storage → Add Source Storage → Local Files
       Absolute local path: {cfg.PROJECT_ROOT}
    3. Settings → Labeling Interface → Code
       Paste content of: {LABELING_CONFIG_XML}
    4. Import → Upload Files → select: {TASKS_FILE}
    5. Click "Label All Tasks" to start reviewing
    6. Each task shows: image + OCR text + pre-filled JSON
    7. Edit JSON if needed, set verdict, click Submit
    8. Export → JSON-MIN → save as label_studio_export.json

  After review, import corrections:
    python 3_label_studio_converter.py --mode import \\
      --file /path/to/label_studio_export.json
{'='*65}
""")


def _generate_report(stats: Dict, issues_log: List[Dict]) -> None:
    """Write VERIFY_REPORT.md summarizing what needs attention."""
    lines = [
        "# RECIPE-DB KIE Verification Report",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total labels | {stats['total']} |",
        f"| Labels with schema issues | {stats['with_issues']} |",
        f"| Missing OCR text | {stats['missing_ocr']} |",
        "",
        "## Schema Issues (fix priority)",
        "",
    ]

    if not issues_log:
        lines.append("✅ No schema issues found.")
    else:
        lines.append(f"Found {len(issues_log)} files with issues:\n")
        for item in issues_log[:50]:   # Show first 50
            lines.append(f"### `{item['split']}/{item['stem']}`")
            for issue in item["issues"]:
                lines.append(f"- {issue}")
            lines.append("")
        if len(issues_log) > 50:
            lines.append(f"... and {len(issues_log) - 50} more. See log file.")

    lines += [
        "",
        "## Review Checklist",
        "",
        "For each receipt in Label Studio, verify:",
        "- [ ] `store_name` matches what's visually prominent at the top",
        "- [ ] `store_location` is complete (not truncated by OCR)",
        "- [ ] `store_contacts.type` is copied as-is (not classified)",
        "- [ ] `items` have correct `item_name` (no SKU codes mixed in)",
        "- [ ] `unit_price` does not include currency symbol",
        "- [ ] `grand_total` matches the final TOTAL line",
        "- [ ] `payment_method` is copied exactly (not translated)",
        "- [ ] `currency` is empty if not printed on receipt",
        "",
        "## Common Errors from Gemini",
        "",
        "| Error pattern | How to fix |",
        "|---------------|------------|",
        "| Currency symbol in price field | Remove symbol, keep number only |",
        "| store_contacts.type guessed (e.g., 'TEL' from unlabelled number) | Set type to '' |",
        "| item_name includes barcode/SKU | Remove code, keep description only |",
        "| total_discount calculated (summed) | Use only explicitly written value |",
        "| payment_method translated (e.g., 'Cash' from 'Tunai') | Restore original |",
    ]

    VERIFY_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Verify report    → {VERIFY_REPORT_MD}")

# ─── Import Mode ──────────────────────────────────────────────────────────────

def run_import(export_file: str) -> None:
    """
    Import corrected annotations from Label Studio export JSON.
    Overwrites labels/*.json with corrected versions.
    """
    export_path = Path(export_file)
    if not export_path.exists():
        logger.error(f"Export file not found: {export_path}")
        sys.exit(1)

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    logger.info(f"Loading {len(tasks)} tasks from {export_path}")

    updated   = 0
    skipped   = 0
    errors    = 0
    changed   = 0

    for task in tasks:
        data  = task.get("data", {})
        stem  = data.get("stem", "")
        split = data.get("split", "")

        if not stem or not split:
            logger.warning(f"Task missing stem/split metadata, skipping: {task.get('id')}")
            skipped += 1
            continue

        # Find the annotation with label_json result
        annotations = task.get("annotations", [])
        if not annotations:
            logger.warning(f"[{stem}] No annotation, skipping")
            skipped += 1
            continue

        # Use the last annotation (most recent review)
        ann = annotations[-1]
        results = ann.get("result", [])

        # Check verdict
        verdict = ""
        json_text = ""
        for r in results:
            if r.get("from_name") == "verdict":
                choices = r.get("value", {}).get("choices", [])
                verdict = choices[0] if choices else ""
            if r.get("from_name") == "label_json":
                texts = r.get("value", {}).get("text", [])
                json_text = texts[0] if texts else ""

        if verdict == "skip":
            logger.info(f"[{stem}] Verdict: skip — not imported")
            skipped += 1
            continue

        if not json_text.strip():
            logger.warning(f"[{stem}] Empty JSON annotation, skipping")
            skipped += 1
            continue

        # Parse the corrected JSON
        try:
            corrected = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"[{stem}] Invalid JSON in annotation: {e}")
            errors += 1
            continue

        # Validate schema
        issues = validate_label(corrected)
        if issues:
            logger.warning(f"[{stem}] Schema issues in corrected annotation: {issues}")
            # Don't block import — user may have intentionally simplified

        # Compare with current label
        lbl_path = cfg.labels_dir(split) / f"{stem}.json"
        old_json = ""
        if lbl_path.exists():
            old_json = lbl_path.read_text(encoding="utf-8").strip()

        new_json = json.dumps(corrected, ensure_ascii=False, indent=2)

        if old_json != new_json:
            changed += 1
            logger.info(f"[{stem}] Updated (verdict={verdict})")
        else:
            logger.debug(f"[{stem}] Unchanged")

        # Write corrected label
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text(new_json, encoding="utf-8")
        updated += 1

    logger.info(f"""
{'='*55}
  Label Studio Import Complete
{'='*55}
  Imported  : {updated}
  Changed   : {changed}
  Skipped   : {skipped}
  Errors    : {errors}
{'='*55}
""")

    if errors > 0:
        logger.warning(f"{errors} tasks had invalid JSON — check log for details")

# ─── Report Mode ──────────────────────────────────────────────────────────────

def run_report() -> None:
    """Quick report without generating Label Studio tasks."""
    stats = {"total": 0, "with_issues": 0, "missing_ocr": 0}
    issues_log = []

    for split in ["train", "test"]:
        pairs = collect_pairs(split)
        for stem, img_path, ocr_path, lbl_path in pairs:
            stats["total"] += 1
            if not ocr_path.exists():
                stats["missing_ocr"] += 1
            try:
                label  = json.loads(lbl_path.read_text(encoding="utf-8"))
                issues = validate_label(label)
                if issues:
                    stats["with_issues"] += 1
                    issues_log.append({"stem": stem, "split": split, "issues": issues})
            except Exception as e:
                stats["with_issues"] += 1
                issues_log.append({"stem": stem, "split": split, "issues": [str(e)]})

    _generate_report(stats, issues_log)
    print(f"\nReport written to: {VERIFY_REPORT_MD}")
    print(f"Total: {stats['total']} | Issues: {stats['with_issues']} | Missing OCR: {stats['missing_ocr']}")

# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Label Studio converter for manual KIE verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Run step 2 to generate labels/*.json
  2. Export to Label Studio:
       python 3_label_studio_converter.py --mode export
  3. Start Label Studio and review/correct tasks
  4. Export from Label Studio as JSON
  5. Import corrections:
       python 3_label_studio_converter.py --mode import --file export.json
  6. Run step 4 (final formatter)

Examples:
  python 3_label_studio_converter.py --mode export
  python 3_label_studio_converter.py --mode export --split train
  python 3_label_studio_converter.py --mode import --file label_studio_export.json
  python 3_label_studio_converter.py --mode report
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["export", "import", "report"],
        required=True,
        help="export: generate Label Studio tasks | import: apply corrections | report: schema check only",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default=None,
        help="Export one split only (default: both)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to Label Studio export JSON (required for --mode import)",
    )
    args = parser.parse_args()

    if args.mode == "export":
        run_export(args.split)
    elif args.mode == "import":
        if not args.file:
            parser.error("--file is required for --mode import")
        run_import(args.file)
    elif args.mode == "report":
        run_report()


if __name__ == "__main__":
    main()