#!/usr/bin/env python3
"""
_fix_annotated.py — RECIPE-DB Annotation Fix Tool
==================================================

Interactive CLI to search and fix annotation inconsistencies in Label Studio
exported labels before running 4_final_formatter.py.

Source (default) : fine-tuning-glm-ocr/label_studio/batch/818-885-fixed.json
Output           : fine-tuning-glm-ocr/label_studio/fix-annotated/<same-filename>.json

Two search modes:
  1. Value contains  — find fields whose value contains a substring
                       e.g. field=payment.currency  contains="."
                       matches: "Rp.", "$.", "€.", etc.

  2. Field length    — find fields where len(value) matches a threshold
                       e.g. field=info.receipt_id  operator=>  length=10
                       matches any receipt_id longer than 10 chars

After searching, edit by picking a row number from the results table,
or jump directly to a stem if you already know what to fix.

After all edits, save → re-run the import:
  uv run raw_data/scripts/recipe_db/3_label_studio_converter.py \\
    --mode import \\
    --file label_studio/fix-annotated/818-885-fixed.json

Usage:
  python _fix_annotated.py
  python _fix_annotated.py --file /path/to/other-export.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Path Resolution ──────────────────────────────────────────────────────────
# _fix_annotated.py lives at: fine-tuning-glm-ocr/raw_data/scripts/recipe_db/
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent          # fine-tuning-glm-ocr/
DEFAULT_FILE = PROJECT_ROOT / "label_studio" / "batch" / "818-885-fixed.json"
OUTPUT_DIR   = PROJECT_ROOT / "label_studio" / "fix-annotated"


# ─── SearchResult ─────────────────────────────────────────────────────────────
class SearchResult:
    """One row in a search result table."""
    __slots__ = ("row", "task_idx", "stem", "split", "display_path", "value")

    def __init__(
        self,
        row: int,
        task_idx: int,
        stem: str,
        split: str,
        display_path: str,
        value: str,
    ) -> None:
        self.row          = row
        self.task_idx     = task_idx
        self.stem         = stem
        self.split        = split
        self.display_path = display_path
        self.value        = value


# ─── File I/O ─────────────────────────────────────────────────────────────────

def load_file(filepath: Path) -> List[Dict]:
    """Load and validate a Label Studio JSON export file."""
    try:
        raw = filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"\n  [error] File not found: {filepath}")
        sys.exit(1)
    except OSError as e:
        print(f"\n  [error] Cannot read file: {e}")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n  [error] Invalid JSON in {filepath.name}: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"\n  [error] Expected JSON array, got {type(data).__name__}")
        sys.exit(1)

    return data


def save_output(tasks: List[Dict], source_file: Path) -> Path:
    """Write modified tasks to fix-annotated/<filename>."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / source_file.name
    out_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ─── Task Parsing — handles all Label Studio export formats ───────────────────

def _extract_json_text(task: Dict) -> str:
    """
    Pull the label JSON string out of a task regardless of LS export format.

    Format 1 — JSON-MIN flat (label_json is a top-level key):
      {"id": 1, "stem": "...", "label_json": "{...}", "verdict": "correct", ...}

    Format 2 — Nested annotations (annotated JSON export):
      {"id": 1, "data": {...}, "annotations": [{"result": [...]}]}

    Format 3 — Predictions only (original tasks.json before annotation):
      {"id": 1, "data": {...}, "predictions": [{"result": [...]}]}
    """
    # Format 1: flat JSON-MIN
    if "label_json" in task:
        return str(task["label_json"])

    # Format 2: nested annotations
    if "annotations" in task:
        anns = task.get("annotations") or []
        if anns:
            for r in anns[-1].get("result", []):
                if r.get("from_name") == "label_json":
                    texts = r.get("value", {}).get("text", [])
                    return texts[0] if texts else ""

    # Format 3: predictions (tasks.json before human annotation)
    if "predictions" in task:
        preds = task.get("predictions") or []
        if preds:
            for r in preds[0].get("result", []):
                if r.get("from_name") == "label_json":
                    texts = r.get("value", {}).get("text", [])
                    return texts[0] if texts else ""

    return ""


def parse_all_tasks(
    tasks: List[Dict],
) -> List[Tuple[str, str, Optional[Dict]]]:
    """
    Parse every task into (stem, split, label_dict).
    label_dict is None for tasks with no parseable JSON.
    """
    parsed: List[Tuple[str, str, Optional[Dict]]] = []

    for task in tasks:
        # stem/split: try nested data dict first, then flat (JSON-MIN)
        data  = task.get("data", task)
        stem  = str(data.get("stem",  task.get("stem",  ""))).strip()
        split = str(data.get("split", task.get("split", ""))).strip()

        json_text = _extract_json_text(task).strip()
        if json_text:
            try:
                label = json.loads(json_text)
                parsed.append((stem, split, label))
                continue
            except json.JSONDecodeError:
                pass

        parsed.append((stem, split, None))

    return parsed


def write_label_back(task: Dict, new_label: Dict) -> None:
    """
    Serialize new_label back into the task dict in-place.
    Handles all three export formats.
    """
    new_str = json.dumps(new_label, ensure_ascii=False, separators=(",", ":"))

    if "label_json" in task:
        task["label_json"] = new_str
        return

    if "annotations" in task:
        anns = task.get("annotations") or []
        if anns:
            for r in anns[-1].get("result", []):
                if r.get("from_name") == "label_json":
                    r["value"]["text"] = [new_str]
        return

    if "predictions" in task:
        preds = task.get("predictions") or []
        if preds:
            for r in preds[0].get("result", []):
                if r.get("from_name") == "label_json":
                    r["value"]["text"] = [new_str]


# ─── Field Navigation ─────────────────────────────────────────────────────────

def get_field_values(label: Dict, field_path: str) -> List[Tuple[str, str]]:
    """
    Navigate label using dot-notation path. Returns (display_path, value) pairs.
    For array fields, returns one entry per array element with an index suffix.

    Examples:
      "payment.currency"
        → [("payment.currency", "Rp.")]

      "items.item_name"
        → [("items[0].item_name", "SHOES"), ("items[1].item_name", "BAG"), ...]

      "info.store_contacts.type"
        → [("info.store_contacts[0].type", "Tel."), ...]

      "payment.taxes.tax_name"
        → [("payment.taxes[0].tax_name", "SST 6%"), ...]
    """
    parts = field_path.split(".")

    def nav(obj: Any, parts_left: List[str], prefix: str) -> List[Tuple[str, str]]:
        # Base case: no more parts → yield current value
        if not parts_left:
            if isinstance(obj, str):
                return [(prefix, obj)]
            if obj is None:
                return [(prefix, "")]
            return [(prefix, json.dumps(obj, ensure_ascii=False))]

        key  = parts_left[0]
        rest = parts_left[1:]

        if isinstance(obj, dict):
            if key in obj:
                child = f"{prefix}.{key}" if prefix else key
                return nav(obj[key], rest, child)
            return []

        if isinstance(obj, list):
            results: List[Tuple[str, str]] = []
            for i, item in enumerate(obj):
                item_prefix = f"{prefix}[{i}]"
                if isinstance(item, dict) and key in item:
                    results.extend(nav(item[key], rest, f"{item_prefix}.{key}"))
            return results

        return []

    return nav(label, parts, "")


def set_field_value(label: Dict, display_path: str, new_value: str) -> bool:
    """
    Set the value at display_path (as returned by get_field_values) to new_value.

    Tokenizes paths like:
      "payment.currency"                 → ["payment", "currency"]
      "items[0].item_name"               → ["items", 0, "item_name"]
      "info.store_contacts[1].type"      → ["info", "store_contacts", 1, "type"]
      "payment.discounts[0].amount"      → ["payment", "discounts", 0, "amount"]

    Returns True on success, False on navigation error.
    """
    tokens: List[Any] = []
    for segment in display_path.split("."):
        # "fieldName[N]" → two tokens: fieldName + N(int)
        m = re.fullmatch(r"(\w+)\[(\d+)\]", segment)
        if m:
            tokens.append(m.group(1))
            tokens.append(int(m.group(2)))
        # "[N]" → one int token
        elif re.fullmatch(r"\[(\d+)\]", segment):
            tokens.append(int(re.fullmatch(r"\[(\d+)\]", segment).group(1)))
        # plain key
        else:
            tokens.append(segment)

    if not tokens:
        return False

    try:
        obj = label
        for token in tokens[:-1]:
            obj = obj[token]  # type: ignore[index]
        obj[tokens[-1]] = new_value  # type: ignore[index]
        return True
    except (KeyError, IndexError, TypeError) as e:
        print(f"\n  [error] Cannot navigate to '{display_path}': {e}")
        return False


# ─── Search Functions ─────────────────────────────────────────────────────────

def search_contains(
    parsed: List[Tuple[str, str, Optional[Dict]]],
    field_path: str,
    substring: str,
) -> List[SearchResult]:
    """Find all fields whose value contains the given substring."""
    results: List[SearchResult] = []
    for task_idx, (stem, split, label) in enumerate(parsed):
        if label is None:
            continue
        for display_path, value in get_field_values(label, field_path):
            if substring in value:
                results.append(SearchResult(
                    len(results) + 1, task_idx, stem, split, display_path, value,
                ))
    return results


def search_by_length(
    parsed: List[Tuple[str, str, Optional[Dict]]],
    field_path: str,
    operator: str,     # ">", "<", or "="
    threshold: int,
) -> List[SearchResult]:
    """Find all fields whose value length matches the operator+threshold."""
    results: List[SearchResult] = []
    for task_idx, (stem, split, label) in enumerate(parsed):
        if label is None:
            continue
        for display_path, value in get_field_values(label, field_path):
            vl = len(value)
            match = (
                (operator == ">" and vl > threshold)
                or (operator == "<" and vl < threshold)
                or (operator == "=" and vl == threshold)
            )
            if match:
                results.append(SearchResult(
                    len(results) + 1, task_idx, stem, split, display_path, value,
                ))
    return results


# ─── Display ──────────────────────────────────────────────────────────────────

def _fit(text: str, width: int) -> str:
    """Fit text into a fixed column width, truncating with ellipsis."""
    if len(text) > width:
        return text[: width - 1] + "…"
    return text.ljust(width)


def print_table(results: List[SearchResult]) -> None:
    """Print search results as a unicode box-drawing table."""
    if not results:
        print("\n  (no results found)\n")
        return

    # Dynamic column widths (capped for readability)
    w_row  = max(3, len(str(len(results))))
    w_stem = max(16, min(35, max(len(r.stem)  for r in results)))
    w_sp   = 5
    w_path = max(20, min(50, max(len(r.display_path) for r in results)))
    w_val  = max(5,  min(60, max(len(r.value) for r in results)))
    w_len  = 4
    cols   = [w_row, w_stem, w_sp, w_path, w_val, w_len]

    def divider(l: str, m: str, r: str) -> str:
        return l + m.join("─" * (w + 2) for w in cols) + r

    def row_line(cells: List[str]) -> str:
        return "│ " + " │ ".join(_fit(c, w) for c, w in zip(cells, cols)) + " │"

    print()
    print(divider("┌", "┬", "┐"))
    print(row_line(["#", "Stem", "Split", "Field Path", "Value", "Len"]))
    print(divider("├", "┼", "┤"))
    for r in results:
        print(row_line([
            str(r.row),
            r.stem,
            r.split,
            r.display_path,
            r.value,
            str(len(r.value)),
        ]))
    print(divider("└", "┴", "┘"))
    print(f"  {len(results)} result(s)\n")


# ─── Edit Helpers ─────────────────────────────────────────────────────────────

def _do_single_edit(
    task_idx: int,
    stem: str,
    split: str,
    target_path: str,
    current_val: str,
    tasks: List[Dict],
    parsed: List[Tuple[str, str, Optional[Dict]]],
) -> bool:
    """
    Prompt for a new value and apply it. Returns True if a change was made.
    label dict is modified in-place; task JSON is synced via write_label_back.
    """
    label = parsed[task_idx][2]
    if label is None:
        print("  [error] No label dict for this task.\n")
        return False

    print(f"\n  Stem     : {stem}")
    print(f"  Split    : {split}")
    print(f"  Path     : {target_path}")
    print(f"  Current  : {current_val!r}  (len={len(current_val)})")

    new_val = _prompt("  New value (blank = cancel): ")
    if new_val == "":
        print("  Cancelled.\n")
        return False

    confirm = _prompt(f"  Set to {new_val!r} ? [y/n]: ")
    if confirm.lower() != "y":
        print("  Cancelled.\n")
        return False

    ok = set_field_value(label, target_path, new_val)
    if ok:
        write_label_back(tasks[task_idx], label)
        print(f"  ✓ Updated: {stem} | {target_path} → {new_val!r}\n")
    return ok


def edit_from_results(
    last_results: List[SearchResult],
    tasks: List[Dict],
    parsed: List[Tuple[str, str, Optional[Dict]]],
) -> Tuple[List[SearchResult], int]:
    """
    Let user pick one or more rows from last_results to edit.
    Returns (updated_results, number_of_changes).
    """
    if not last_results:
        print("\n  No search results yet. Run a search first.\n")
        return last_results, 0

    print_table(last_results)
    total_changes = 0

    while True:
        row_input = _prompt(
            f"Row to edit [1-{len(last_results)}, blank=back, 'all'=edit all]: "
        )
        if not row_input:
            break

        # 'all' — iterate through every result
        if row_input.lower() == "all":
            for result in last_results:
                changed = _do_single_edit(
                    result.task_idx,
                    result.stem,
                    result.split,
                    result.display_path,
                    result.value,
                    tasks,
                    parsed,
                )
                if changed:
                    total_changes += 1
                    result.value = parsed[result.task_idx][2]  # type: ignore[index]
                    # re-read updated value
                    vals = get_field_values(parsed[result.task_idx][2], result.display_path.split("[")[0].rsplit(".", 1)[-1] if "[" in result.display_path else result.display_path)
                    # simpler: just leave it, it's already applied
            break

        try:
            row_num = int(row_input)
        except ValueError:
            print("  Invalid input. Enter a row number, 'all', or blank.\n")
            continue

        result = next((r for r in last_results if r.row == row_num), None)
        if result is None:
            print(f"  Row {row_num} not in results.\n")
            continue

        changed = _do_single_edit(
            result.task_idx,
            result.stem,
            result.split,
            result.display_path,
            result.value,
            tasks,
            parsed,
        )
        if changed:
            total_changes += 1
            # Update the in-memory result so re-printing table shows new value
            new_vals = get_field_values(parsed[result.task_idx][2], result.display_path)
            if new_vals:
                result.value = new_vals[0][1]

        again = _prompt("Edit another row from these results? [y/n]: ")
        if again.lower() != "y":
            break

    return last_results, total_changes


def edit_by_stem(
    tasks: List[Dict],
    parsed: List[Tuple[str, str, Optional[Dict]]],
) -> int:
    """
    Let user type a stem and field path directly (no prior search required).
    Returns number of changes made.
    """
    stem_input = _prompt("Stem (e.g. prima_GAMBAR_20): ")
    if not stem_input:
        return 0

    # Find by stem (exact match)
    task_idx = next(
        (i for i, (s, _, _) in enumerate(parsed) if s == stem_input),
        None,
    )
    if task_idx is None:
        # Try prefix match as fallback
        matches = [(i, s) for i, (s, _, _) in enumerate(parsed) if s.startswith(stem_input)]
        if not matches:
            print(f"\n  Stem '{stem_input}' not found.\n")
            return 0
        if len(matches) == 1:
            task_idx, stem_input = matches[0]
            print(f"  Matched: {stem_input}")
        else:
            print(f"\n  Multiple matches:")
            for i, (idx, s) in enumerate(matches[:20], 1):
                print(f"    [{i}] {s}")
            sel = _prompt(f"Select [1-{min(len(matches), 20)}]: ")
            try:
                task_idx, stem_input = matches[int(sel) - 1]
            except (ValueError, IndexError):
                print("  Invalid selection.\n")
                return 0

    stem, split, label = parsed[task_idx]
    if label is None:
        print(f"\n  No label for '{stem}'.\n")
        return 0

    field_path = _prompt("Field path (e.g. payment.currency): ")
    if not field_path:
        return 0

    matches_fv = get_field_values(label, field_path)
    if not matches_fv:
        print(f"\n  No values found at path '{field_path}' in '{stem}'.\n")
        return 0

    # Show current values
    print(f"\n  Stem  : {stem} | Split: {split}")
    for i, (disp_path, val) in enumerate(matches_fv, 1):
        print(f"  [{i}] {disp_path} = {val!r}  (len={len(val)})")

    if len(matches_fv) == 1:
        target_path, current_val = matches_fv[0]
    else:
        idx_str = _prompt(f"Select entry [1-{len(matches_fv)}]: ")
        try:
            target_path, current_val = matches_fv[int(idx_str) - 1]
        except (ValueError, IndexError):
            print("  Invalid selection.\n")
            return 0

    changed = _do_single_edit(
        task_idx, stem, split, target_path, current_val, tasks, parsed
    )
    return 1 if changed else 0


# ─── Field Reference ──────────────────────────────────────────────────────────

FIELD_HELP = """
  Field paths reference (dot-notation):
  ┌─ info ─────────────────────────────────────────────────────────┐
  │  info.store_name         info.store_location                   │
  │  info.tax_id             info.receipt_id                       │
  │  info.payment_date       info.payment_time    info.time_unit   │
  │  info.store_contacts.type                                      │
  │  info.store_contacts.value                                     │
  ├─ items ────────────────────────────────────────────────────────┤
  │  items.item_name         items.quantity       items.unit_price │
  │  items.discount_label    items.discount_price items.tax_label  │
  │  items.total_price                                             │
  ├─ returned_items ───────────────────────────────────────────────┤
  │  returned_items.item_name  returned_items.quantity             │
  │  returned_items.unit_price returned_items.total_refund         │
  ├─ payment ──────────────────────────────────────────────────────┤
  │  payment.total_items     payment.currency    payment.grand_total│
  │  payment.subtotal_price  payment.rounding    payment.change    │
  │  payment.payment_method  payment.tendered                      │
  │  payment.discounts.discount_name                               │
  │  payment.discounts.amount                                      │
  │  payment.taxes.tax_name  payment.taxes.amount                  │
  │  payment.additional_charges.charge_name                        │
  │  payment.additional_charges.amount                             │
  └────────────────────────────────────────────────────────────────┘
  Note: Array paths (items.*, payment.taxes.*, etc.) return one
  row per array element, each with its index: items[0].item_name
"""

SEARCH_EXAMPLES = """
  Examples:
    Search value contains:
      field=payment.currency       contains=.       → finds "Rp.", "$.", "€."
      field=info.store_contacts.type  contains=Tel  → finds "Tel", "Tel.", "TEL"
      field=items.item_name        contains=  (blank) → finds all non-empty item_names

    Search by length:
      field=info.receipt_id        operator=>  length=10  → receipt_id > 10 chars
      field=payment.payment_method operator==  length=0   → empty payment methods
      field=info.store_name        operator=<  length=3   → suspiciously short names
"""


# ─── Prompt Helper ────────────────────────────────────────────────────────────

def _prompt(text: str) -> str:
    try:
        return input(f"  {text}").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")
        return ""


# ─── Main Interactive Loop ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive annotation fix tool for RECIPE-DB pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Label Studio export JSON (default: label_studio/batch/818-885-fixed.json)",
    )
    args = parser.parse_args()
    source_file = args.file.resolve()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RECIPE-DB Annotation Fix Tool")
    print("═" * 65)
    print(f"  Source : {source_file}")
    print(f"  Output : {OUTPUT_DIR}/<same-filename>")
    print()

    tasks  = load_file(source_file)
    parsed = parse_all_tasks(tasks)

    n_valid    = sum(1 for _, _, lbl in parsed if lbl is not None)
    n_no_label = sum(1 for _, _, lbl in parsed if lbl is None)

    print(f"  Loaded : {len(tasks)} tasks")
    print(f"  Parsed : {n_valid} with labels  |  {n_no_label} skipped (no JSON)")

    # ── Interactive Loop ───────────────────────────────────────────────────────
    edit_count   = 0
    last_results: List[SearchResult] = []

    while True:
        dirty_indicator = (
            f"  ● UNSAVED — {edit_count} edit(s)" if edit_count > 0 else "  ○ No changes"
        )
        print("\n" + "─" * 65)
        print(dirty_indicator)
        print()
        print("  [1]  Search: value contains substring")
        print("  [2]  Search: field length  (>, <, =)")
        print("  [3]  Edit  — pick row from last search results")
        print("  [4]  Edit  — go directly to a stem")
        print("  [5]  Show field path reference")
        print("  [6]  Show search examples")
        print("  [7]  Save & exit")
        print("  [8]  Exit without saving")
        print()

        choice = _prompt("Choose [1-8]: ")

        # ── Search: contains ──────────────────────────────────────────────────
        if choice == "1":
            print("\n  ── Search: value contains ─────────────────────────────")
            print(FIELD_HELP)
            field = _prompt("Field path: ")
            if not field:
                continue
            substr = _prompt("Contains string (blank = match all non-empty): ")
            last_results = search_contains(parsed, field, substr)
            print_table(last_results)

        # ── Search: length ────────────────────────────────────────────────────
        elif choice == "2":
            print("\n  ── Search: field length ───────────────────────────────")
            print(FIELD_HELP)
            field = _prompt("Field path: ")
            if not field:
                continue
            op = _prompt("Operator [> / < / =]: ")
            if op not in (">", "<", "="):
                print("  Invalid operator. Use >, <, or =\n")
                continue
            try:
                threshold = int(_prompt("Length value (integer): "))
            except ValueError:
                print("  Invalid length — must be an integer.\n")
                continue
            last_results = search_by_length(parsed, field, op, threshold)
            print_table(last_results)

        # ── Edit from results ─────────────────────────────────────────────────
        elif choice == "3":
            last_results, n = edit_from_results(last_results, tasks, parsed)
            edit_count += n

        # ── Edit by stem ──────────────────────────────────────────────────────
        elif choice == "4":
            print("\n  ── Edit by stem ───────────────────────────────────────")
            print(FIELD_HELP)
            n = edit_by_stem(tasks, parsed)
            edit_count += n

        # ── Field reference ───────────────────────────────────────────────────
        elif choice == "5":
            print(FIELD_HELP)

        # ── Search examples ───────────────────────────────────────────────────
        elif choice == "6":
            print(SEARCH_EXAMPLES)

        # ── Save & exit ───────────────────────────────────────────────────────
        elif choice == "7":
            out = save_output(tasks, source_file)
            if edit_count > 0:
                print(f"\n  ✓ Saved {edit_count} edit(s) → {out}")
            else:
                print(f"\n  Saved (no changes) → {out}")
            print()
            print("  ── Re-run import with: ─────────────────────────────────")
            print(f"  uv run raw_data/scripts/recipe_db/3_label_studio_converter.py \\")
            print(f"    --mode import \\")
            print(f"    --file label_studio/fix-annotated/{source_file.name}")
            print()
            sys.exit(0)

        # ── Exit without saving ───────────────────────────────────────────────
        elif choice == "8":
            if edit_count > 0:
                confirm = _prompt(
                    f"You have {edit_count} unsaved edit(s). Exit anyway? [y/n]: "
                )
                if confirm.lower() != "y":
                    continue
            print("  Exiting without saving.\n")
            sys.exit(0)

        else:
            print("  Invalid choice. Enter a number 1–8.\n")


if __name__ == "__main__":
    main()