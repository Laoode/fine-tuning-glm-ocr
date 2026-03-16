"""
config.py
=========
Single source of truth for the RECIPE-DB pipeline.
All prompts, schemas, source definitions, and path resolution live here.

Location: fine-tuning-glm-ocr/raw_data/scripts/recipe_db/config.py

Design decisions:
- EXTRACTION_SCHEMA is the ground truth — FEW_SHOT_OUTPUT MUST mirror it exactly
- EXTRACTION_RULES references all schema fields — no orphan rules
- All monetary amounts strip currency symbols (clean numbers only)
- store_contacts.type = label as-is from receipt (no classification/guessing)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Path Resolution ──────────────────────────────────────────────────────────
# config.py  → raw_data/scripts/recipe_db/
# .parent    → raw_data/scripts/
# .parent    → raw_data/
# .parent    → fine-tuning-glm-ocr/   ← PROJECT_ROOT

_HERE         = Path(__file__).resolve().parent
PROJECT_ROOT  = _HERE.parent.parent.parent          # fine-tuning-glm-ocr/
RAW_DATA_DIR  = PROJECT_ROOT / "raw_data"
RECIPE_DB_DIR = PROJECT_ROOT / "LLaMA-Factory" / "data" / "recipe_db"
LOGS_DIR      = PROJECT_ROOT / "logs" / "recipe_db"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "recipe_db"


def split_dir(split: str) -> Path:
    return RECIPE_DB_DIR / split

def images_dir(split: str) -> Path:
    return split_dir(split) / "images"

def ocr_dir(split: str) -> Path:
    return split_dir(split) / "ocr"

def labels_dir(split: str) -> Path:
    return split_dir(split) / "labels"


# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

LLM_ENDPOINT    = os.getenv("LLM_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemini-3.1-pro-preview")
LLM_API_KEY     = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

GLM_ENDPOINT = os.getenv("GLM_ENDPOINT", "http://localhost:8000/v1")
GLM_MODEL    = os.getenv("GLM_MODEL", "zai-org/GLM-OCR")

OXEN_REMOTE     = os.getenv("OXEN_REMOTE", "https://hub.oxen.ai/Laoode/RECIPE-DB")
OXEN_AUTH_TOKEN = os.getenv("OXEN_AUTH_TOKEN", "")

# Prompt Repetition:
#   false → thinking models (gemini-3.1-pro-preview): reason internally, repetition wastes tokens
#   true  → non-thinking models (gemini-2.5-flash etc.): 47 wins / 0 losses (paper)
PROMPT_REPETITION = os.getenv("PROMPT_REPETITION", "false").lower() == "true"

# ─── Source Dataset Configurations ────────────────────────────────────────────
SOURCE_CONFIGS = [
    {
        "name":       "cord",
        "prefix":     "cord",
        "path":       RAW_DATA_DIR / "cord-v2" / "images" / "train",
        "exts":       ["*.jpg", "*.jpeg", "*.png"],
        "limit":      50,
        "test_ratio": 0.05,
    },
    {
        "name":       "ereceipt",
        "prefix":     "erec",
        "path":       RAW_DATA_DIR / "e_receipt" / "images",
        "exts":       ["*.jpg", "*.jpeg", "*.png"],
        "limit":      None,   # 53 total → take all
        "test_ratio": 0.05,
    },
    {
        "name":       "expressexpense",
        "prefix":     "expr",
        "path":       RAW_DATA_DIR / "expressexpense" / "images",
        "exts":       ["*.JPG", "*.jpg", "*.jpeg", "*.png"],
        "limit":      50,
        "test_ratio": 0.05,
    },
    {
        "name":       "nanonets",
        "prefix":     "nano",
        "path":       RAW_DATA_DIR / "nanonets" / "images",
        "exts":       ["*.JPG", "*.jpg", "*.jpeg", "*.png"],
        "limit":      50,
        "test_ratio": 0.05,
    },
    {
        "name":       "roboflow",
        "prefix":     "robo",
        "path":       RAW_DATA_DIR / "roboflow" / "train" / "images",
        "exts":       ["*.JPG", "*.jpg", "*.jpeg", "*.png"],
        "limit":      50,
        "test_ratio": 0.05,
    },
    {
        "name":       "pinterest",
        "prefix":     "pint",
        "path":       RAW_DATA_DIR / "pinterest" / "images",
        "exts":       ["*.jpg", "*.jpeg", "*.png"],
        "limit":      None,   # 502 → take all
        "test_ratio": 0.05,
    },
    {
        "name":       "sroie",
        "prefix":     "sroe",
        "path":       RAW_DATA_DIR / "sroie" / "train" / "img",
        "exts":       ["*.JPG", "*.jpg", "*.jpeg", "*.png"],
        "limit":      50,
        "test_ratio": 0.05,
    },
    {
        "name":       "uniquedata",
        "prefix":     "uniq",
        "path":       RAW_DATA_DIR / "uniquedata" / "images",
        "exts":       ["*.JPG", "*.jpg", "*.jpeg", "*.png"],
        "limit":      None,   # 20 → take all
        "test_ratio": 0.05,
    },
]
# name: threads
# prefix: threx
# path: RAW_DATA_DIR / threads
# limit: None # 74 total → take all
# test_ratio: 0.134

# ─── GLM-OCR ──────────────────────────────────────────────────────────────────
# Per GLM-OCR docs: predefined prompt for raw text extraction
GLM_OCR_PROMPT = "Text Recognition:"

# vLLM batches internally — flood with concurrent requests to saturate L4 24GB GPU
# GLM-OCR is 0.9B; L4 can handle many concurrent requests before VRAM saturation
GLM_MAX_WORKERS = 24

# ─── KIE Rate Limits ──────────────────────────────────────────────────────────
# gemini-3.1-pro-preview Tier 1: 25 RPM / 250 RPD
# Safe margins: 20 RPM / 240 RPD
KIE_RPM_LIMIT     = 20
KIE_RPD_LIMIT     = 240
KIE_DELAY_SECONDS = 60.0 / KIE_RPM_LIMIT   # 3.0s between requests

# ─── EXTRACTION SCHEMA (GROUND TRUTH) ────────────────────────────────────────
# ⚠️  This is the single source of truth for the data schema.
# ⚠️  FEW_SHOT_OUTPUT below MUST mirror every key exactly.
# ⚠️  If you add/remove a field here, update FEW_SHOT_OUTPUT and EXTRACTION_RULES too.
EXTRACTION_SCHEMA = {
    "info": {
        "store_name": "",
        "store_location": "",
        "store_contacts": [
            {"type": "", "value": ""}
        ],
        "tax_id": "",
        "receipt_id": "",
        "payment_date": "",
        "payment_time": "",
        "time_unit": ""
    },
    "items": [
        {
            "item_name": "",
            "quantity": "",
            "unit_price": "",
            "discount_label": "",
            "discount_price": "",
            "tax_label": "",
            "total_price": ""
        }
    ],
    "returned_items": [
        {
            "item_name": "",
            "quantity": "",
            "unit_price": "",
            "total_refund": ""
        }
    ],
    "payment": {
        "total_items": "",
        "currency": "",
        "subtotal_price": "",
        "discounts": [
            {"discount_name": "", "amount": ""}
        ],
        "taxes": [
            {"tax_name": "", "amount": ""}
        ],
        "additional_charges": [
            {"charge_name": "", "amount": ""}
        ],
        "grand_total": "",
        "rounding": "",
        "payment_method": "",
        "tendered": "",
        "change": ""
    }
}

EXTRACTION_RULES = """
EXTRACTION RULES (read carefully):
- Format output as JSON matching the schema exactly — no extra keys, no missing keys.
- EXACT COPY (CASE SENSITIVE): Copy all text exactly as written. Do not autocorrect or normalize.
- Zero and Placeholder Values: If a label exists but its value is "0", "0.00", or "-", copy it exactly. Do NOT treat as empty.
- Extract ONLY text explicitly written in the receipt. Never calculate, sum, or infer.
- Numbers: copy EXACTLY as written ("193.00", "5.00%", "0.00") — no conversion.
- Clean Numbers: For all price/amount fields, extract ONLY the numeric value. NEVER include currency symbols (Rp, $, RM, €, etc.).
- Strings not found: use "" (empty string).
- Arrays not found: use [] (empty array). Do NOT put empty placeholder objects inside arrays.

receipt_info:
- receipt_id: look for Receipt No, Invoice No, Bill No, Ref No, or similar.
- tax_id: is the store's tax registration number (e.g. NPWP, GST ID).

store_name & store_location:
- store_name: extract the official name of the store as printed on the receipt. This is often at the top and sometimes at the bottom. Also be aware with store include the branch name in the store name (e.g., "ALFAMART Kebayoran Lama"). It make sure included, cause that's not location store.
- store_location: extract the full address of the store as printed on the receipt. This may include street name, number, city, postal code, and other location details. If the receipt only has a general location (e.g., "Jakarta"), copy that as is.

store_contacts:
- Extract the label (e.g., "Tel", "Tel.", "Phone", "Fax", "WA", "Website", etc.) exactly as written into "type".
- If there is NO label for a contact value, "type" MUST be "".
- DO NOT classify or guess the type. No label = empty type, always.
- Example: "Tel.: 07-123" → {"type": "Tel.", "value": "07-123"}
- Example: "Fax: -" (no value) → {"type": "Fax", "value": "-"}
- Example: "www.store.com" (no "Website" label) → {"type": "", "value": "www.store.com"}

payment_time & time_unit:
- payment_time: Copy only numbers/colon (e.g., "15:34:15", "02:44").
- time_unit: Copy only the unit IF written (e.g., "AM", "PM", "WIB", "UTC", "+07:00", "+08:00"). Otherwise "".
- If multiple times exist, select the chronologically latest time, or pick the one closest to "TOTAL".

items & returned_items:
- item_name: extract ONLY the descriptive product text. DO NOT include internal codes, SKUs, or barcodes.
- quantity: the number of units. Copy only the number — never include "x", "@", "PCS", "*".
- unit_price: price for ONE single unit.
- total_price / total_refund: final extended price for that line.
- discount_label: copy exact text of any per-item discount label (e.g., "DISKON: (20%)", "Member Disc."). Use "" if none.
- discount_price: copy the number printed next to/below discount_label. Include "-" if written. Use "" if none.
- tax_label: tax code or category on the item line (e.g., "SR", "ZRL", "6%"). Use "" if not shown per item.

payment summary:
- total_items: fill ONLY if explicitly written (e.g., "Total Qty", "Item Count"). Do NOT count rows yourself.
- subtotal_price: look for "Subtotal", "Total Sales (Excl. GST)", "Net Amount", "Total (Excl. Tax)".
- currency: symbol ($, Rp, RM, €, etc.) ONLY if printed on receipt. Do not infer from location.
- discount labels and amounts: for each line item that reduces the total price (e.g., 'Discount 35%', 'KOTA HEMAT', 'Voucher', 'Promo Code', 'Total Diskon', 'Total Voucher', or similar), create a separate entry with the exact label as 'discount_name' and the numeric value as 'amount'. Include the negative sign '-' if written.
- taxes: each tax line — exact label → "tax_name", numeric value → "amount" (e.g. tax_name: "GST 6%", amount: "5000", etc) also include total tax if writen.
- additional_charges: service charge, delivery fee, etc. — exact label → "charge_name", value → "amount" (e.g. charge_name: "Biaya perngiriman", amount: "16,000", etc). also include total charge if writen.
- grand_total: final amount paid. Copy exactly as printed. Do NOT calculate.
- payment_method: copy label exactly as written. Do NOT translate (keep "Tunai", not "Cash").
- tendered: amount handed over by customer.
- rounding: include "+" or "-" sign if explicitly written.
- change: amount returned to customer.

DATA PRIVACY: Skip personal names (cashier/customer) and card numbers.
""".strip()

# ─── FEW-SHOT EXAMPLE ────────────────────────────────────────────────────────
# CRITICAL: FEW_SHOT_OUTPUT must contain EVERY key from EXTRACTION_SCHEMA.
# This receipt exercises every rule edge case listed above.
# If you update EXTRACTION_SCHEMA, update this too.

FEW_SHOT_RECEIPT_TEXT = """SUNRISE TRADING SDN BHD
No. 12, Jalan Bahagia 3, Taman Sejahtera
81300 Johor Bahru, Johor
Tel.: 07-3881234
Fax: 07-3885678
ryuk@gmail.com
GST ID: 001234567890
------------------------------
TAX INVOICE
INVOICE NO : INV-20190115-0042
DATE       : 15/01/2019 14:35:22 +08:00
------------------------------
DESCRIPTION           QTY   PRICE   DISC        AMOUNT
SAFETY SHOES KWD       2   95.00   MEMBER -9.50  180.50 SR
WORK GLOVES            3    2.50                   7.50
------------------------------
RETURN ITEM
SAFETY HELMET          1   45.00              -45.00
------------------------------
Total Qty   : 5
SUBTOTAL    : 188.00
MEMBER DISC : -5.00
DISC TOTAL  : -5.00
SST 6%      : 11.28
SERVICE CHARGE 5% : 9.40
ROUNDING    : +0.02
TOTAL       : 203.70
VISA CARD   : 203.70"""

# MUST mirror EXTRACTION_SCHEMA exactly — every key present, correct array types
FEW_SHOT_OUTPUT = {
    "info": {
        "store_name": "SUNRISE TRADING SDN BHD",
        "store_location": "No. 12, Jalan Bahagia 3, Taman Sejahtera 81300 Johor Bahru, Johor",
        "store_contacts": [
            {"type": "Tel.", "value": "07-3881234"},
            {"type": "Fax",  "value": "07-3885678"},
            {"type": "",     "value": "ryuk@gmail.com"}
        ],
        "tax_id": "001234567890",
        "receipt_id": "INV-20190115-0042",
        "payment_date": "15/01/2019",
        "payment_time": "14:35:22",
        "time_unit": "+08:00"
    },
    "items": [
        {
            "item_name": "SAFETY SHOES KWD",
            "quantity": "2",
            "unit_price": "95.00",
            "discount_label": "MEMBER",
            "discount_price": "-9.50",
            "tax_label": "SR",
            "total_price": "180.50"
        },
        {
            "item_name": "WORK GLOVES",
            "quantity": "3",
            "unit_price": "2.50",
            "discount_label": "",
            "discount_price": "",
            "tax_label": "",
            "total_price": "7.50"
        }
    ],
    "returned_items": [
        {
            "item_name": "SAFETY HELMET",
            "quantity": "1",
            "unit_price": "45.00",
            "total_refund": "-45.00"
        }
    ],
    "payment": {
        "total_items": "5",
        "currency": "",
        "subtotal_price": "188.00",
        "discounts": [
            {"discount_name": "MEMBER DISC", "amount": "-5.00"},
            {"discount_name": "DISC TOTAL", "amount": "-5.00"}
        ],
        "taxes": [
            {"tax_name": "SST 6%", "amount": "11.28"}
        ],
        "additional_charges": [
            {"charge_name": "SERVICE CHARGE 5%", "amount": "9.40"}
        ],
        "grand_total": "203.70",
        "rounding": "+0.02",
        "payment_method": "VISA CARD",
        "tendered": "203.70",
        "change": ""
    }
}

# ─── Prompt Builders ──────────────────────────────────────────────────────────

def build_kie_prompt(ocr_text: str) -> str:
    """
    Build KIE extraction prompt for Gemini.

    Technique stack (per KIE paper + Prompt Repetition paper):
      1. Field-specific extraction rules    [KIE paper — Manual Prompt approach]
      2. One-shot example                   [KIE paper — one-shot outperforms zero-shot]
      3. Prompt Repetition x2 (optional)    [Prompt Repetition paper]
         PROMPT_REPETITION=false: thinking models (default for gemini-3.1-pro-preview)
         PROMPT_REPETITION=true : non-thinking flash models
    """
    schema_str  = json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)
    example_str = json.dumps(FEW_SHOT_OUTPUT,   ensure_ascii=False, indent=2)

    core = (
        "A receipt OCR text is shown below.\n\n"
        f"{EXTRACTION_RULES}\n\n"
        "OUTPUT SCHEMA — fill ALL fields, match EXACTLY:\n"
        f"{schema_str}\n\n"
        "ONE-SHOT EXAMPLE\n"
        "Receipt OCR text:\n"
        f"{FEW_SHOT_RECEIPT_TEXT}\n\n"
        "Expected Output:\n"
        f"{example_str}\n\n"
        "---BEGIN RECEIPT OCR TEXT---\n"
        f"{ocr_text}\n"
        "---END RECEIPT OCR TEXT---\n\n"
        "Output ONLY valid JSON. No markdown fences. No explanation. No extra text."
    )

    if PROMPT_REPETITION:
        separator = (
            "\n\n---\n"
            "[Re-reading receipt and instructions for accuracy — Prompt Repetition]\n"
            "---\n\n"
        )
        return core + separator + core

    return core


def build_glm_finetune_user_prompt() -> str:
    """
    User-side prompt for GLM-OCR fine-tuning (Information Extraction task).
    Per GLM-OCR docs: prompt format is '请按下列JSON格式输出图中信息:' + schema
    Image token '<image>' must precede the text.
    """
    schema_str = json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)
    return f"<image>请按下列JSON格式输出图中信息:\n{schema_str}"