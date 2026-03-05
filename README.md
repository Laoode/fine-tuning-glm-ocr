# RECIPE-DB (RECeipt Image Processing & Extraction DataBase)

## Structure project
```
tree -L 3 -I '*.png|*.jpg|*.jpeg|.DS_Store|.git|__pycache__'
```
.
├── fine-tuning-glm-ocr
│   ├── README.md
│   ├── main.py
│   ├── pyproject.toml
│   ├── LLaMA-Factory/
│   │   ├── config/
│   │   │   ├──glm_ocr_full_sft.yaml
│   │   │   └──glm_ocr_lora_sft.yaml
│   │   ├── data/
│   │   │   ├── recipe_db/
│   │   │   │   ├──test/
│   │   │   │   ├──train/
│   │   │   │   │   ├── images/  (Semua image gabungan dari raw_data) *.jpg
│   │   │   │   │   ├── ocr/     (Hasil OCR mentah/text per file) *.txt
│   │   │   │   │   └── labels/  (Hasil JSON Key-Value per file) *.json
│   │   │   │   └──validation/
│   │   │   └── recipe_db.json
│   │   └── scripts/
│   │   │   └── main.py
│   ├── raw_data/
│   │   ├── cord-v2/
│   │   │   └── images/
│   │   │       ├──test/GAMBAR_0.jpg, GAMBAR_1.jpg, ...
│   │   │       ├──train/GAMBAR_0.jpg, GAMBAR_1.jpg, ...
│   │   │       └──validation/GAMBAR_0.jpg, GAMBAR_1.jpg, ...
│   │   ├── e_receipt/
│   │   │   └──  images/GAMBAR_0.jpg, GAMBAR_1.jpg, ...
│   │   ├── expressexpense/
│   │   │   └──  images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   ├── nanonets/
│   │   │   └──  images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   ├── roboflow/
│   │   │   ├── test/
│   │   │   │   └──images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   │   ├── train/
│   │   │   │   └──images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   │   └── valid/
│   │   │       └──images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   └── pinterest/
│   │   │   └──  images/GAMBAR_0.jpg, GAMBAR_1.jpg, ...
│   │   ├── scripts/
│   │   │   ├── convert_data.py
│   │   │   ├── get_hf.py
│   │   │   ├── get_kaggle.py
│   │   │   ├── get_oxen.py
│   │   │   ├── get_repo.py
│   │   │   ├── get_roboflow.py
│   │   │   ├── get_zip.py
│   │   │   ├── normalize_cord.py
│   │   │   ├── normalize_sroie.py
│   │   │   └── recipe_db/
│   │   │       ├── 1_ocr_extractor.py
│   │   │       ├── 2_kie_processor.py
│   │   │       ├── 3_label_studio_converter.py
│   │   │       ├── 4_final_formatter.py
│   │   │       └── config.py
│   │   ├── sroie/
│   │   │   ├── test/
│   │   │   │   └──img/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   │   └── train/
│   │   │       └──img/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   │   └── uniquedata/
│   │       └──  images/GAMBAR_0.JPG, GAMBAR_1.JPG, ...
│   └── uv.lock
├── main.py
└── vLLM-Server-Klaudia
    ├── README.md
    ├── glm-ocr
    │   ├── extract.py
    │   └── ocr.py
    └── light_on_ocr.py


--- Statistik Dataset ---
CORD-V2: 999
E-Receipt: 53
Express Expense: 200
Nanonets: 987
Roboflow: 1746
Pinterest: 502
SROIE: 973
Unique Data: 20
--------------------------
TOTAL SEMUA: 5480

```
echo "--- Statistik Dataset ---"
echo "CORD-V2: $(find raw_data/cord-v2/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "E-Receipt: $(find raw_data/e_receipt/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Express Expense: $(find raw_data/expressexpense/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Nanonets: $(find raw_data/nanonets/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Roboflow: $(find raw_data/roboflow -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Pinterest: $(find raw_data/pinterest -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "SROIE: $(find raw_data/sroie -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Unique Data: $(find raw_data/uniquedata/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "--------------------------"
echo "TOTAL SEMUA: $(find raw_data -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
```

(fine-tuning-glm-ocr) ⚡ main ~/fine-tuning-glm-ocr oxen init
🐂 repository initialized at: "/teamspace/studios/this_studio/fine-tuning-glm-ocr"

    📖 If this is your first time using Oxen, check out the CLI docs at:
            https://docs.oxen.ai/getting-started/cli

    💬 For more support, or to chat with the Oxen team, join our Discord:
            https://discord.gg/s3tBEn7Ptg

(fine-tuning-glm-ocr) ⚡ main ~/fine-tuning-glm-ocr oxen config --auth hub.oxen.ai SFMyNTY.g2gDbQAAAC9hcGlfa2V5X3YxOjdlNGUzMmZlLTE0NGItNDRiZC1iYjkwLWEzZWYwZWRmOGQ0Ym4GAACEAK-cAWIAAVGA.NYyoo5RfIKwGZArCkE41wc64jBQe0oi-pE17metrDpo
Authentication token set for host: hub.oxen.ai
(fine-tuning-glm-ocr) ⚡ main ~/fine-tuning-glm-ocr 


vllm serve zai-org/GLM-OCR --port 8000 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --allowed-local-media-path 

==================================================
  RECIPE-DB KIE Processor Status
==================================================
  Today (2026-03-04):
    API calls used  : 219 / 240
    Remaining today : 21
  All time:
    Processed       : 219
    Failed          : 21
==================================================