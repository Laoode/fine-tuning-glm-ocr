# RECIPE-DB (RECeipt Image Processing & Extraction DataBase)

## Structure project
```bash
tree -L 3 -I '*.png|*.jpg|*.jpeg|.DS_Store|.git|__pycache__'
```
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
│   ├── checkpoints
│   │   └── recipe_db
│   │       ├── kie_processor.json
│   │       └── ocr_extractor.json
│   ├── label_studio
│   │   ├── VERIFY_REPORT.md
│   │   ├── batch
│   │   │   └── 818-885-fixed.json
│   │   ├── labeling_config.xml
│   │   ├── skipped.json
│   │   └── tasks.json
│   ├── logs
│   │   └── recipe_db
│   │       ├── 1_ocr_extractor.log
│   │       ├── 2_kie_processor.log
│   │       └── 3_label_studio_converter.log
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
│   │   └── primary/
│   │   │   └──  GAMBAR_0.jpg, GAMBAR_1.jpg, ...
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
│   │   ├── threads/
│   │   │   └──  GAMBAR_0.JPG, GAMBAR_1.JPG, ...
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
```
```
--- Statistik Dataset ---
CORD-V2: 999
E-Receipt: 53
Express Expense: 200
Nanonets: 987
Roboflow: 1746
Pinterest: 502
Primary: 65
SROIE: 973
Threads: 74
Unique Data: 20
--------------------------
TOTAL SEMUA: 5480
```

```bash
echo "--- Statistik Dataset ---"
echo "CORD-V2: $(find raw_data/cord-v2/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "E-Receipt: $(find raw_data/e_receipt/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Express Expense: $(find raw_data/expressexpense/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Nanonets: $(find raw_data/nanonets/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Roboflow: $(find raw_data/roboflow -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Pinterest: $(find raw_data/pinterest -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Primary: $(find raw_data/primary -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "SROIE: $(find raw_data/sroie -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Threads: $(find raw_data/threads -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "Unique Data: $(find raw_data/uniquedata/images -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
echo "--------------------------"
echo "TOTAL SEMUA: $(find raw_data -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)"
```

Running Label Studio:
```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/teamspace/studios/this_studio/fine-tuning-glm-ocr \
CSRF_TRUSTED_ORIGINS=https://8081-01kjd8jvd2eprsmsg5x8mq4g39.cloudspaces.litng.ai,https://*.cloudspaces.litng.ai \
DJANGO_CSRF_TRUSTED_ORIGINS=https://8081-01kjd8jvd2eprsmsg5x8mq4g39.cloudspaces.litng.ai,https://*.cloudspaces.litng.ai \
uv run label-studio start --port 8081
```

Import Annotations:
```bash
uv run raw_data/scripts/recipe_db/3_label_studio_converter.py \
--mode import \
--file label_studio/fix-annotated/818-885-fixed.json
```