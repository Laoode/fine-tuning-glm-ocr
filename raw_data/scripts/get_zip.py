import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

url = "https://expressexpense.com/large-receipt-image-dataset-SRD.zip"

raw_data_dir = Path(__file__).resolve().parent.parent  # raw_data/
output_zip = raw_data_dir / "large-receipt-image-dataset-SRD.zip"
extract_dir = raw_data_dir / "expressexpense_srd"

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_zip, "wb") as f, tqdm(
    desc="Downloading ZIP",
    total=total_size,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = f.write(data)
        bar.update(size)

print(f"Downloaded ZIP to: {output_zip}")

with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted to folder: {extract_dir}")