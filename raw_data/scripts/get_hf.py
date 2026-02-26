from datasets import load_dataset

dataset = load_dataset(
    "nanonets/key_information_extraction",
    cache_dir="/teamspace/studios/this_studio/fine-tuning-glm-ocr/raw_data"
)

print(dataset)