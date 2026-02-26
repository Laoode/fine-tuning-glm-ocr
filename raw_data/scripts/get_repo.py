from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="UniqueData/ocr-receipts-text-detection",
    repo_type="dataset",
    local_dir="/teamspace/studios/this_studio/fine-tuning-glm-ocr/raw_data"
)