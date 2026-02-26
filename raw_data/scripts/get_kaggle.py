import kagglehub
KAGGLE_API_TOKEN = "KGAT_4e71cd05529c823c4b4b997a742587a8"

# Download latest version
path = kagglehub.dataset_download("urbikn/sroie-datasetv2")

print("Path to dataset files:", path)