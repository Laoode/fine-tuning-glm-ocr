from roboflow import Roboflow
import os

DOWNLOAD_LOCATION = "/teamspace/studios/this_studio/fine-tuning-glm-ocr/raw_data/roboflow" 
FORMAT = "yolov8"  
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key= ROBOFLOW_API_KEY)
project = rf.workspace("receipts-77003").project("receipts-dy2wq")
version = project.version(5)
dataset = version.download(model_format=FORMAT, location=DOWNLOAD_LOCATION)