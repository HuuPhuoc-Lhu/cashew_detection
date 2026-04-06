from ultralytics import YOLO
from PIL import Image, ImageOps
import torch

# Load model 
_model = None

def load_model(path="best.pt"):
    global _model
    if _model is None:
        try:
            _model = YOLO(path)
        except Exception as e:
            raise RuntimeError(f"Không thể tải mô hình: {e}")
    return _model

def predict_image(image: Image.Image, conf=0.35, device=None):
    model = load_model()
    image = ImageOps.exif_transpose(image).convert("RGB")
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(image, conf=conf, device=device, verbose=False)
    return results