from PIL import Image, ImageOps

def predict_image(model, image: Image.Image, conf=0.35):
    image = ImageOps.exif_transpose(image).convert("RGB")

    results = model.predict(
        source=image,
        imgsz=320,          # 🔥 giảm RAM
        conf=conf,
        verbose=False
    )

    return results
