from PIL import Image, ImageOps

def predict_image(model, image: Image.Image, conf=0.3):
    image = ImageOps.exif_transpose(image).convert("RGB")

    # 🔥 resize nhỏ để nhẹ RAM
    image = image.resize((320, 320))

    results = model.predict(
        source=image,
        imgsz=320,
        conf=conf,
        device="cpu",
        verbose=False
    )

    return results
