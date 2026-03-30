import os
import random
from ultralytics import YOLO
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"

IMAGE_DIR = "test/images"     # 📁 folder ảnh
LABEL_DIR = "test/labels"     # 📁 folder label YOLO

CONF_THRESHOLD = 0.1
IOU_THRESHOLD  = 0.01
MAX_TEST_IMAGES = 100

OUTPUT_FILE = "yolo_test_result2.txt"

# Class mapping đúng theo file data.yaml
CLASS_NAMES = {
    0: "healthy",
    1: "leaf miner",
    2: "red rust"
}

# =========================
# IOU FUNCTION
# =========================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# =========================
# LOAD MODEL
# =========================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")


# =========================
# GET RANDOM IMAGES
# =========================
all_images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

random.shuffle(all_images)
test_images = all_images[:MAX_TEST_IMAGES]


# =========================
# OPEN FILE
# =========================
f = open(OUTPUT_FILE, "w", encoding="utf-8")

correct = 0
wrong_class = 0
no_detection = 0


# =========================
# TEST LOOP
# =========================
for file in test_images:

    img_path   = os.path.join(IMAGE_DIR, file)
    label_path = os.path.join(LABEL_DIR, file.rsplit(".", 1)[0] + ".txt")

    # ------------------
    # LOAD IMAGE SIZE
    # ------------------
    image = Image.open(img_path)
    w_img, h_img = image.size

    # ------------------
    # LOAD GT (YOLO LABEL)
    # ------------------
    gt_boxes = []

    if not os.path.exists(label_path):
        print("⚠️ No GT box:", file)
        no_detection += 1
        continue

    with open(label_path, "r") as fgt:
        for line in fgt:
            parts = line.strip().split()

            # ⚠️ Có dataset polygon → chỉ lấy 5 phần đầu
            if len(parts) < 5:
                continue

            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
            except:
                continue

            # Convert YOLO normalized → pixel xyxy
            xmin = (xc - bw / 2) * w_img
            ymin = (yc - bh / 2) * h_img
            xmax = (xc + bw / 2) * w_img
            ymax = (yc + bh / 2) * h_img

            if xmax <= xmin or ymax <= ymin:
                continue

            gt_boxes.append((cls, [xmin, ymin, xmax, ymax]))

    if len(gt_boxes) == 0:
        print("⚠️ No valid GT box:", file)
        no_detection += 1
        continue

    # ------------------
    # PREDICT
    # ------------------
    results = model(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes   = results.boxes

    if boxes is None or len(boxes) == 0:
        no_detection += 1
        msg = f"{file} : ⚫ NO_DETECTION\n"
        print(msg.strip())
        f.write(msg)
        continue

    pred_boxes  = boxes.xyxy.cpu().numpy()
    pred_labels = boxes.cls.cpu().numpy().astype(int)

    # ------------------
    # MATCHING LOGIC
    # ------------------
    matched = False
    wrong_cls = False

    for pbox, pcls in zip(pred_boxes, pred_labels):
        for gtcls, gtbox in gt_boxes:

            iou = compute_iou(pbox, gtbox)

            if iou >= IOU_THRESHOLD and pcls == gtcls:
                matched = True

            elif iou >= IOU_THRESHOLD and pcls != gtcls:
                wrong_cls = True

    # ------------------
    # RESULT
    # ------------------
    if matched:
        correct += 1
        msg = f"{file} : ✅ CORRECT\n"

    elif wrong_cls:
        wrong_class += 1
        msg = f"{file} : 🔴 WRONG_CLASS\n"

    else:
        no_detection += 1
        msg = f"{file} : ⚫ NO_DETECTION\n"

    print(msg.strip())
    f.write(msg)


# =========================
# SUMMARY
# =========================
f.write("\n========== SUMMARY ==========\n")
f.write(f"Total images      : {len(test_images)}\n")
f.write(f"Correct           : {correct}\n")
f.write(f"Wrong class        : {wrong_class}\n")
f.write(f"No detection       : {no_detection}\n")

f.close()
print("\n✅ Saved:", OUTPUT_FILE)
