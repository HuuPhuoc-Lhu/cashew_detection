import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
from backend import predict_image
import cloudinary
import cloudinary.uploader
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from ultralytics import YOLO

# ================== LOAD YOLO ==================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# ================== CONFIG PAGE ==================
st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="wide")

# ================== GEMINI ==================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    GEMINI_OK = True
except:
    GEMINI_OK = False

# ================== CLOUDINARY ==================
cloudinary.config(
    cloud_name=st.secrets["CLOUD_NAME"],
    api_key=st.secrets["API_KEY"],
    api_secret=st.secrets["API_SECRET"]
)

# ================== GOOGLE SHEETS ==================
def save_log(user, diseases, image_url):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp"], scope
    )

    client = gspread.authorize(creds)
    sheet = client.open("cashew_log").sheet1

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user,
        ", ".join(diseases),
        f'=IMAGE("{image_url}", 3)'
    ], value_input_option="USER_ENTERED")

# ================== CLOUDINARY UPLOAD ==================
def upload_to_cloudinary(image_pil, user="guest"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    temp_path = f"temp_{timestamp}.jpg"
    image_pil.save(temp_path)

    result = cloudinary.uploader.upload(
        temp_path,
        folder=f"cashew/{user}",
        public_id=timestamp
    )

    return result["secure_url"]

# ================== DRAW BOXES (NO CV2) ==================
def draw_boxes(image, results, model):
    draw = ImageDraw.Draw(image)

    boxes = results[0].boxes
    if boxes is None:
        return image

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")

    return image

# ================== FALLBACK ==================
def generate_fallback(diseases):
    return f"""
📌 **Gợi ý xử lý**

- Bệnh: {", ".join(diseases)}
- Cắt bỏ lá bệnh
- Phun neem oil / nano đồng
- Giữ vườn khô thoáng
"""

# ================== UI ==================
st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌿 Cashew Leaf Disease Detection</h1>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Tùy chỉnh")

if GEMINI_OK:
    st.sidebar.success("🤖 Gemini Online")
else:
    st.sidebar.warning("⚠️ Gemini Offline")

conf_thres = st.sidebar.slider("Độ tin cậy", 0.1, 1.0, 0.35)

uploaded_file = st.file_uploader("📤 Tải ảnh", type=["jpg","png","jpeg"])

# ================== MAIN ==================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Ảnh gốc")

    # ===== YOLO =====
    with st.spinner("🔍 Đang quét..."):
        image_resized = image.resize((320, 320))
        results = predict_image(yolo_model, image_resized, conf=conf_thres)
        result_img = draw_boxes(image_resized.copy(), results, yolo_model)

    with col2:
        st.image(result_img, caption="Kết quả")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        detected = list(set([
            yolo_model.names[int(b.cls[0])] for b in boxes
        ]))

        st.warning(f"Phát hiện: {', '.join(detected)}")

        # ===== GEMINI =====
        if st.button("✨ Tư vấn AI"):
            if GEMINI_OK:
                try:
                    prompt = f"Cây điều bị: {', '.join(detected)}. Cách xử lý?"
                    res = gemini_model.generate_content(prompt)
                    st.write(res.text)
                except:
                    st.info(generate_fallback(detected))
            else:
                st.info(generate_fallback(detected))

        # ===== SAVE =====
        if st.button("💾 Lưu"):
            with st.spinner("Đang lưu..."):
                url = upload_to_cloudinary(result_img)
                save_log("guest", detected, url)

            st.success("Đã lưu!")
            st.write(url)

    else:
        st.success("Không phát hiện bệnh")

st.markdown("---")
st.caption("Lite YOLO • Streamlit Cloud Ready 🚀")
