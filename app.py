import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import torch

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Cashew Leaf Disease Detection",
    layout="centered"
)

st.title("🌿 ỨNG DỤNG KHOANH VÙNG BỆNH TRÊN LÁ ĐIỀU (YOLOv8)")

# ================== THÔNG TIN BỆNH ==================
disease_info = {
    "healthy": {
        "description": "Lá khoẻ mạnh, không có dấu hiệu bệnh.",
        "treatment": "- Không cần xử lý.\n- Duy trì chăm sóc bình thường."
    },
    "leaf miner": {
        "description": "Sâu đục lá tạo đường hầm ngoằn ngoèo làm lá vàng và giảm quang hợp.",
        "treatment": "- Cắt bỏ lá bị nặng.\n- Dùng bẫy pheromone.\n- Phun thuốc sinh học chứa Abamectin hoặc Spinosad."
    },
    "red rust": {
        "description": "Bệnh rỉ sắt gây các đốm màu đỏ–cam, làm lá vàng và giảm năng suất.",
        "treatment": "- Cắt tỉa lá bệnh.\n- Tăng thông thoáng vườn.\n- Phun Copper Oxychloride hoặc Mancozeb."
    },
}

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình: {e}")
        return None

model = load_model()
if model:
    st.success("✅ Mô hình YOLOv8 đã được tải thành công")

# ================== SIDEBAR (TÙY CHỈNH) ==================
st.sidebar.header("⚙️ Cấu hình dự đoán")
conf_thres = st.sidebar.slider(
    "Ngưỡng độ tin cậy (Confidence)",
    min_value=0.2,
    max_value=0.7,
    value=0.35,
    step=0.05
)

resize_enable = st.sidebar.checkbox("Chuẩn hóa ảnh (khuyên dùng cho mobile)", value=True)

# ================== UPLOAD IMAGE ==================
uploaded_file = st.file_uploader(
    "📤 Tải lên ảnh lá điều",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # --- Load & preprocess image ---
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Chuẩn hóa độ phân giải (giảm khác biệt PC vs Mobile)
    if resize_enable:
        max_w = 1024
        if image.width > max_w:
            new_h = int(image.height * max_w / image.width)
            image = image.resize((max_w, new_h))

    st.image(image, caption="Ảnh gốc", use_container_width=True)
    st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")

    # --- YOLO Predict ---
    results = model.predict(
        source=image,
        conf=conf_thres,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    result_img = results[0].plot()
    st.image(result_img, caption="Ảnh đã khoanh vùng bệnh", use_container_width=True)

    # ================== RESULT INFO ==================
    st.subheader("📋 Thông tin chi tiết từng vùng bệnh")

    boxes = results[0].boxes
    class_names = model.names

    if boxes is None or len(boxes) == 0:
        st.info("✔ Không phát hiện bệnh nào trên lá.")
    else:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf = float(box.conf[0]) * 100

            # Đánh giá mức độ tin cậy (UX cho mobile)
            if conf >= 75:
                level = "🟢 Rất cao"
            elif conf >= 50:
                level = "🟡 Trung bình"
            else:
                level = "🔴 Thấp"

            st.markdown(f"### 🟩 Vùng {i + 1}")
            st.write(f"**Bệnh phát hiện:** `{cls_name}`")
            st.write(f"**Độ tin cậy:** {conf:.2f}% — {level}")

            info = disease_info.get(cls_name)
            if info:
                st.write("**📌 Mô tả:**")
                st.write(info["description"])
                st.write("**🛠 Cách xử lý:**")
                st.write(info["treatment"])

            st.markdown("---")

else:
    st.info("⬆️ Hãy tải lên 1 ảnh lá điều để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
