import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np

st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="centered")
st.title("🌿 ỨNG DỤNG KHOANH VÙNG BỆNH TRÊN LÁ ĐIỀU (YOLOv8)")

# 📌 Chú thích cho từng loại bệnh
disease_desc = {
    "leaf_miner": "Bệnh sâu đục lá xuất hiện các vệt trắng do sâu tấn công.",
    "red_rust": "Bệnh rỉ sắt đỏ xuất hiện các đốm hoặc mảng màu đỏ gỉ trên lá.",
    "healthy": "Lá hoàn toàn khỏe mạnh, không có dấu hiệu bệnh.",
}

@st.cache_resource
def load_model():
    detect_path = "best.pt"  # Lưu best.pt cùng thư mục app.py
    try:
        detect_model = YOLO(detect_path)
        return detect_model
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình khoanh vùng: {e}")
        return None

detect_model = load_model()

if detect_model:
    st.success("✅ Mô hình đã được tải thành công!")

uploaded_file = st.file_uploader("📤 Tải lên ảnh lá điều", type=["jpg", "jpeg", "png"])

# =====================  PROCESS IMAGE  ===========================
if uploaded_file is not None:

    # Đọc ảnh
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")

    image_np = np.array(image)

    # Chạy mô hình YOLO
    results = detect_model(
        image_np,
        conf=0.5,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    # Hiển thị ảnh đã khoanh vùng
    result_img = results[0].plot()
    st.image(result_img, caption="Ảnh đã khoanh vùng bệnh", use_container_width=True)

    # Lấy bounding boxes
    boxes = results[0].boxes

    # =====================  NO DISEASE => HEALTHY  ===========================
    if boxes is None or len(boxes) == 0:
        st.subheader("📘 Kết luận:")
        st.success("🌱 Lá khỏe mạnh (healthy) Không phát hiện dấu hiệu bệnh.")

        st.markdown(f"**🔹 healthy**: {disease_desc['healthy']}")
    else:
        # =====================  SHOW DISEASE DESCRIPTIONS  ====================
        pred_classes = [detect_model.names[int(c)] for c in boxes.cls.cpu().numpy()]

        st.subheader("📘 Chú thích bệnh được phát hiện:")

        for cls in set(pred_classes):
            desc = disease_desc.get(cls, "Không có mô tả cho bệnh này.")
            st.markdown(f"**🔹 {cls}**: {desc}")

else:
    st.info("⬆️ Hãy tải lên 1 ảnh để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8 • Framework: Streamlit 🚀")
