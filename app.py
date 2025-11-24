import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np

st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="centered")
st.title("🌿 ỨNG DỤNG KHOANH VÙNG BỆNH TRÊN LÁ ĐIỀU (YOLOv8)")

# 📌 Chú thích cho từng loại bệnh
disease_desc = {
    "leaf miner": "Bệnh sâu đục lá – xuất hiện các vệt trắng do sâu tấn công.",
    "red rust": "Bệnh rỉ sắt đỏ – xuất hiện các đốm hoặc mảng màu đỏ gỉ trên bề mặt lá.",
    "healthy": "Lá khỏe mạnh – không có dấu hiệu bệnh tật.",
}

@st.cache_resource
def load_model():
    detect_path = "best.pt"  # ⚠️ Lưu best.pt cùng thư mục với app.py khi deploy
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

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")

    # Chuyển PIL Image sang numpy array
    image_np = np.array(image)

    # Gọi model trực tiếp (không dùng predict)
    results = detect_model(
        image_np,
        conf=0.5,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    # Vẽ kết quả
    result_img = results[0].plot()
    st.image(result_img, caption="Ảnh đã khoanh vùng bệnh", use_container_width=True)

    # 📌 Lấy danh sách class dự đoán
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        pred_classes = [detect_model.names[int(c)] for c in boxes.cls.cpu().numpy()]

        st.subheader("📘 Chú thích bệnh được phát hiện:")

        # Loại bỏ trùng lặp để hiển thị gọn
        for cls in set(pred_classes):
            desc = disease_desc.get(cls, "Không có chú thích cho bệnh này.")
            st.markdown(f"**🔹 {cls}**: {desc}")
    else:
        st.warning("❗ Không phát hiện bệnh nào trong ảnh.")

else:
    st.info("⬆️ Hãy tải lên 1 ảnh để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
