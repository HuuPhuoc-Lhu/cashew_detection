import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="centered")
st.title("🌿 ỨNG DỤNG KHOANH VÙNG BỆNH TRÊN LÁ ĐIỀU (YOLOv8)")


@st.cache_resource
def load_model():
    detect_path = "best.pt"    # ⚠️ lưu best.pt cùng thư mục với app.py khi deploy

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
    st.image(image, caption="Ảnh gốc", width='stretch')

    st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")

    results = detect_model.predict(
        image,
        conf=0.5,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    result_img = results[0].plot()

    st.image(result_img, caption="Ảnh đã khoanh vùng bệnh", width='stretch')

else:
    st.info("⬆️ Hãy tải lên 1 ảnh để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
