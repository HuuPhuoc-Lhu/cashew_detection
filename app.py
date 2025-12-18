import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="centered")
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


# ============= LOAD MODEL =============
@st.cache_resource
def load_model():
    detect_path = "best.pt"
    try:
        detect_model = YOLO(detect_path)
        return detect_model
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình khoanh vùng: {e}")
        return None


detect_model = load_model()
if detect_model:
    st.success("✅ Mô hình đã được tải thành công!")


# ============= UPLOAD ẢNH =============
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

    # ======= HIỂN THỊ THÔNG TIN BỆNH =======
    st.subheader("📋 Thông tin chi tiết từng vùng bệnh")

    class_names = detect_model.names

    if len(results[0].boxes) == 0:
        st.info("✔ Không phát hiện bệnh nào.")
    else:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf = float(box.conf[0]) * 100

            st.markdown(f"### 🟩 Vùng {i+1}")
            st.write(f"**Bệnh phát hiện:** `{cls_name}`")
            st.write(f"**Độ tin cậy:** {conf:.2f}%")

            # Lấy mô tả và cách xử lý
            info = disease_info.get(cls_name, None)
            if info:
                st.write("**📌 Mô tả:**")
                st.write(info["description"])

                st.write("**🛠 Cách xử lý:**")
                st.write(info["treatment"])

            st.markdown("---")

else:
    st.info("⬆️ Hãy tải lên 1 ảnh để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
