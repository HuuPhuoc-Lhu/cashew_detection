import streamlit as st
from PIL import Image
from backend import predict_image, load_model

# ================== PAGE CONFIG & UI STYLE ==================
st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="wide")
st.markdown("""
<style>
body { background-color: #f6f8fa; }
.block-container { padding-top: 2rem; }

.card { 
    background-color: rgba(255, 255, 255, 0); /* trong suốt hoàn toàn */
    padding: 1.2rem; 
    border-radius: 14px; 
    box-shadow: none; /* loại bỏ shadow */
    margin-bottom: 1rem; 
}

.card-blur {
    background-color: rgba(255, 255, 255, 0.1); /* 10% màu trắng */
    backdrop-filter: blur(8px); /* mờ nhẹ nền phía sau */
    padding: 1.2rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

.header-title { text-align: center; color: #2e7d32; font-size: 36px; font-weight: 700; }
.header-sub { text-align: center; color: #555; margin-bottom: 2rem; }
</style>

""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🌿 Cashew Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Ứng dụng AI khoanh vùng và nhận dạng bệnh trên lá điều bằng YOLOv8</div>', unsafe_allow_html=True)

# ================== DISEASE INFO ==================
disease_info = {
    "healthy": {"description": "Lá khoẻ mạnh, không có dấu hiệu bệnh.", "treatment": "- Không cần xử lý.\n- Duy trì chăm sóc bình thường."},
    "leaf miner": {"description": "Sâu đục lá tạo đường hầm ngoằn ngoèo làm lá vàng và giảm quang hợp.", "treatment": "- Cắt bỏ lá bị nặng.\n- Dùng bẫy pheromone.\n- Phun thuốc sinh học chứa Abamectin hoặc Spinosad."},
    "red rust": {"description": "Bệnh rỉ sắt gây các đốm màu đỏ–cam, làm lá vàng và giảm năng suất.", "treatment": "- Cắt tỉa lá bệnh.\n- Tăng thông thoáng vườn.\n- Phun Copper Oxychloride hoặc Mancozeb."},
}

# ================== LOAD MODEL ==================
model = load_model()
if model:
    st.success("✅ Mô hình YOLOv8 đã được tải thành công")

# ================== SIDEBAR CONFIG ==================
st.sidebar.header("⚙️ Cấu hình dự đoán")
conf_thres = st.sidebar.slider("Ngưỡng độ tin cậy (Confidence)", min_value=0.2, max_value=0.7, value=0.35, step=0.05)
resize_enable = st.sidebar.checkbox("Chuẩn hóa ảnh (khuyên dùng cho mobile)", value=True)

# ================== UPLOAD IMAGE ==================
uploaded_file = st.file_uploader("📤 Tải lên ảnh lá điều", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    if resize_enable:
        max_w = 1024
        if image.width > max_w:
            new_h = int(image.height * max_w / image.width)
            image = image.resize((max_w, new_h))

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📷 Ảnh gốc")
        st.image(image, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")
    results = predict_image(image, conf=conf_thres)
    result_img = results[0].plot()

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("🧠 Kết quả phát hiện")
        st.image(result_img, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    boxes = results[0].boxes
    class_names = model.names
    if boxes is None or len(boxes) == 0:
        st.info("✔ Không phát hiện bệnh nào trên lá.")
    else:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf_val = float(box.conf[0]) * 100
            level = "🟢 Rất cao" if conf_val>=75 else "🟡 Trung bình" if conf_val>=50 else "🔴 Thấp"
            st.markdown(f"""
            <div class="card">
                <h4>🟩 Vùng {i+1}</h4>
                <b>Bệnh:</b> {cls_name}<br>
                <b>Độ tin cậy:</b> {conf_val:.2f}% — {level}<br><br>
                <b>📌 Mô tả:</b><br>{disease_info.get(cls_name, {}).get('description','')}<br><br>
                <b>🛠 Cách xử lý:</b><br>{disease_info.get(cls_name, {}).get('treatment','')}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("⬆️ Hãy tải lên 1 ảnh lá điều để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
