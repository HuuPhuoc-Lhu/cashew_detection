import streamlit as st
from PIL import Image
import google.generativeai as genai
from backend import predict_image, load_model
import os
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    st.sidebar.success("🤖 Gemini: Online")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi AI: {e}")
# ================== 1. CONFIG TRANG (BẮT BUỘC Ở DÒNG ĐẦU TIÊN) ==================
st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="wide")


# ================== 3. GIAO DIỆN (UI STYLE) ==================
st.markdown("""
<style>
    .header-title { text-align: center; color: #2e7d32; font-size: 36px; font-weight: 700; }
    .header-sub { text-align: center; color: #555; margin-bottom: 2rem; }
    .card-result { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #2e7d32;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🌿 Cashew Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Hệ thống nhận diện bệnh lá điều thông minh (YOLOv8 + Gemini AI)</div>', unsafe_allow_html=True)

# ================== 4. TẢI MÔ HÌNH YOLOv8 ==================
yolo_model = load_model()

# ================== 5. THANH SIDEBAR ==================
st.sidebar.header("⚙️ Tùy chỉnh")
conf_thres = st.sidebar.slider("Độ tin cậy (Confidence)", 0.1, 1.0, 0.35)

# ================== 6. XỬ LÝ UPLOAD VÀ DỰ ĐOÁN ==================
uploaded_file = st.file_uploader("📤 Tải lên ảnh lá điều để phân tích", type=["jpg", "jpeg", "png"])

if uploaded_file and yolo_model:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Ảnh gốc")
        st.image(image, use_container_width=True)

    # Chạy YOLOv8
    with st.spinner("🔍 Đang quét vùng bệnh..."):
        results = predict_image(image, conf=conf_thres)
        result_img = results[0].plot()

    with col2:
        st.subheader("🧠 Kết quả AI quét")
        st.image(result_img, use_container_width=True)

    # Phân tích kết quả từ YOLO để gửi cho Gemini
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.divider()
        st.subheader("📝 Phân tích chi tiết & Tư vấn điều trị")
        
        # Lấy danh sách tên các bệnh phát hiện được (loại bỏ trùng lặp)
        detected_diseases = list(set([yolo_model.names[int(b.cls[0])] for b in boxes]))
        disease_list_str = ", ".join(detected_diseases)

        # Hiển thị tóm tắt
        st.warning(f"Phát hiện dấu hiệu của: **{disease_list_str}**")

        # Nút gọi Gemini tư vấn
        if st.button("✨ Nhận lời khuyên từ Chuyên gia Gemini"):
            with st.spinner("Gemini đang phân tích bệnh trạng..."):
                # Tạo prompt thông minh gửi cho Gemini
                prompt = f"""
                Với tư cách là một chuyên gia nông nghiệp, hãy phân tích và đưa ra giải pháp cho cây điều khi lá bị các triệu chứng sau: {disease_list_str}.
                Yêu cầu: 
                1. Giải thích ngắn gọn nguyên nhân.
                2. Đưa ra các bước xử lý (hóa học hoặc sinh học).
                3. Cách phòng ngừa cho cả vườn.
                Trả lời bằng tiếng Việt, trình bày rõ ràng bằng bullet points.
                """
                try:
                    response = gemini_model.generate_content(prompt)
                    st.markdown(f'<div class="card-result">{response.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Không thể kết nối Gemini: {e}")
    else:
        st.success("✅ Tuyệt vời! Không phát hiện dấu hiệu bệnh lý trên lá này.")

# ================== 7. FOOTER ==================
st.markdown("---")
st.caption("Ứng dụng chạy trên GitHub Codespaces • Kết nối Google AI Studio")