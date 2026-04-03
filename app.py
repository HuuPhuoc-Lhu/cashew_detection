import streamlit as st
from PIL import Image
import google.generativeai as genai
from backend import predict_image, load_model
import os
from dotenv import load_dotenv

# ================== LOAD ENV ==================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        GEMINI_OK = True
    except Exception:
        gemini_model = None
        GEMINI_OK = False
else:
    gemini_model = None
    GEMINI_OK = False

# ================== CONFIG PAGE ==================
st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="wide")

# ================== UI ==================
st.markdown("""
<style>
    .header-title { text-align: center; color: #2e7d32; font-size: 36px; font-weight: 700; }
    .header-sub { text-align: center; color: #555; margin-bottom: 2rem; }
    .card-result { 
       .card-result { 
    background-color: #ffffff; 
    color: #000000;  /* thêm dòng này */
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #2e7d32;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🌿 Cashew Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">YOLOv8 + Gemini 2.5 Flash</div>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Tùy chỉnh")

if GEMINI_OK:
    st.sidebar.success("🤖 Gemini 2.5: Online")
else:
    st.sidebar.warning("⚠️ Gemini: Offline (fallback mode)")

conf_thres = 0.60

# ================== LOAD YOLO ==================
yolo_model = load_model()

# ================== UPLOAD ==================
uploaded_file = st.file_uploader("📤 Tải ảnh lá điều", type=["jpg", "jpeg", "png"])

if uploaded_file and yolo_model:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Ảnh gốc")
        st.image(image, width='stretch')

    # YOLO predict
    with st.spinner("🔍 Đang quét bệnh..."):
        results = predict_image(image, conf=conf_thres)
        result_img = results[0].plot()

    with col2:
        st.subheader("🧠 Kết quả")
        st.image(result_img, width='stretch')

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        st.divider()
        st.subheader("📝 Phân tích & Tư vấn")

        detected_diseases = list(set([yolo_model.names[int(b.cls[0])] for b in boxes]))
        disease_list_str = ", ".join(detected_diseases)

        st.warning(f"Phát hiện: **{disease_list_str}**")

        if st.button("✨ Nhận tư vấn AI"):
            with st.spinner("Đang phân tích..."):

                prompt = f"""
                Bạn là chuyên gia nông nghiệp.
                Cây điều có các bệnh: {disease_list_str}

                Hãy:
                1. Giải thích nguyên nhân ngắn gọn
                2. Đưa ra cách xử lý
                3. Cách phòng ngừa

                Trả lời bằng tiếng Việt, dạng bullet.
                """

                # ================== GEMINI ==================
                if GEMINI_OK:
                    try:
                        response = gemini_model.generate_content(prompt)
                        st.markdown(f'<div class="card-result">{response.text}</div>', unsafe_allow_html=True)

                    except Exception as e:
                        if "429" in str(e):
                            st.error("🚫 Hết quota Gemini → dùng fallback")
                        else:
                            st.error(f"Lỗi Gemini: {e}")

                        # fallback
                        st.info(generate_fallback(detected_diseases))

                else:
                    # fallback luôn nếu Gemini off
                    st.info(generate_fallback(detected_diseases))

    else:
        st.success("✅ Không phát hiện bệnh")

# ================== FALLBACK ==================
def generate_fallback(diseases):
    return f"""
📌 **Gợi ý xử lý (offline AI)**

- Bệnh phát hiện: {", ".join(diseases)}
- Cắt bỏ lá bị nhiễm nặng
- Phun thuốc sinh học (nano đồng, neem oil)
- Giữ vườn thông thoáng, tránh ẩm cao
- Theo dõi 5–7 ngày để đánh giá lại

👉 Khuyến nghị: kết hợp xử lý sớm để tránh lan rộng
"""

# ================== FOOTER ==================
st.markdown("---")
st.caption("YOLOv8 + Gemini 2.5 Flash • Có fallback khi lỗi API")
