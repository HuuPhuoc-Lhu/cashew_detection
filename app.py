import streamlit as st
from PIL import Image
import google.generativeai as genai
from backend import predict_image, load_model
import cloudinary
import cloudinary.uploader
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

    creds_dict = st.secrets["gcp"]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, scope
    )

    client = gspread.authorize(creds)
    sheet = client.open("cashew_log").sheet1

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user,
        ", ".join(diseases),
        f'=IMAGE("{image_url}";3)'
    ],
     value_input_option="USER_ENTERED"
    )
    

# ================== UPLOAD CLOUDINARY ==================
def upload_to_cloudinary(image_np, user="guest"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    temp_path = f"temp_{timestamp}.jpg"
    Image.fromarray(image_np).save(temp_path)

    result = cloudinary.uploader.upload(
        temp_path,
        folder=f"cashew/{user}",
        public_id=timestamp
    )

    return result["secure_url"]

# ================== FALLBACK ==================
def generate_fallback(diseases):
    return f"""
📌 **Gợi ý xử lý (offline AI)**

- Bệnh phát hiện: {", ".join(diseases)}
- Cắt bỏ lá bị nhiễm nặng
- Phun thuốc sinh học (nano đồng, neem oil)
- Giữ vườn thông thoáng, tránh ẩm cao
- Theo dõi 5–7 ngày

👉 Xử lý sớm để tránh lan rộng
"""

# ================== UI ==================
st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌿 Cashew Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.caption("YOLOv8 + Gemini + Cloudinary + Google Sheets")

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Tùy chỉnh")

if GEMINI_OK:
    st.sidebar.success("🤖 Gemini: Online")
else:
    st.sidebar.warning("⚠️ Gemini: Offline")

conf_thres = 0.65

# ================== LOAD MODEL ==================
yolo_model = load_model()
#=======dich lai sang tieng viet =============
disease_vi = {
    "leaf miner": "Sâu vẽ bùa",
    "anthracnose": "Bệnh thán thư",
    "powdery mildew": "Bệnh phấn trắng",
    "healthy": "Lá khỏe mạnh"
}

# ================== UPLOAD ==================
uploaded_file = st.file_uploader("📤 Tải ảnh lá điều", type=["jpg","jpeg","png"])

if uploaded_file and yolo_model:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Ảnh gốc")
        st.image(image, width='stretch')

    # ===== YOLO =====
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

        detected_info = []

        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
        
            name_en = yolo_model.names[cls_id]
            name_vi = disease_vi.get(name_en, name_en)
        
            detected_info.append((name_en, name_vi, conf))
            detected = [vi for _, vi, _ in detected_info]
    
        st.warning("Phát hiện bệnh:")

        for en, vi, conf in detected_info:
            st.markdown(f"- 🦠 **{vi}** ({en}) → Độ tin cậy: **{conf*100:.1f}%**")

        # ===== GEMINI =====
        if st.button("✨ Nhận tư vấn AI"):
            with st.spinner("Đang phân tích..."):
                if GEMINI_OK:
                    try:
                        prompt = f"""
                        Bạn là chuyên gia nông nghiệp.

                        Cây điều bị: {', '.join(detected)}

                        1. Nguyên nhân
                        2. Cách xử lý
                        3. Phòng ngừa

                        Trả lời ngắn gọn dạng bullet.
                        """

                        response = gemini_model.generate_content(prompt)
                        st.success("📊 Kết quả AI")
                        st.write(response.text)

                    except Exception as e:
                        st.error("🚫 Lỗi Gemini → dùng fallback")
                        st.info(generate_fallback(detected))
                else:
                    st.info(generate_fallback(detected))

        # ===== SAVE =====
        if st.button("💾 Lưu kết quả"):
            with st.spinner("Đang lưu..."):
                image_url = upload_to_cloudinary(result_img)

                save_log(
                    user="guest",
                    diseases=[f"{vi} ({conf*100:.1f}%)" for _, vi, conf in detected_info],
                    image_url=image_url
                )

            st.success("✅ Đã lưu thành công!")
            st.markdown("[🔗 Mở Google Sheet](https://docs.google.com/spreadsheets/d/1OdFoEcEkpB0S468oEAH7jU31bZLAnc2m_OHMXvjHtTY/edit?usp=sharing)")

    else:
        st.success("✅ Không phát hiện bệnh")

# ================== FOOTER ==================
st.markdown("---")
st.caption("YOLOv8 + Gemini 2.5 Flash • Deploy bằng Streamlit Cloud")
