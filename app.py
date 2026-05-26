import streamlit as st
from PIL import Image
import google.generativeai as genai
from backend import predict_image, load_model
import cloudinary
import cloudinary.uploader
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import hashlib
import re
import os
from datetime import datetime
import pytz

# ================== CONFIG PAGE ==================
st.set_page_config(page_title="Cashew Leaf Disease Detection", layout="wide")
st.markdown("""
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="manage-app-button"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# ================== HELPER ==================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_gsheet_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp"], scope
    )
    return gspread.authorize(creds)

def extract_url(cell_value):
    if not cell_value:
        return ""
    cell_value = str(cell_value).strip()
    if cell_value.startswith("http"):
        return cell_value
    match = re.search(r'=IMAGE\(["\'](.+?)["\']', cell_value, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def safe_get(row, idx):
    if len(row) <= idx:
        return ""
    val = row[idx]
    if val is None:
        return ""
    return str(val).strip()

# ================== USER FUNCTIONS ==================
def load_users():
    try:
        client = get_gsheet_client()
        sheet = client.open("cashew_log").worksheet("users")
        return sheet.get_all_records()
    except Exception as e:
        st.error(f"❌ Lỗi tải user: {e}")
        return []

def verify_login(username, password):
    users = load_users()
    pw_hash = hash_password(password)
    for u in users:
        if u["username"] == username and u["password"] == pw_hash:
            if u.get("status", "active") != "active":
                return False, None, "locked"
            return True, u.get("role", "user"), "ok"
    return False, None, "wrong"

def register_user(username, password):
    try:
        users = load_users()
        for u in users:
            if u["username"] == username:
                return False, "Username đã tồn tại"
        client = get_gsheet_client()
        sheet = client.open("cashew_log").worksheet("users")
        sheet.append_row([
            username,
            hash_password(password),
            "user",
            "active",
            datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S"),
        ])
        return True, "Đăng ký thành công"
    except Exception as e:
        return False, f"Lỗi: {e}"

def get_all_users():
    try:
        client = get_gsheet_client()
        sheet = client.open("cashew_log").worksheet("users")
        return sheet, sheet.get_all_records()
    except:
        return None, []

def get_all_logs():
    try:
        client = get_gsheet_client()
        sheet = client.open("cashew_log").sheet1
        return sheet.get_all_records()
    except:
        return []

# ================== INIT SESSION ==================
for key, default in {
    "logged_in": False,
    "username": "",
    "role": "",
    "page": "login",
    "last_treatment": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ================== AUTH PAGES ==================
def show_login():
    st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌿 Cashew Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888'>Hệ thống nhận diện bệnh lá điều</p>", unsafe_allow_html=True)
    st.divider()

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.subheader("🔐 Đăng nhập")
        with st.form("login_form"):
            username = st.text_input("👤 Tên đăng nhập")
            password = st.text_input("🔑 Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)

        if submitted:
            if not username or not password:
                st.warning("⚠️ Vui lòng nhập đầy đủ thông tin")
            else:
                with st.spinner("Đang xác thực..."):
                    ok, role, reason = verify_login(username, password)
                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = role
                    st.session_state["page"] = "admin" if role == "admin" else "app"
                    st.rerun()
                elif reason == "locked":
                    st.error("🔒 Tài khoản đã bị khóa. Liên hệ admin.")
                else:
                    st.error("❌ Sai tên đăng nhập hoặc mật khẩu")

        st.markdown("---")
        st.markdown("<p style='text-align:center'>Chưa có tài khoản?</p>", unsafe_allow_html=True)
        if st.button("📝 Đăng ký ngay", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()

def show_register():
    st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌿 Cashew Disease Detection</h1>", unsafe_allow_html=True)
    st.divider()

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.subheader("📝 Đăng ký tài khoản")
        with st.form("register_form"):
            new_user = st.text_input("👤 Tên đăng nhập")
            new_pass = st.text_input("🔑 Mật khẩu", type="password")
            confirm_pass = st.text_input("🔑 Xác nhận mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng ký", use_container_width=True)

        if submitted:
            if not new_user or not new_pass or not confirm_pass:
                st.warning("⚠️ Vui lòng nhập đầy đủ thông tin")
            elif len(new_user) < 3:
                st.warning("⚠️ Tên đăng nhập tối thiểu 3 ký tự")
            elif len(new_pass) < 6:
                st.warning("⚠️ Mật khẩu tối thiểu 6 ký tự")
            elif new_pass != confirm_pass:
                st.error("❌ Mật khẩu không khớp")
            else:
                with st.spinner("Đang đăng ký..."):
                    ok, msg = register_user(new_user, new_pass)
                if ok:
                    st.success(f"✅ {msg}! Vui lòng đăng nhập.")
                    import time; time.sleep(1.5)
                    st.session_state["page"] = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown("---")
        if st.button("← Quay lại đăng nhập", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()

# ================== SIDEBAR ==================
def show_sidebar():
    st.sidebar.markdown(f"👤 **{st.session_state['username']}**")
    st.sidebar.markdown(f"🏷️ Role: `{st.session_state['role']}`")
    st.sidebar.divider()

    if st.session_state["role"] == "admin":
        menu = st.sidebar.radio("📋 Menu", [
            "🔍 Detect bệnh",
            "📜 Lịch sử của tôi",
            "👥 Quản lý user",
            "📊 Xem tất cả log"
        ])
    else:
        menu = st.sidebar.radio("📋 Menu", [
            "🔍 Detect bệnh",
            "📜 Lịch sử của tôi"
        ])

    page_map = {
        "🔍 Detect bệnh":     "app",
        "📜 Lịch sử của tôi": "history",
        "👥 Quản lý user":    "admin_users",
        "📊 Xem tất cả log":  "admin_logs"
    }
    st.session_state["page"] = page_map[menu]

    st.sidebar.divider()
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.clear()
        st.rerun()

# ================== HISTORY ==================
def show_history():
    st.title("📜 Lịch sử phát hiện của tôi")
    current_user = st.session_state["username"]

    try:
        client = get_gsheet_client()
        sheet = client.open("cashew_log").sheet1
        formatted = sheet.get_all_values(value_render_option='FORMATTED_VALUE')
        formula   = sheet.get_all_values(value_render_option='FORMULA')
    except Exception as e:
        st.error(f"❌ Không thể tải lịch sử: {e}")
        return

    if len(formatted) <= 1:
        st.info("📭 Chưa có lịch sử nào.")
        return

    rows_fmt = formatted[1:]
    rows_fml = formula[1:]

    COL_TIME      = 0
    COL_USER      = 1
    COL_DISEASE   = 2
    COL_TREATMENT = 3
    COL_IMAGE     = 4

    if st.session_state["role"] == "admin":
        indices = list(range(len(rows_fmt)))
        st.caption("👑 Admin — đang xem tất cả lịch sử")
    else:
        indices = [i for i, r in enumerate(rows_fmt) if safe_get(r, COL_USER) == current_user]

    if not indices:
        st.info("📭 Chưa có lịch sử nào. Hãy detect và lưu kết quả!")
        return

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search = st.text_input("🔍 Tìm theo tên bệnh", placeholder="Nhập tên bệnh...")
    with col_f2:
        sort_order = st.selectbox("Sắp xếp", ["Mới nhất", "Cũ nhất"])

    if search:
        indices = [i for i in indices if search.lower() in safe_get(rows_fmt[i], COL_DISEASE).lower()]

    if sort_order == "Mới nhất":
        indices = list(reversed(indices))

    st.markdown(f"**Tổng: {len(indices)} bản ghi**")
    st.divider()

    if not indices:
        st.warning("Không tìm thấy kết quả phù hợp.")
        return

    for i in indices:
        row_fmt = rows_fmt[i]
        row_fml = rows_fml[i] if i < len(rows_fml) else []

        time_str    = safe_get(row_fmt, COL_TIME)
        disease_str = safe_get(row_fmt, COL_DISEASE)
        treatment   = safe_get(row_fmt, COL_TREATMENT)
        log_user    = safe_get(row_fmt, COL_USER)
        raw_image   = safe_get(row_fml, COL_IMAGE)
        image_url   = extract_url(raw_image)

        label = f"🕐 {time_str}  |  🦠 {disease_str}" if time_str and disease_str else "🕐 (bản ghi trống)"

        with st.expander(label):
            col_img, col_info = st.columns([1, 1.5])

            with col_img:
                if image_url:
                    st.image(image_url, caption="Ảnh đã phát hiện", use_container_width=True)
                else:
                    st.info("📷 Không có ảnh")

            with col_info:
                if st.session_state["role"] == "admin":
                    st.markdown(f"👤 **User:** {log_user}")
                st.markdown("🦠 **Bệnh phát hiện:**")
                for d in disease_str.split(", "):
                    if d.strip():
                        st.markdown(f"  - {d.strip()}")
                st.markdown("---")
                if treatment:
                    st.markdown("💊 **Cách điều trị (AI):**")
                    st.markdown(treatment)
                else:
                    st.caption("_(Không có tư vấn AI cho lần này)_")

# ================== ADMIN - QUẢN LÝ USER ==================
def show_admin_users():
    st.title("👥 Quản lý người dùng")
    sheet, users = get_all_users()

    if not users:
        st.info("Không có user nào.")
        return

    st.markdown(f"**Tổng: {len(users)} tài khoản**")
    st.divider()

    for i, u in enumerate(users):
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
        with col1:
            st.markdown(f"👤 **{u['username']}**")
        with col2:
            st.markdown(f"📅 {u.get('created_at', 'N/A')}")
        with col3:
            status = u.get("status", "active")
            if status == "active":
                st.success("🟢 Active")
            else:
                st.error("🔴 Locked")
        with col4:
            if u["username"] != st.session_state["username"]:
                row_num = i + 2
                if status == "active":
                    if st.button("🔒 Khóa", key=f"lock_{i}"):
                        sheet.update_cell(row_num, 4, "locked")
                        st.rerun()
                else:
                    if st.button("🔓 Mở", key=f"unlock_{i}"):
                        sheet.update_cell(row_num, 4, "active")
                        st.rerun()
            else:
                st.markdown("*(bạn)*")

# ================== ADMIN - XEM TẤT CẢ LOG ==================
def show_admin_logs():
    st.title("📊 Tất cả lịch sử phát hiện")
    logs = get_all_logs()

    if not logs:
        st.info("Chưa có log nào.")
        return

    all_users = list(set(l.get("user", "") for l in logs))
    filter_user = st.selectbox("🔍 Lọc theo user", ["Tất cả"] + sorted(all_users))

    if filter_user != "Tất cả":
        logs = [l for l in logs if l.get("user") == filter_user]

    st.markdown(f"**Tổng: {len(logs)} bản ghi**")
    st.divider()

    for log in reversed(logs):
        with st.expander(f"🕐 {log.get('time', '')} — 👤 {log.get('user', '')}"):
            st.markdown(f"🦠 **Bệnh:** {log.get('diseases', log.get('disease', ''))}")
            treatment = log.get("treatment", "")
            if treatment:
                st.markdown("💊 **Điều trị:**")
                st.markdown(treatment)

# ================== SETUP CLOUD ==================
def setup_cloud():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        gemini_ok = True
    except:
        gemini_model = None
        gemini_ok = False

    cloudinary.config(
        cloud_name=st.secrets["CLOUD_NAME"],
        api_key=st.secrets["API_KEY"],
        api_secret=st.secrets["API_SECRET"]
    )
    return gemini_model, gemini_ok

# ================== MAIN APP ==================
def show_app():
    gemini_model, GEMINI_OK = setup_cloud()

    st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌿 Cashew Leaf Disease Detection</h1>", unsafe_allow_html=True)
    st.caption("YOLOv8 + Gemini + Cloudinary + Google Sheets")

    st.sidebar.divider()
    if GEMINI_OK:
        st.sidebar.success("🤖 Gemini: Online")
    else:
        st.sidebar.warning("⚠️ Gemini: Offline")

    conf_thres = 0.65
    yolo_model = load_model()

    disease_vi = {
        "leaf miner":     "Sâu vẽ bùa",
        "anthracnose":    "Bệnh thán thư",
        "powdery mildew": "Bệnh phấn trắng",
        "healthy":        "Lá khỏe mạnh"
    }

    def generate_fallback(diseases):
        return (
            f"📌 **Gợi ý xử lý (offline AI)**\n\n"
            f"- Bệnh phát hiện: {', '.join(diseases)}\n"
            f"- Cắt bỏ lá bị nhiễm nặng\n"
            f"- Phun thuốc sinh học (nano đồng, neem oil)\n"
            f"- Giữ vườn thông thoáng, tránh ẩm cao\n"
            f"- Theo dõi 5–7 ngày\n\n"
            f"👉 Xử lý sớm để tránh lan rộng"
        )

    def upload_to_cloudinary(image_np, user="guest"):
        timestamp = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S")
        temp_path = f"temp_{timestamp}.jpg"
        try:
            Image.fromarray(image_np).save(temp_path)
            result = cloudinary.uploader.upload(
                temp_path,
                folder=f"cashew/{user}",
                public_id=timestamp
            )
            return result["secure_url"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def save_log(user, diseases, image_url, treatment=""):
        client = get_gsheet_client()
        sheet = client.open("cashew_log").sheet1
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user,
            ", ".join(diseases),
            treatment,
            image_url,                      # cột E: URL thuần
            f'=IMAGE("{image_url}";3)'      # cột F: hiển thị ảnh trong sheet
        ], value_input_option="USER_ENTERED")

    # ===== UPLOAD =====
    uploaded_file = st.file_uploader("📤 Tải ảnh lá điều", type=["jpg", "jpeg", "png"])

    if uploaded_file and yolo_model:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Ảnh gốc")
            st.image(image, width='stretch')

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
                cls_id  = int(b.cls[0])
                conf    = float(b.conf[0])
                name_en = yolo_model.names[cls_id]
                name_vi = disease_vi.get(name_en, name_en)
                detected_info.append((name_en, name_vi, conf))

            all_healthy = all(en == "healthy" for en, _, _ in detected_info)

            if all_healthy:
                # ===== HEALTHY: tự động lưu, không tư vấn AI =====
                st.success("✅ Lá khỏe mạnh!")
                for en, vi, conf in detected_info:
                    st.markdown(f"- 🌿 **{vi}** ({en}) → Độ tin cậy: **{conf*100:.1f}%**")

                with st.spinner("💾 Đang tự động lưu..."):
                    image_url = upload_to_cloudinary(result_img, user=st.session_state["username"])
                    save_log(
                        user=st.session_state["username"],
                        diseases=[f"{vi} ({conf*100:.1f}%)" for _, vi, conf in detected_info],
                        image_url=image_url,
                        treatment=""
                    )
                st.info("📁 Đã tự động lưu vào lịch sử")

            else:
                # ===== CÓ BỆNH =====
                st.warning("⚠️ Phát hiện bệnh:")
                for en, vi, conf in detected_info:
                    icon = "🌿" if en == "healthy" else "🦠"
                    st.markdown(f"- {icon} **{vi}** ({en}) → Độ tin cậy: **{conf*100:.1f}%**")

                # Chỉ tư vấn các bệnh, bỏ qua healthy
                sick_only = [vi for en, vi, _ in detected_info if en != "healthy"]

                if st.button("✨ Nhận tư vấn AI"):
                    with st.spinner("Đang phân tích..."):
                        if GEMINI_OK:
                            try:
                                prompt = f"""
Bạn là chuyên gia nông nghiệp.
Cây điều bị: {', '.join(sick_only)}
1. Nguyên nhân
2. Cách xử lý
3. Phòng ngừa
Trả lời ngắn gọn dạng bullet.
"""
                                response = gemini_model.generate_content(prompt)
                                treatment_text = response.text
                                st.session_state["last_treatment"] = treatment_text
                                st.success("📊 Kết quả AI")
                                st.write(treatment_text)
                            except Exception as e:
                                st.error("🚫 Lỗi Gemini → dùng fallback")
                                fallback = generate_fallback(sick_only)
                                st.session_state["last_treatment"] = fallback
                                st.info(fallback)
                        else:
                            fallback = generate_fallback(sick_only)
                            st.session_state["last_treatment"] = fallback
                            st.info(fallback)

                # ===== LƯU THỦ CÔNG =====
                if st.button("💾 Lưu kết quả"):
                    with st.spinner("Đang lưu..."):
                        image_url = upload_to_cloudinary(result_img, user=st.session_state["username"])
                        save_log(
                            user=st.session_state["username"],
                            diseases=[f"{vi} ({conf*100:.1f}%)" for _, vi, conf in detected_info],
                            image_url=image_url,
                            treatment=st.session_state.get("last_treatment", "")
                        )
                    st.success("✅ Đã lưu thành công!")
                    st.session_state["last_treatment"] = ""

        else:
            # ===== KHÔNG PHÁT HIỆN =====
            st.error("❌ Không phải lá điều hoặc ảnh không rõ — vui lòng thử lại với ảnh khác")

    st.markdown("---")
    st.caption("YOLOv8 + Gemini 2.5 Flash • Deploy bằng Streamlit Cloud")

# ================== ROUTER (luôn ở cuối) ==================
page = st.session_state["page"]

if not st.session_state["logged_in"]:
    if page == "register":
        show_register()
    else:
        show_login()
else:
    show_sidebar()
    current_page = st.session_state["page"]
    if current_page in ("app", "admin"):
        show_app()
    elif current_page == "history":
        show_history()
    elif current_page == "admin_users":
        show_admin_users()
    elif current_page == "admin_logs":
        show_admin_logs()
