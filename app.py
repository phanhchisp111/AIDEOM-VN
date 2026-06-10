import streamlit as st
import importlib

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="AIDEOM-VN | Mô hình ra quyết định",
    page_icon="📊",
    layout="wide"
)

# =========================
# BACKGROUND VIDEO + GLASS STYLE
# =========================
st.markdown("""
<style>
/* Toàn bộ app nền trong suốt để hiện video */
[data-testid="stAppViewContainer"] {
    background: transparent;
}

.stApp {
    background: transparent;
    color: #ffffff;
}

/* Video background full màn hình */
.video-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -3;
    overflow: hidden;
    background: #000000;
}

.video-background video {
    position: absolute;
    top: 0;
    left: 50%;
    width: 120%;
    height: 120%;
    transform: translateX(-50%);
    object-fit: cover;
    object-position: top center;
}

/* Lớp phủ tối để chữ dễ đọc */
.video-overlay {
    position: fixed;
    inset: 0;
    z-index: -2;
    background:
        radial-gradient(circle at top, rgba(255,255,255,0.08), transparent 35%),
        linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.82));
}

/* Header mặc định của Streamlit */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Sidebar kính mờ */
section[data-testid="stSidebar"] {
    background: rgba(5, 10, 25, 0.70);
    border-right: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Text */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    letter-spacing: -0.03em;
}

p, li, label, span, div {
    color: #f8fafc;
}

/* Main container */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

/* Card kính */
.glass-card {
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 28px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 22px 70px rgba(0,0,0,0.42);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* Card cho AI Agent */
.agent-card {
    background: rgba(255,255,255,0.085);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 28px;
    padding: 28px;
    margin-top: 22px;
    box-shadow: 0 22px 70px rgba(0,0,0,0.42);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}

/* Metric card */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.085);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.32);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

[data-testid="stMetricLabel"] {
    color: #dbeafe !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Tabs dạng pill */
button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.085);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    margin-right: 8px;
    padding: 8px 16px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

button[data-baseweb="tab"] p {
    color: #ffffff !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.18);
}

/* Alert box */
[data-testid="stAlert"] {
    border-radius: 20px;
}

/* Slider text */
.stSlider label {
    color: #ffffff !important;
}

/* Radio sidebar item */
div[role="radiogroup"] label {
    border-radius: 14px;
    padding: 4px 8px;
}

/* Markdown link */
a {
    color: #bae6fd !important;
}

/* Làm bảng trên trang chủ dễ nhìn hơn */
.home-table {
    background: rgba(255,255,255,0.06);
    border-radius: 24px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.16);
}

/* Sửa màu chữ trong selectbox, input, file uploader */
div[data-baseweb="select"] * {
    color: #111827 !important;
}

div[data-baseweb="popover"] * {
    color: #111827 !important;
}

input {
    color: #111827 !important;
}

textarea {
    color: #111827 !important;
}

/* File uploader dễ nhìn hơn */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 12px;
}

section[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* Student info box */
.student-info-box {
    margin-top: 14px;
    margin-bottom: 18px;
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.18);
    line-height: 1.55;
    font-size: 0.92rem;
}
.student-info-box b {
    color: #ffffff !important;
}

</style>

<div class="video-background">
    <video autoplay muted loop playsinline>
        <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260418_080021_d598092b-c4c2-4e53-8e46-94cf9064cd50.mp4" type="video/mp4">
    </video>
</div>
<div class="video-overlay"></div>
""", unsafe_allow_html=True)


# =========================
# HÀM GỌI MODULE
# =========================
def run_module(module_name, bai_label):
    """
    Gọi module theo tên.
    Mỗi module trong thư mục modules cần có hàm run().
    Ví dụ: modules/bai1_cobb_douglas.py cần có def run():
    """
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)

        if hasattr(module, "run"):
            module.run()
        else:
            st.error(f"{bai_label} chưa có hàm run().")
            st.info(f"Hãy kiểm tra file {module_name.replace('.', '/')}.py có dòng: def run(): hay chưa.")

    except ModuleNotFoundError:
        st.markdown(f"""
        <div class="glass-card">
            <h1>{bai_label}</h1>
            <p>
            Module của bài này chưa được tạo trong thư mục <b>modules</b>.
            </p>
            <p>
            Tên file cần có: <b>{module_name.replace("modules.", "")}.py</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("Bạn cần tạo file module tương ứng trước, sau đó refresh web.")

    except Exception as e:
        st.error(f"Có lỗi khi tải {bai_label}.")
        st.exception(e)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")

st.sidebar.markdown("""
<div class="student-info-box">
<b>👩‍🎓 Sinh viên thực hiện</b><br>
<b>Họ và tên:</b> Lê Phương Anh<br>
<b>MSV:</b> 23050412<br>
<b>Lớp:</b> QH 2023-E KTPT5
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Chọn bài thực hành",
    [
        "Trang chủ",
        "Bài 1 – Cobb-Douglas + AI",
        "Bài 2 – LP ngân sách số",
        "Bài 3 – Priority 10 ngành",
        "Bài 4 – LP ngành-vùng",
        "Bài 5 – MIP 15 dự án",
        "Bài 6 – TOPSIS 6 vùng",
        "Bài 7 – NSGA-II Pareto",
        "Bài 8 – Tối ưu động",
        "Bài 9 – Lao động & AI",
        "Bài 10 – Stochastic LP",
        "Bài 11 – Q-learning",
        "Bài 12 – Dashboard tích hợp",
    ]
)


# =========================
# TRANG CHỦ
# =========================
if menu == "Trang chủ":
    st.markdown("""
    <div class="glass-card">
        <h1>📊 AIDEOM-VN</h1>
        <h3>Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI</h3>
        <p><b>Sinh viên thực hiện:</b> Lê Phương Anh &nbsp; | &nbsp; <b>MSV:</b> 23050412 &nbsp; | &nbsp; <b>Lớp:</b> QH 2023-E KTPT5</p>
        <p>
        Web app này được xây dựng cho học phần <b>Các mô hình ra quyết định</b>.
        Hệ thống gồm 12 bài thực hành, cho phép người dùng xem mô hình, chạy tính toán,
        quan sát bảng kết quả, biểu đồ và phần tác nhân phân tích kết quả.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cấp độ dễ", "Bài 1–3")
    with col2:
        st.metric("Trung bình", "Bài 4–6")
    with col3:
        st.metric("Khá khó", "Bài 7–9")
    with col4:
        st.metric("Khó", "Bài 10–12")

    st.markdown("""
    <div class="glass-card">
        <h3>🧭 Danh sách 12 bài thực hành</h3>
        <p>
        Các bài được tổ chức theo mức độ khó tăng dần, từ hàm sản xuất, quy hoạch tuyến tính,
        ra quyết định đa tiêu chí đến tối ưu động, mô phỏng ngẫu nhiên và học tăng cường.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        {
            "Bài": [
                "Bài 1", "Bài 2", "Bài 3", "Bài 4", "Bài 5", "Bài 6",
                "Bài 7", "Bài 8", "Bài 9", "Bài 10", "Bài 11", "Bài 12"
            ],
            "Nội dung": [
                "Cobb-Douglas mở rộng với AI và số hóa",
                "Quy hoạch tuyến tính phân bổ ngân sách số",
                "Tính chỉ số ưu tiên ngành",
                "LP phân bổ ngân sách ngành-vùng",
                "MIP lựa chọn dự án chuyển đổi số",
                "TOPSIS xếp hạng vùng",
                "Tối ưu đa mục tiêu Pareto",
                "Tối ưu động",
                "Mô phỏng lao động và AI",
                "Stochastic LP",
                "Q-learning",
                "Dashboard tích hợp"
            ],
            "Kết quả chính": [
                "TFP, MAPE, GDP dự báo 2030",
                "Phân bổ ngân sách tối ưu",
                "Xếp hạng 10 ngành",
                "Ma trận phân bổ vùng",
                "Danh mục dự án được chọn",
                "Xếp hạng 6 vùng",
                "Đường biên Pareto",
                "Quỹ đạo đầu tư",
                "Việc làm bị thay thế/tạo mới",
                "Kết quả theo kịch bản",
                "Q-table và chính sách tối ưu",
                "Tổng hợp kết quả"
            ]
        },
        use_container_width=True
    )


# =========================
# GỌI 12 MODULE
# =========================
elif menu == "Bài 1 – Cobb-Douglas + AI":
    run_module("modules.bai1_cobb_douglas", "Bài 1 – Cobb-Douglas + AI")

elif menu == "Bài 2 – LP ngân sách số":
    run_module("modules.bai2_lp_ngan_sach", "Bài 2 – LP ngân sách số")

elif menu == "Bài 3 – Priority 10 ngành":
    run_module("modules.bai3_priority_sectors", "Bài 3 – Priority 10 ngành")

elif menu == "Bài 4 – LP ngành-vùng":
    run_module("modules.bai4_lp_nganh_vung", "Bài 4 – LP ngành-vùng")

elif menu == "Bài 5 – MIP 15 dự án":
    run_module("modules.bai5_mip_15_du_an", "Bài 5 – MIP 15 dự án")

elif menu == "Bài 6 – TOPSIS 6 vùng":
    run_module("modules.bai6_topsis_6_vung", "Bài 6 – TOPSIS 6 vùng")

elif menu == "Bài 7 – NSGA-II Pareto":
    run_module("modules.bai7_nsga2_pareto", "Bài 7 – NSGA-II Pareto")

elif menu == "Bài 8 – Tối ưu động":
    run_module("modules.bai8_toi_uu_dong", "Bài 8 – Tối ưu động")

elif menu == "Bài 9 – Lao động & AI":
    run_module("modules.bai9_lao_dong_ai", "Bài 9 – Lao động & AI")

elif menu == "Bài 10 – Stochastic LP":
    run_module("modules.bai10_stochastic_lp", "Bài 10 – Stochastic LP")

elif menu == "Bài 11 – Q-learning":
    run_module("modules.bai11_q_learning", "Bài 11 – Q-learning")

elif menu == "Bài 12 – Dashboard tích hợp":
    run_module("modules.bai12_dashboard_tich_hop", "Bài 12 – Dashboard tích hợp")
