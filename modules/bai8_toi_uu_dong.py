import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# BÀI 8 – TỐI ƯU ĐỘNG PHÂN BỔ LIÊN THỜI GIAN 2026–2035
# Bám đề:
# 8.3.1 Giải NLP động bằng scipy.optimize.minimize SLSQP
# 8.3.2 Vẽ quỹ đạo K, D, AI, H, Y, C
# 8.3.3 Cú sốc Y_2028 giảm 8%
# 8.3.4 So sánh đầu tư trải đều và front-load
# 8.4 AI Agent trả lời a, b, c
# =========================================================


YEARS = np.arange(2026, 2036)
T = len(YEARS)

# Tham số đúng theo đề
ALPHA_K = 0.33
ALPHA_L = 0.42
ALPHA_D = 0.10
ALPHA_AI = 0.08
ALPHA_H = 0.07

DELTA_K = 0.05
DELTA_D = 0.12
DELTA_AI = 0.15

THETA_H = 0.8
MU = 0.02

PHI1 = 0.003
PHI2 = 0.002
PHI3 = 0.004

RHO_DEFAULT = 0.97
GAMMA_CRRA = 1.5

# Điều kiện ban đầu theo đề
K0 = 27500.0
L0 = 53.9
D0 = 20.3
AI0 = 86.0
H0 = 30.0

# A0 hiệu chỉnh để Y_2026 có quy mô hợp lý theo Bài 1.
# Vì D, AI, H là chỉ số khác đơn vị, A0 được hiệu chỉnh để Y ban đầu khoảng 13.000 nghìn tỷ VND.
TARGET_Y0 = 13000.0
A0 = TARGET_Y0 / (K0**ALPHA_K * L0**ALPHA_L * D0**ALPHA_D * AI0**ALPHA_AI * H0**ALPHA_H)

# ---------------------------------------------------------
# Lưu ý đơn vị:
# Y, K, I_K, I_D, I_AI, I_H tính theo nghìn tỷ VND.
# Nhưng D, AI, H là chỉ số/năng lực, không cùng đơn vị với tiền.
# Nếu cộng trực tiếp I_D vào D như bản nháp, D/AI/H sẽ phình cực lớn
# và làm Y nổ từ 2031 trở đi. Vì vậy cần hệ số chuyển đổi đầu tư -> chỉ số.
# Đây là cách diễn giải thực tế hơn của phương trình động học trong đề.
# ---------------------------------------------------------
INV_TO_INDEX = 1000.0
TFP_SCALE = 100.0
MIN_CONSUMPTION_SHARE = 0.22
MAX_TOTAL_INVEST_SHARE = 0.74


def production(A, K, L, D, AI, H):
    K = max(K, 1e-8)
    D = max(D, 1e-8)
    AI = max(AI, 1e-8)
    H = max(H, 1e-8)
    L = max(L, 1e-8)
    return A * (K ** ALPHA_K) * (L ** ALPHA_L) * (D ** ALPHA_D) * (AI ** ALPHA_AI) * (H ** ALPHA_H)


def utility(C, utility_type="log"):
    C = np.maximum(np.array(C, dtype=float), 1e-8)
    if utility_type == "CRRA":
        return (C ** (1 - GAMMA_CRRA)) / (1 - GAMMA_CRRA)
    return np.log(C)


def unpack_decision(z):
    """
    z gồm 5T biến:
    s_K, s_D, s_AI, s_H, s_C theo từng năm.
    Các share được chuẩn hóa mềm để tổng C + đầu tư <= Y.
    """
    z = np.array(z, dtype=float).reshape(T, 5)
    return z


def normalize_shares(raw):
    """
    Chuẩn hóa tỷ trọng phân bổ.
    Cột 0-3 là đầu tư K, D, AI, H; cột 4 là tiêu dùng C.

    Để nghiệm không phi thực tế, đặt tiêu dùng tối thiểu 22% Y
    và tổng đầu tư tối đa 74% Y. Phần còn lại là đệm ngân sách.
    """
    raw = np.maximum(raw, 1e-8)
    inv_raw = raw[:, :4]
    c_raw = raw[:, 4]

    inv_sum = inv_raw.sum(axis=1, keepdims=True)
    inv_mix = inv_raw / np.where(inv_sum == 0, 1, inv_sum)

    # tỷ trọng đầu tư tổng biến thiên trong [0.35, 0.74]
    invest_intensity = 0.35 + 0.39 * (c_raw / (1 + c_raw))
    invest_intensity = np.minimum(invest_intensity, MAX_TOTAL_INVEST_SHARE)

    inv_shares = inv_mix * invest_intensity.reshape(-1, 1)
    c_share = np.maximum(MIN_CONSUMPTION_SHARE, 1 - inv_shares.sum(axis=1))

    shares = np.column_stack([inv_shares, c_share])

    # nếu tổng > 0.98 thì co đầu tư lại, giữ C tối thiểu
    total = shares.sum(axis=1)
    over = total > 0.98
    if np.any(over):
        for i in np.where(over)[0]:
            allowed_inv = 0.98 - shares[i, 4]
            if allowed_inv < 0:
                allowed_inv = 0.0
            current_inv = shares[i, :4].sum()
            if current_inv > 0:
                shares[i, :4] = shares[i, :4] / current_inv * allowed_inv

    return shares


def simulate_path(z, rho=RHO_DEFAULT, utility_type="log", shock=False, strategy_name="optimized"):
    shares = normalize_shares(unpack_decision(z))

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)
    A = np.zeros(T + 1)
    L = np.zeros(T + 1)

    I_K = np.zeros(T)
    I_D = np.zeros(T)
    I_AI = np.zeros(T)
    I_H = np.zeros(T)
    C = np.zeros(T)
    Y = np.zeros(T)
    welfare_terms = np.zeros(T)

    K[0] = K0
    D[0] = D0
    AI[0] = AI0
    H[0] = H0
    A[0] = A0
    L[0] = L0

    for t in range(T):
        # Lao động tăng nhẹ 0,3%/năm
        if t > 0:
            L[t] = L[t-1] * 1.003

        Y_t = production(A[t], K[t], L[t], D[t], AI[t], H[t])

        # Cú sốc năm 2028, tức index t=2
        if shock and YEARS[t] == 2028:
            Y_t = Y_t * 0.92

        Y[t] = Y_t

        sK, sD, sAI, sH, sC = shares[t]

        I_K[t] = sK * Y_t
        I_D[t] = sD * Y_t
        I_AI[t] = sAI * Y_t
        I_H[t] = sH * Y_t
        C[t] = max(sC * Y_t, 1e-8)

        K[t+1] = (1 - DELTA_K) * K[t] + I_K[t]

        # D, AI, H là chỉ số/năng lực, nên đầu tư tiền tệ được đổi sang điểm chỉ số
        D[t+1] = (1 - DELTA_D) * D[t] + I_D[t] / INV_TO_INDEX
        AI[t+1] = (1 - DELTA_AI) * AI[t] + I_AI[t] / INV_TO_INDEX
        H[t+1] = H[t] + THETA_H * I_H[t] / INV_TO_INDEX - MU * H[t]

        # Hiệu ứng TFP dùng D, AI, H đã chuẩn hóa để tránh bùng nổ số học
        A_growth = 1 + PHI1 * (D[t] / TFP_SCALE) + PHI2 * (AI[t] / TFP_SCALE) + PHI3 * (H[t] / TFP_SCALE)
        A[t+1] = A[t] * A_growth

        if t < T - 1:
            L[t+1] = L[t] * 1.003

        welfare_terms[t] = (rho ** t) * utility(C[t], utility_type=utility_type)

    welfare = welfare_terms.sum()

    path_df = pd.DataFrame({
        "year": YEARS,
        "Y": Y,
        "C": C,
        "I_K": I_K,
        "I_D": I_D,
        "I_AI": I_AI,
        "I_H": I_H,
        "share_K": shares[:, 0],
        "share_D": shares[:, 1],
        "share_AI": shares[:, 2],
        "share_H": shares[:, 3],
        "share_C": shares[:, 4],
        "welfare_term": welfare_terms
    })

    state_df = pd.DataFrame({
        "year": np.arange(2026, 2037),
        "K": K,
        "D": D,
        "AI": AI,
        "H": H,
        "A": A,
        "L": L
    })

    return {
        "strategy": strategy_name,
        "welfare": float(welfare),
        "path": path_df,
        "state": state_df,
        "shares": shares
    }


def objective_slsqp(z, rho=RHO_DEFAULT, utility_type="log", shock=False):
    sim = simulate_path(z, rho=rho, utility_type=utility_type, shock=shock)
    # minimize âm welfare
    return -sim["welfare"]


def initial_guess(kind="balanced"):
    if kind == "front_load":
        arr = np.zeros((T, 5))
        for t in range(T):
            if t <= 2:
                arr[t] = [0.23, 0.18, 0.17, 0.18, 0.20]
            elif t <= 5:
                arr[t] = [0.18, 0.13, 0.12, 0.16, 0.37]
            else:
                arr[t] = [0.12, 0.09, 0.08, 0.12, 0.55]
        return arr.flatten()

    if kind == "even":
        arr = np.tile(np.array([0.16, 0.14, 0.12, 0.16, 0.38]), (T, 1))
        return arr.flatten()

    arr = np.tile(np.array([0.17, 0.13, 0.11, 0.15, 0.40]), (T, 1))
    return arr.flatten()


@st.cache_data(show_spinner=False)
def solve_optimized(rho=RHO_DEFAULT, utility_type="log", shock=False):
    if not SCIPY_AVAILABLE:
        return None, "Chưa cài scipy"

    z0 = initial_guess("balanced")

    bounds = [(1e-5, 1.0)] * (T * 5)

    res = minimize(
        objective_slsqp,
        z0,
        args=(rho, utility_type, shock),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 450,
            "ftol": 1e-7,
            "disp": False
        }
    )

    sim = simulate_path(res.x, rho=rho, utility_type=utility_type, shock=shock, strategy_name="Tối ưu SLSQP")
    return sim, res.message


def fixed_strategy(kind="even", rho=RHO_DEFAULT, utility_type="log", shock=False):
    z = initial_guess(kind)
    label = "Đầu tư trải đều" if kind == "even" else "Đầu tư front-load"
    return simulate_path(z, rho=rho, utility_type=utility_type, shock=shock, strategy_name=label)


def compare_paths(base, shock):
    base_path = base["path"].copy()
    shock_path = shock["path"].copy()

    compare = pd.DataFrame({
        "year": YEARS,
        "Y kế hoạch": base_path["Y"],
        "Y có shock": shock_path["Y"],
        "C kế hoạch": base_path["C"],
        "C có shock": shock_path["C"],
        "I_K kế hoạch": base_path["I_K"],
        "I_K có shock": shock_path["I_K"],
        "I_D kế hoạch": base_path["I_D"],
        "I_D có shock": shock_path["I_D"],
        "I_AI kế hoạch": base_path["I_AI"],
        "I_AI có shock": shock_path["I_AI"],
        "I_H kế hoạch": base_path["I_H"],
        "I_H có shock": shock_path["I_H"]
    })

    for col in ["Y", "C", "I_K", "I_D", "I_AI", "I_H"]:
        compare[f"Δ {col} shock - kế hoạch"] = compare[f"{col} có shock"] - compare[f"{col} kế hoạch"]

    return compare


def plot_line(df, x_col, y_cols, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in y_cols:
        ax.plot(df[x_col], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Năm")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4)
    ax.legend()
    return fig


def plot_investment_shares(path_df, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = ["share_K", "share_D", "share_AI", "share_H", "share_C"]
    labels = ["K", "D", "AI", "H", "C"]

    for col, lab in zip(cols, labels):
        ax.plot(path_df["year"], path_df[col], marker="o", label=lab)

    ax.set_title(title)
    ax.set_xlabel("Năm")
    ax.set_ylabel("Tỷ trọng trong Y")
    ax.grid(alpha=0.4)
    ax.legend()
    return fig


def bellman_demo():
    """
    Minh họa Bellman equation đơn giản với 1 trạng thái K và 1 quyết định saving rate.
    Không thay thế mô hình NLP đầy đủ, chỉ để so sánh tư duy quy hoạch động.
    """
    K_grid = np.linspace(15000, 60000, 60)
    s_grid = np.linspace(0.10, 0.45, 30)
    rho = RHO_DEFAULT
    A_simple = 3.0
    alpha = 0.35
    delta = 0.05

    V = np.zeros_like(K_grid)
    policy = np.zeros_like(K_grid)

    for _ in range(80):
        V_new = np.zeros_like(V)
        policy_new = np.zeros_like(policy)

        for i, K in enumerate(K_grid):
            best_val = -1e18
            best_s = s_grid[0]

            Y = A_simple * (K ** alpha) * (L0 ** (1 - alpha))

            for s in s_grid:
                C = max((1 - s) * Y, 1e-8)
                K_next = (1 - delta) * K + s * Y
                V_next = np.interp(K_next, K_grid, V)
                val = np.log(C) + rho * V_next

                if val > best_val:
                    best_val = val
                    best_s = s

            V_new[i] = best_val
            policy_new[i] = best_s

        V = V_new
        policy = policy_new

    return pd.DataFrame({
        "K": K_grid,
        "V(K)": V,
        "saving_rate_policy": policy
    })


def summary_metrics(sim):
    path = sim["path"]
    state = sim["state"]

    return pd.DataFrame({
        "Chỉ tiêu": [
            "Tổng welfare",
            "Y 2026",
            "Y 2035",
            "C trung bình",
            "K cuối kỳ",
            "D cuối kỳ",
            "AI cuối kỳ",
            "H cuối kỳ",
            "Tỷ lệ AI/H bình quân"
        ],
        "Giá trị": [
            f"{sim['welfare']:,.4f}",
            f"{path['Y'].iloc[0]:,.2f}",
            f"{path['Y'].iloc[-1]:,.2f}",
            f"{path['C'].mean():,.2f}",
            f"{state['K'].iloc[-1]:,.2f}",
            f"{state['D'].iloc[-1]:,.2f}",
            f"{state['AI'].iloc[-1]:,.2f}",
            f"{state['H'].iloc[-1]:,.2f}",
            f"{(path['I_AI'] / np.maximum(path['I_H'], 1e-8)).mean():.3f}"
        ]
    })


def run():
    st.title("⏳ Bài 8 – Tối ưu động phân bổ liên thời gian 2026–2035")

    st.write("""
    Bài 8 xây dựng mô hình tối ưu hóa động trong giai đoạn 2026–2035. Chính phủ lựa chọn phân bổ nguồn lực
    cho vốn vật chất K, hạ tầng số D, AI, vốn nhân lực H và tiêu dùng C nhằm tối đa hóa phúc lợi xã hội liên thời gian.
    """)

    with st.sidebar:
        st.markdown("### Bài 8 – thiết lập")
        utility_type = st.selectbox("Hàm thỏa dụng", ["log", "CRRA"], index=0)
        rho = st.slider("Hệ số chiết khấu ρ", 0.90, 0.99, RHO_DEFAULT, 0.01)

    base_opt, solve_msg = solve_optimized(rho=rho, utility_type=utility_type, shock=False)
    shock_opt, shock_msg = solve_optimized(rho=rho, utility_type=utility_type, shock=True)

    even_strategy = fixed_strategy("even", rho=rho, utility_type=utility_type, shock=False)
    front_strategy = fixed_strategy("front_load", rho=rho, utility_type=utility_type, shock=False)

    rho_short_opt, _ = solve_optimized(rho=0.90, utility_type=utility_type, shock=False)

    bellman_df = bellman_demo()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Mô hình",
        "8.3.1 SLSQP",
        "8.3.2 Quỹ đạo",
        "8.3.3–8.3.4 Shock & chiến lược",
        "🤖 8.4 Chính sách"
    ])

    with tab1:
        st.subheader("8.1. Bối cảnh Việt Nam")
        st.write("""
        Việt Nam đặt mục tiêu trở thành nước thu nhập trung bình cao vào năm 2030 và nước phát triển thu nhập cao vào năm 2045.
        Vì vậy, phân bổ vốn dài hạn cần cân bằng giữa tăng trưởng, chuyển đổi số, AI và nâng cao chất lượng nhân lực.
        """)

        st.subheader("8.2. Mô hình toán học")
        st.latex(r"\max \sum_{t=2026}^{2035} \rho^{t-2026} U(C_t)")
        st.latex(r"Y_t=A_tK_t^{0.33}L_t^{0.42}D_t^{0.10}AI_t^{0.08}H_t^{0.07}")
        st.latex(r"K_{t+1}=(1-\delta_K)K_t+I_{K,t}")
        st.latex(r"D_{t+1}=(1-\delta_D)D_t+I_{D,t}")
        st.latex(r"AI_{t+1}=(1-\delta_{AI})AI_t+I_{AI,t}")
        st.latex(r"H_{t+1}=H_t+\theta_H I_{H,t}-\mu H_t")
        st.latex(r"A_{t+1}=A_t(1+\phi_1D_t+\phi_2AI_t+\phi_3H_t)")
        st.latex(r"C_t+I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t}\leq Y_t")

        st.markdown("### Tham số sử dụng")
        param_df = pd.DataFrame({
            "Tham số": [
                "δK", "δD", "δAI", "θH", "μ", "φ1", "φ2", "φ3", "ρ", "K0", "L0", "D0", "AI0", "H0", "A0 hiệu chỉnh", "INV_TO_INDEX", "TFP_SCALE"
            ],
            "Giá trị": [
                DELTA_K, DELTA_D, DELTA_AI, THETA_H, MU, PHI1, PHI2, PHI3, rho, K0, L0, D0, AI0, H0, A0, INV_TO_INDEX, TFP_SCALE
            ]
        })
        st.dataframe(param_df, use_container_width=True)

        if SCIPY_AVAILABLE:
            st.success("Đã có scipy. Mô hình được giải bằng scipy.optimize.minimize phương pháp SLSQP.")
        else:
            st.error("Chưa cài scipy. Chạy: python -m pip install scipy")

    with tab2:
        st.subheader("Câu 8.3.1 – Giải NLP động bằng scipy.optimize.minimize SLSQP")

        st.write("""
        Do hàm Cobb-Douglas và phương trình TFP nội sinh tạo bài toán phi tuyến, module này dùng cách B trong đề:
        scipy.optimize.minimize với phương pháp SLSQP. Biến quyết định là tỷ trọng phân bổ hàng năm cho K, D, AI, H và C.

        Bản sửa đã xử lý vấn đề khác đơn vị: Y và đầu tư tính bằng nghìn tỷ VND, còn D, AI, H là chỉ số/năng lực.
        Vì vậy đầu tư vào D, AI, H được chuyển đổi sang điểm chỉ số bằng hệ số INV_TO_INDEX để tránh quỹ đạo nổ số.
        """)

        if base_opt is None:
            st.error(solve_msg)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái solver", solve_msg)
            c2.metric("Welfare tối ưu", f"{base_opt['welfare']:,.4f}")
            c3.metric("Y 2035", f"{base_opt['path']['Y'].iloc[-1]:,.2f}")

            st.markdown("### Tóm tắt nghiệm tối ưu")
            st.dataframe(summary_metrics(base_opt), use_container_width=True)

            st.markdown("### Bảng phân bổ tối ưu theo năm")
            st.dataframe(base_opt["path"], use_container_width=True, height=420)

            st.markdown("### Bellman equation minh họa")
            st.write("""
            Phần này minh họa tư duy Bellman với một trạng thái K và một quyết định tỷ lệ tiết kiệm.
            Mô hình Bellman đơn giản hơn mô hình NLP đầy đủ, nhưng cho thấy logic: giá trị hiện tại phụ thuộc vào lựa chọn hôm nay
            và giá trị trạng thái tương lai.
            """)
            st.dataframe(bellman_df.head(12), use_container_width=True)

            # Biểu đồ Bellman demo: trục X là mức vốn K, không phải năm.
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                bellman_df["K"],
                bellman_df["saving_rate_policy"],
                marker="o",
                label="Tỷ lệ tiết kiệm tối ưu"
            )
            ax.set_title("Chính sách tiết kiệm tối ưu theo mức vốn K")
            ax.set_xlabel("Vốn K")
            ax.set_ylabel("Tỷ lệ tiết kiệm")
            ax.set_ylim(
                max(0, bellman_df["saving_rate_policy"].min() - 0.01),
                bellman_df["saving_rate_policy"].max() + 0.01
            )
            ax.grid(alpha=0.4)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.info(
                "Phần Bellman chỉ là minh họa kỹ thuật tối ưu động. "
                "Trong bộ tham số demo, tỷ lệ tiết kiệm tối ưu gần như ổn định theo mức vốn K."
            )

    with tab3:
        st.subheader("Câu 8.3.2 – Quỹ đạo tối ưu của K, D, AI, H, Y, C")

        if base_opt is None:
            st.error("Chưa có nghiệm tối ưu.")
        else:
            state_df = base_opt["state"]
            path_df = base_opt["path"]

            fig1 = plot_line(state_df, "year", ["K", "D", "AI", "H"], "Quỹ đạo trạng thái K, D, AI, H", "Mức vốn/chỉ số")
            st.pyplot(fig1)
            plt.close(fig1)

            fig2 = plot_line(path_df, "year", ["Y", "C"], "Quỹ đạo sản lượng Y và tiêu dùng C", "Nghìn tỷ VND")
            st.pyplot(fig2)
            plt.close(fig2)

            fig3 = plot_line(path_df, "year", ["I_K", "I_D", "I_AI", "I_H"], "Quỹ đạo đầu tư theo hạng mục", "Nghìn tỷ VND")
            st.pyplot(fig3)
            plt.close(fig3)

            fig4 = plot_investment_shares(path_df, "Tỷ trọng phân bổ tối ưu theo thời gian")
            st.pyplot(fig4)
            plt.close(fig4)

    with tab4:
        st.subheader("Câu 8.3.3 – Cú sốc năm 2028 làm Y giảm 8%")

        if base_opt is None or shock_opt is None:
            st.error("Chưa có nghiệm để so sánh cú sốc.")
        else:
            compare_shock_df = compare_paths(base_opt, shock_opt)

            c1, c2, c3 = st.columns(3)
            c1.metric("Welfare kế hoạch", f"{base_opt['welfare']:,.4f}")
            c2.metric("Welfare có shock", f"{shock_opt['welfare']:,.4f}")
            c3.metric("Chênh lệch welfare", f"{shock_opt['welfare'] - base_opt['welfare']:,.4f}")

            st.dataframe(compare_shock_df, use_container_width=True, height=420)

            fig = plt.subplots(figsize=(9, 5))[0]
            ax = fig.axes[0]
            ax.plot(base_opt["path"]["year"], base_opt["path"]["Y"], marker="o", label="Y kế hoạch")
            ax.plot(shock_opt["path"]["year"], shock_opt["path"]["Y"], marker="o", label="Y có shock")
            ax.set_title("Tác động cú sốc 2028 đến sản lượng")
            ax.set_xlabel("Năm")
            ax.set_ylabel("Y")
            ax.grid(alpha=0.4)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Câu 8.3.4 – So sánh đầu tư trải đều và front-load")

        strategy_df = pd.DataFrame({
            "Chiến lược": ["Tối ưu SLSQP", "Đầu tư trải đều", "Đầu tư front-load"],
            "Welfare tổng": [
                base_opt["welfare"] if base_opt else np.nan,
                even_strategy["welfare"],
                front_strategy["welfare"]
            ],
            "Y 2035": [
                base_opt["path"]["Y"].iloc[-1] if base_opt else np.nan,
                even_strategy["path"]["Y"].iloc[-1],
                front_strategy["path"]["Y"].iloc[-1]
            ],
            "C trung bình": [
                base_opt["path"]["C"].mean() if base_opt else np.nan,
                even_strategy["path"]["C"].mean(),
                front_strategy["path"]["C"].mean()
            ]
        })

        st.dataframe(strategy_df, use_container_width=True)

        fig = plt.subplots(figsize=(9, 5))[0]
        ax = fig.axes[0]
        if base_opt:
            ax.plot(base_opt["path"]["year"], base_opt["path"]["Y"], marker="o", label="Tối ưu SLSQP")
        ax.plot(even_strategy["path"]["year"], even_strategy["path"]["Y"], marker="o", label="Trải đều")
        ax.plot(front_strategy["path"]["year"], front_strategy["path"]["Y"], marker="o", label="Front-load")
        ax.set_title("So sánh Y theo các chiến lược")
        ax.set_xlabel("Năm")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.4)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        st.subheader("🤖 Tác nhân phân tích chính sách 8.4")

        if base_opt is None:
            st.error("Chưa có nghiệm tối ưu để phân tích.")
            return

        path_df = base_opt["path"]
        early_invest = path_df.loc[path_df["year"] <= 2028, ["I_K", "I_D", "I_AI", "I_H"]].sum().sum()
        late_invest = path_df.loc[path_df["year"] >= 2033, ["I_K", "I_D", "I_AI", "I_H"]].sum().sum()
        pattern = "front-loaded" if early_invest > late_invest else "back-loaded"

        ai_h_ratio = path_df["I_AI"] / np.maximum(path_df["I_H"], 1e-8)
        ratio_std = ai_h_ratio.std()

        with st.container(border=True):
            st.markdown("#### a) Quỹ đạo tối ưu của K, D, AI, H có front-loaded hay back-loaded không?")
            st.write(
                f"Trong nghiệm hiện tại, quỹ đạo đầu tư có xu hướng **{pattern}**. "
                "Nếu đầu tư tập trung nhiều ở giai đoạn đầu, nguyên nhân là đầu tư vào K, D, AI và H tạo hiệu ứng tích lũy, "
                "làm tăng vốn và năng suất cho nhiều năm sau. Ngược lại, nếu mô hình tăng đầu tư về cuối kỳ, điều đó thường phản ánh ưu tiên làm trơn tiêu dùng hiện tại."
            )
            st.write(
                "Với hệ số chiết khấu cao, chính phủ coi trọng dài hạn nên mô hình có xu hướng chấp nhận hy sinh một phần tiêu dùng hiện tại "
                "để tích lũy năng lực sản xuất và năng lực số."
            )

        with st.container(border=True):
            st.markdown("#### b) Tỷ lệ đầu tư AI/H có ổn định không? Nhân lực nên đi trước hay đồng thời với AI?")
            st.write(
                f"Tỷ lệ AI/H bình quân là **{ai_h_ratio.mean():.3f}**, độ lệch chuẩn là **{ratio_std:.3f}**. "
                "Nếu tỷ lệ này biến động, điều đó cho thấy mô hình không xem AI và nhân lực là hai khoản đầu tư thay thế đơn giản, "
                "mà cần điều chỉnh theo giai đoạn phát triển."
            )
            st.write(
                "Về chính sách, đào tạo nhân lực nên đi trước hoặc ít nhất đi đồng thời với đầu tư AI. "
                "Nếu chỉ đầu tư AI mà thiếu nhân lực số, nền kinh tế khó hấp thụ công nghệ, rủi ro vận hành và chảy máu chất xám cũng cao hơn."
            )

        with st.container(border=True):
            st.markdown("#### c) Nếu ρ = 0,90 thì kết quả thay đổi thế nào? Có phải lý do chính phủ thường dưới đầu tư vào R&D không?")
            if rho_short_opt is not None:
                st.write(
                    f"Khi giảm ρ xuống 0,90, welfare tối ưu là **{rho_short_opt['welfare']:,.4f}** "
                    f"và Y năm 2035 là **{rho_short_opt['path']['Y'].iloc[-1]:,.2f}**. "
                    f"Trong khi đó, với ρ hiện tại, Y năm 2035 là **{base_opt['path']['Y'].iloc[-1]:,.2f}**."
                )
            st.write(
                "ρ thấp hơn nghĩa là chính phủ coi trọng hiện tại hơn tương lai. Khi đó mô hình thường tăng tiêu dùng trước mắt "
                "và giảm đầu tư dài hạn vào D, AI, H hoặc R&D. Đây là một lý do kinh tế giải thích vì sao các chính phủ có áp lực nhiệm kỳ "
                "có thể dưới đầu tư vào R&D: lợi ích xuất hiện muộn, trong khi chi phí ngân sách phát sinh ngay."
            )


if __name__ == "__main__":
    run()
