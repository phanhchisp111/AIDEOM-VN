import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# =========================================================
# BÀI 9 – TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG VIỆT NAM
# BẢN ĐÚNG THEO TỪNG CÂU GIẢNG VIÊN HỎI
# - 9.4.1: mô hình gốc đúng đề
# - 9.4.2: ngưỡng x_H tối thiểu riêng cho ngành 2
# - 9.4.3: mô phỏng nhóm dễ tổn thương riêng
# - 9.4.4: mô hình mở rộng riêng với ràng buộc 5%
# - 9.5: thảo luận chính sách riêng
# Không gộp kịch bản mở rộng vào nghiệm chính.
# =========================================================


SECTORS = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo"
]

N = 8
LABOR_MILLION = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100

# Tham số đúng theo gợi ý đề
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])     # NewJob từ AI
A2 = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 18.5])     # NewJob từ D, hiển thị tham số
B1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])    # UpgradeJob từ H
C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])     # DisplacedJob từ AI*risk
D1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])    # RetrainingCapacity từ H

BUDGET = 30000.0


def make_data():
    df = pd.DataFrame({
        "Ngành": SECTORS,
        "Lao động (triệu)": LABOR_MILLION,
        "Risk": RISK,
        "a1 NewJob AI": A1,
        "a2 NewJob D": A2,
        "b1 Upgrade H": B1,
        "c1 Displaced AI": C1,
        "d1 Retrain H": D1
    })
    df["AI tạo việc ròng/tỷ trước đào tạo"] = A1 - C1 * RISK
    df["Displaced/tỷ AI"] = C1 * RISK
    return df


def build_result(x_ai, x_h):
    new_job = A1 * x_ai
    upgrade = B1 * x_h
    displaced = C1 * RISK * x_ai
    retrain = D1 * x_h
    netjob = new_job + upgrade - displaced

    out = pd.DataFrame({
        "Ngành": SECTORS,
        "Lao động (triệu)": LABOR_MILLION,
        "Risk": RISK,
        "x_AI": x_ai,
        "x_H": x_h,
        "NewJob": new_job,
        "UpgradeJob": upgrade,
        "DisplacedJob": displaced,
        "RetrainingCapacity": retrain,
        "NetJob": netjob
    })
    out["Tổng đầu tư"] = out["x_AI"] + out["x_H"]
    out["Displaced / Lao động"] = out["DisplacedJob"] / (out["Lao động (triệu)"] * 1_000_000)
    out["NetJob / Lao động"] = out["NetJob"] / (out["Lao động (triệu)"] * 1_000_000)
    return out


def solve_cvxpy_original(extra_5pct=False):
    if not CVXPY_AVAILABLE:
        return None, "Chưa cài CVXPY"

    x_ai = cp.Variable(N, nonneg=True)
    x_h = cp.Variable(N, nonneg=True)

    new_job = cp.multiply(A1, x_ai)
    upgrade = cp.multiply(B1, x_h)
    displaced = cp.multiply(cp.multiply(C1, RISK), x_ai)
    retrain = cp.multiply(D1, x_h)
    netjob = new_job + upgrade - displaced

    constraints = [
        cp.sum(x_ai + x_h) <= BUDGET,
        netjob >= 0,
        displaced <= retrain
    ]

    if extra_5pct:
        max_displaced = 0.05 * LABOR_MILLION * 1_000_000
        constraints.append(displaced <= max_displaced)

    prob = cp.Problem(cp.Maximize(cp.sum(netjob)), constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception as e:
            return None, f"Lỗi CVXPY: {e}"

    if x_ai.value is None:
        return None, prob.status

    return build_result(np.maximum(x_ai.value, 0), np.maximum(x_h.value, 0)), prob.status


def solve_scipy_original(extra_5pct=False):
    if not SCIPY_AVAILABLE:
        return None, "Chưa cài scipy"

    net_ai_coef = A1 - C1 * RISK
    c = -np.concatenate([net_ai_coef, B1])

    A_ub = []
    b_ub = []

    # Ngân sách
    A_ub.append(np.ones(2 * N))
    b_ub.append(BUDGET)

    # NetJob_i >= 0
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = -net_ai_coef[i]
        row[N + i] = -B1[i]
        A_ub.append(row)
        b_ub.append(0)

    # Displaced_i <= RetrainingCapacity_i
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = C1[i] * RISK[i]
        row[N + i] = -D1[i]
        A_ub.append(row)
        b_ub.append(0)

    # Mở rộng 5%
    if extra_5pct:
        max_displaced = 0.05 * LABOR_MILLION * 1_000_000
        for i in range(N):
            row = np.zeros(2 * N)
            row[i] = C1[i] * RISK[i]
            A_ub.append(row)
            b_ub.append(max_displaced[i])

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * (2 * N),
        method="highs"
    )

    if not res.success:
        return None, res.message

    return build_result(res.x[:N], res.x[N:]), "Optimal bằng scipy.linprog"


def solve_original(extra_5pct=False):
    result, status = solve_cvxpy_original(extra_5pct=extra_5pct)
    if result is not None:
        return result, status, "CVXPY"

    result2, status2 = solve_scipy_original(extra_5pct=extra_5pct)
    if result2 is not None:
        return result2, status2, "scipy.linprog fallback"

    return None, f"{status}; {status2}", "Không có solver"


def summary(result, status, solver):
    if result is None:
        return pd.DataFrame({"Chỉ tiêu": ["Trạng thái"], "Giá trị": [status]})

    return pd.DataFrame({
        "Chỉ tiêu": [
            "Solver",
            "Trạng thái",
            "Tổng ngân sách sử dụng",
            "Tổng x_AI",
            "Tổng x_H",
            "Tổng NewJob",
            "Tổng UpgradeJob",
            "Tổng DisplacedJob",
            "Tổng NetJob"
        ],
        "Giá trị": [
            solver,
            status,
            f"{result['Tổng đầu tư'].sum():,.2f}",
            f"{result['x_AI'].sum():,.2f}",
            f"{result['x_H'].sum():,.2f}",
            f"{result['NewJob'].sum():,.0f}",
            f"{result['UpgradeJob'].sum():,.0f}",
            f"{result['DisplacedJob'].sum():,.0f}",
            f"{result['NetJob'].sum():,.0f}"
        ]
    })


def manufacturing_threshold(x_ai_value):
    idx = 1

    displaced_per_ai = C1[idx] * RISK[idx]
    net_ai_coef = A1[idx] - displaced_per_ai

    # Điều kiện NetJob >= 0:
    # (a1 - c1*risk)xAI + b1*xH >= 0
    if net_ai_coef >= 0:
        h_for_netjob = 0.0
    else:
        h_for_netjob = (-net_ai_coef / B1[idx]) * x_ai_value

    # Điều kiện Displaced <= RetrainCap:
    # c1*risk*xAI <= d1*xH
    h_for_retrain = (displaced_per_ai / D1[idx]) * x_ai_value

    h_min = max(h_for_netjob, h_for_retrain)

    return pd.DataFrame({
        "Điều kiện": [
            "NetJob₂ ≥ 0",
            "DisplacedJob₂ ≤ RetrainingCapacity₂",
            "Ngưỡng cuối cùng"
        ],
        "Công thức": [
            "x_H ≥ max(0, -(a1-c1*risk)/b1 × x_AI)",
            "x_H ≥ (c1*risk/d1) × x_AI",
            "Lấy max của hai điều kiện"
        ],
        "x_AI ngành 2": [x_ai_value, x_ai_value, x_ai_value],
        "x_H tối thiểu": [h_for_netjob, h_for_retrain, h_min]
    })


def vulnerable_simulation(result):
    vulnerable_idx = [0, 2, 3]
    rows = []
    for i in vulnerable_idx:
        r = result.iloc[i]
        rows.append({
            "Ngành": r["Ngành"],
            "Lao động phổ thông ban đầu": r["Lao động (triệu)"] * 1_000_000,
            "Bị tự động hóa dịch chuyển": r["DisplacedJob"],
            "Được đào tạo lại": min(r["DisplacedJob"], r["RetrainingCapacity"]),
            "Việc làm nâng cấp": r["UpgradeJob"],
            "Việc làm AI mới": r["NewJob"],
            "NetJob": r["NetJob"]
        })
    return pd.DataFrame(rows)


def plot_investment(result):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(result))
    width = 0.35
    ax.bar(x - width / 2, result["x_AI"], width, label="x_AI")
    ax.bar(x + width / 2, result["x_H"], width, label="x_H")
    ax.set_xticks(x)
    ax.set_xticklabels(result["Ngành"], rotation=25, ha="right")
    ax.set_title("Phân bổ tối ưu x_AI và x_H theo ngành")
    ax.set_ylabel("Tỷ VND")
    ax.grid(axis="y", alpha=0.4)
    ax.legend()
    return fig


def plot_netjob(result):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = result.sort_values("NetJob")
    ax.barh(plot_df["Ngành"], plot_df["NetJob"])
    ax.set_title("NetJob ròng theo ngành")
    ax.set_xlabel("Việc làm")
    ax.grid(axis="x", alpha=0.4)
    for i, v in enumerate(plot_df["NetJob"]):
        ax.text(v, i, f"{v:,.0f}", va="center")
    return fig


def plot_vulnerable_sankey(result):
    """
    9.4.3: Mô phỏng nhóm lao động dễ bị tổn thương.
    Bản cũ gom tất cả ngành vào một luồng 'Đào tạo lại -> Việc làm nâng cấp' nên Sankey bị thành một mảng xám rất to.
    Bản này tách từng ngành thành từng lane riêng và quy đổi đơn vị sang triệu lao động để dễ đọc.
    """
    flow = vulnerable_simulation(result).copy()

    # Quy đổi sang triệu lao động để Sankey không bị khối quá lớn
    for col in [
        "Lao động phổ thông ban đầu",
        "Bị tự động hóa dịch chuyển",
        "Được đào tạo lại",
        "Việc làm nâng cấp",
        "Việc làm AI mới",
        "NetJob"
    ]:
        flow[col] = flow[col] / 1_000_000

    if PLOTLY_AVAILABLE:
        labels = []
        source, target, value, link_label = [], [], [], []

        # Tạo node riêng cho từng ngành để không bị gộp thành một khối xám
        for idx, row in flow.iterrows():
            sector = row["Ngành"]

            n_start = len(labels); labels.append(f"{sector}<br>LĐ phổ thông")
            n_displaced = len(labels); labels.append(f"{sector}<br>Bị tự động hóa")
            n_retrain = len(labels); labels.append(f"{sector}<br>Đào tạo lại")
            n_upgrade = len(labels); labels.append(f"{sector}<br>Việc làm nâng cấp")
            n_newai = len(labels); labels.append(f"{sector}<br>Việc làm AI mới")
            n_net = len(labels); labels.append(f"{sector}<br>NetJob ròng")

            displaced = max(float(row["Bị tự động hóa dịch chuyển"]), 0.0001)
            retrained = max(float(row["Được đào tạo lại"]), 0.0001)
            upgraded = max(float(row["Việc làm nâng cấp"]), 0.0001)
            new_ai = max(float(row["Việc làm AI mới"]), 0.0001)
            net = max(float(row["NetJob"]), 0.0001)

            source += [n_start, n_displaced, n_retrain, n_start, n_upgrade, n_newai]
            target += [n_displaced, n_retrain, n_upgrade, n_newai, n_net, n_net]
            value += [displaced, retrained, upgraded, new_ai, upgraded, new_ai]
            link_label += [
                "Bị tự động hóa",
                "Được đào tạo lại",
                "Chuyển sang việc làm nâng cấp",
                "Việc làm AI mới",
                "Đóng góp vào NetJob",
                "Đóng góp vào NetJob"
            ]

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=22,
                thickness=16,
                line=dict(width=0.5)
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                label=link_label
            )
        )])
        fig.update_layout(
            title_text="Swimming lane/Sankey: nhóm lao động dễ bị tổn thương theo từng ngành (triệu lao động)",
            height=680,
            font_size=11
        )
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(flow))
    width = 0.22
    ax.bar(x - width, flow["Bị tự động hóa dịch chuyển"], width, label="Bị tự động hóa")
    ax.bar(x, flow["Được đào tạo lại"], width, label="Đào tạo lại")
    ax.bar(x + width, flow["NetJob"], width, label="NetJob")
    ax.set_xticks(x)
    ax.set_xticklabels(flow["Ngành"], rotation=20, ha="right")
    ax.set_title("Nhóm lao động dễ bị tổn thương theo ngành")
    ax.set_ylabel("Triệu lao động")
    ax.grid(axis="y", alpha=0.4)
    ax.legend()
    return fig



def run():
    st.title("👷 Bài 9 – Tác động AI tới thị trường lao động Việt Nam")

    st.write("""
    Module này tách riêng từng yêu cầu của giảng viên: mô hình gốc, ngưỡng đào tạo ngành 2,
    mô phỏng nhóm dễ bị tổn thương, mô hình mở rộng 5%, và thảo luận chính sách. 
    Phần mở rộng không bị gộp vào nghiệm chính.
    """)

    base_result, base_status, base_solver = solve_original(extra_5pct=False)
    extended_result, extended_status, extended_solver = solve_original(extra_5pct=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 Mô hình",
        "📋 Dữ liệu",
        "9.4.1 Mô hình gốc",
        "9.4.2 Ngưỡng ngành 2",
        "9.4.3–9.4.4 Mở rộng riêng",
        "🤖 9.5 Chính sách"
    ])

    with tab1:
        st.subheader("9.1. Bối cảnh Việt Nam")
        st.write("""
        AI có thể làm một phần việc làm bị tự động hóa, nhưng đồng thời tạo việc làm mới và nâng cấp kỹ năng.
        Bài toán đặt ra là cần phân bổ ngân sách đào tạo lại và đầu tư AI như thế nào để NetJob ròng không âm.
        """)

        st.subheader("9.2. Mô hình toán học")
        st.latex(r"NetJob_i = NewJob^{AI}_i + UpgradeJob_i - DisplacedJob^{Automation}_i")
        st.latex(r"NewJob_i = a_{1i}x^{AI}_i + a_{2i}x^D_i")
        st.latex(r"UpgradeJob_i = b_{1i}x^H_i")
        st.latex(r"DisplacedJob_i = c_{1i}x^{AI}_i risk_i")
        st.latex(r"RetrainingCapacity_i = d_{1i}x^H_i")

        st.markdown("""
        **Mô hình tối ưu gốc 9.4.1:**
        - Maximize: \(\sum_i NetJob_i\)
        - \(\sum_i(x^{AI}_i+x^H_i)\leq 30.000\)
        - \(NetJob_i\geq 0\)
        - \(DisplacedJob_i\leq RetrainingCapacity_i\)
        - \(x^{AI}_i,x^H_i\geq 0\)
        """)

        if CVXPY_AVAILABLE:
            st.success("Đã có CVXPY. Mô hình chính dùng CVXPY.")
        elif SCIPY_AVAILABLE:
            st.warning("Chưa có CVXPY nên đang dùng scipy.linprog fallback. Nên cài: python -m pip install cvxpy")
        else:
            st.error("Chưa có solver. Cài: python -m pip install cvxpy scipy")

        if not PLOTLY_AVAILABLE:
            st.info("Muốn Sankey đẹp hơn thì cài thêm: python -m pip install plotly")

    with tab2:
        st.subheader("9.3. Tham số 8 ngành Việt Nam")
        st.dataframe(make_data(), use_container_width=True, height=430)
        st.info("Dữ liệu được hiển thị trực tiếp trên web, không có upload.")

    with tab3:
        st.subheader("Câu 9.4.1 – Mô hình gốc đúng đề")

        if base_result is None:
            st.error(base_status)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Solver", base_solver)
            c2.metric("Trạng thái", base_status)
            c3.metric("Tổng NetJob", f"{base_result['NetJob'].sum():,.0f}")
            c4.metric("Ngân sách dùng", f"{base_result['Tổng đầu tư'].sum():,.0f}")

            st.markdown("### Bảng tóm tắt")
            st.dataframe(summary(base_result, base_status, base_solver), use_container_width=True)

            st.markdown("### Phân bổ tối ưu")
            st.dataframe(
                base_result[[
                    "Ngành", "x_AI", "x_H", "NewJob", "UpgradeJob",
                    "DisplacedJob", "RetrainingCapacity", "NetJob",
                    "Displaced / Lao động"
                ]],
                use_container_width=True,
                height=430
            )

            st.warning(
                "Lưu ý khi đọc kết quả: đây là mô hình LP gốc đúng đề. Nếu nghiệm dồn nhiều vào một ngành/hạng mục, "
                "đó là nghiệm góc của bài toán tuyến tính, không phải lỗi code. Phần 9.4.4 mới là mô hình mở rộng để kiểm tra ràng buộc an sinh."
            )

            fig1 = plot_investment(base_result)
            st.pyplot(fig1)
            plt.close(fig1)

            fig2 = plot_netjob(base_result)
            st.pyplot(fig2)
            plt.close(fig2)

    with tab4:
        st.subheader("Câu 9.4.2 – Ngưỡng đầu tư đào tạo tối thiểu ngành 2")

        x_ai_value = st.slider(
            "Giả định mức đầu tư AI tối đa vào ngành 2",
            min_value=1000,
            max_value=30000,
            value=30000,
            step=1000
        )

        th = manufacturing_threshold(x_ai_value)
        st.dataframe(th, use_container_width=True)

        h_min = th["x_H tối thiểu"].iloc[-1]
        st.success(
            f"Với x_AI ngành 2 = {x_ai_value:,.0f} tỷ, ngành chế biến chế tạo cần x_H tối thiểu khoảng {h_min:,.0f} tỷ."
        )

        st.write("""
        Kết quả này cho thấy trong ngành chế biến chế tạo, vấn đề chính không chỉ là NetJob ròng,
        mà còn là năng lực đào tạo lại phải đủ lớn để hấp thụ số lao động bị tự động hóa dịch chuyển.
        """)

    with tab5:
        st.subheader("Câu 9.4.3 – Nhóm dễ bị tổn thương ngành 1, 3, 4")

        if base_result is None:
            st.error("Chưa có nghiệm gốc để mô phỏng.")
        else:
            vulnerable = vulnerable_simulation(base_result)
            st.dataframe(vulnerable, use_container_width=True)

            fig3 = plot_vulnerable_sankey(base_result)
            if PLOTLY_AVAILABLE:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.pyplot(fig3)
                plt.close(fig3)

        st.subheader("Câu 9.4.4 – Mô hình mở rộng riêng: không ngành nào mất quá 5% lao động")

        if extended_result is None:
            st.error(f"Mô hình mở rộng không khả thi hoặc lỗi solver: {extended_status}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái mở rộng", extended_status)
            c2.metric("Tổng NetJob mở rộng", f"{extended_result['NetJob'].sum():,.0f}")
            c3.metric("Ngân sách dùng", f"{extended_result['Tổng đầu tư'].sum():,.0f}")

            st.dataframe(
                extended_result[[
                    "Ngành", "x_AI", "x_H", "DisplacedJob",
                    "Displaced / Lao động", "NetJob"
                ]],
                use_container_width=True,
                height=430
            )

            compare = pd.DataFrame({
                "Kịch bản": ["Gốc 9.4.1", "Mở rộng 9.4.4"],
                "Trạng thái": [base_status, extended_status],
                "Tổng NetJob": [
                    base_result["NetJob"].sum() if base_result is not None else np.nan,
                    extended_result["NetJob"].sum()
                ],
                "Tổng x_AI": [
                    base_result["x_AI"].sum() if base_result is not None else np.nan,
                    extended_result["x_AI"].sum()
                ],
                "Tổng x_H": [
                    base_result["x_H"].sum() if base_result is not None else np.nan,
                    extended_result["x_H"].sum()
                ],
                "Max displaced/lao động": [
                    base_result["Displaced / Lao động"].max() if base_result is not None else np.nan,
                    extended_result["Displaced / Lao động"].max()
                ]
            })
            st.markdown("### So sánh mô hình gốc và mô hình mở rộng")
            st.dataframe(compare, use_container_width=True)

    with tab6:
        st.subheader("🤖 Câu hỏi thảo luận chính sách 9.5")

        if base_result is None:
            st.error("Chưa có nghiệm để phân tích.")
            return

        most_h = base_result.sort_values("x_H", ascending=False).iloc[0]
        finance = base_result.loc[base_result["Ngành"] == "Tài chính-Ngân hàng"].iloc[0]
        agriculture = base_result.loc[base_result["Ngành"] == "Nông-Lâm-Thủy sản"].iloc[0]

        with st.container(border=True):
            st.markdown("#### a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?")
            st.write(
                f"Theo mô hình gốc, ngành cần đầu tư đào tạo lại nhiều nhất là **{most_h['Ngành']}**, "
                f"với x_H khoảng **{most_h['x_H']:,.0f} tỷ VND**."
            )
            st.write(
                "Nếu kết quả tập trung mạnh vào một ngành, cần hiểu đây là đặc điểm của mô hình tuyến tính: "
                "ngân sách có xu hướng chảy vào nơi có hệ số tạo việc làm biên cao nhất. Về chính sách, nên bổ sung ràng buộc phân bổ tối thiểu "
                "cho các ngành nhiều lao động phổ thông để thực tế hơn."
            )

        with st.container(border=True):
            st.markdown("#### b) Tài chính-Ngân hàng risk 52% nhưng tạo việc mới cao")
            st.write(
                f"Tài chính-Ngân hàng có x_AI = **{finance['x_AI']:,.0f}** và x_H = **{finance['x_H']:,.0f}**."
            )
            st.write(
                "Mô hình gợi ý không nên né tránh AI trong ngành này, vì AI có thể tạo việc làm mới có kỹ năng cao. "
                "Tuy nhiên, tự động hóa phải đi kèm đào tạo lại để chuyển lao động từ nghiệp vụ lặp lại sang phân tích dữ liệu, quản trị rủi ro, "
                "an ninh mạng và dịch vụ tài chính số."
            )

        with st.container(border=True):
            st.markdown("#### c) Có nên đầu tư AI vào Nông-Lâm-Thủy sản không?")
            st.write(
                f"Nông-Lâm-Thủy sản có x_AI = **{agriculture['x_AI']:,.0f}** và NetJob = **{agriculture['NetJob']:,.0f}**."
            )
            st.write(
                "Do hệ số tạo việc làm AI của nông nghiệp thấp nhưng quy mô lao động lớn, đầu tư AI trong ngành này cần thận trọng. "
                "Nên ưu tiên AI hỗ trợ năng suất, khuyến nông số, truy xuất nguồn gốc và dịch vụ nông nghiệp, đồng thời tăng đào tạo lại "
                "để tránh dịch chuyển lao động quy mô lớn."
            )

        with st.container(border=True):
            st.markdown("#### d) Ràng buộc nào biểu diễn 'tốc độ tự động hóa không vượt quá năng lực đào tạo lại'?")
            st.latex(r"DisplacedJob_i \leq RetrainingCapacity_i")
            st.write(
                "Đây là ràng buộc trung tâm của bài. Nó bảo đảm số lao động bị dịch chuyển bởi tự động hóa trong mỗi ngành không vượt quá "
                "năng lực đào tạo lại của ngành đó."
            )
            st.write(
                "Có thể bổ sung ràng buộc an sinh như: DisplacedJobᵢ ≤ 5% lao động ngành, x_H tối thiểu cho ngành 1, 3, 4, "
                "hoặc quỹ hỗ trợ chuyển đổi nghề cho lao động phổ thông."
            )


if __name__ == "__main__":
    run()
