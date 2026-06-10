import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except Exception:
    PYOMO_AVAILABLE = False


# =========================================================
# BÀI 10 – QUY HOẠCH NGẪU NHIÊN HAI GIAI ĐOẠN
# Bám đúng đề:
# 10.5.1: two-stage stochastic programming, first-stage x, second-stage y_s
# 10.5.2: deterministic từng kịch bản + EV solution
# 10.5.3: VSS và EVPI
# 10.5.4: robust optimization cực tiểu hóa regret xấu nhất
# =========================================================


J = ["I", "D", "AI", "H"]
J_NAME = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số",
    "AI": "Trí tuệ nhân tạo",
    "H": "Nhân lực số"
}

S = ["s1", "s2", "s3", "s4"]
S_NAME = {
    "s1": "Lạc quan",
    "s2": "Cơ sở",
    "s3": "Bi quan",
    "s4": "Khủng hoảng"
}

PROB = {
    "s1": 0.30,
    "s2": 0.45,
    "s3": 0.20,
    "s4": 0.05
}

SCENARIO_INFO = {
    "s1": {"Tăng trưởng TG (%)": 3.5, "FDI VN": 32.0, "Xuất khẩu tăng (%)": 12.0, "Xác suất": 0.30},
    "s2": {"Tăng trưởng TG (%)": 2.8, "FDI VN": 27.0, "Xuất khẩu tăng (%)": 8.0, "Xác suất": 0.45},
    "s3": {"Tăng trưởng TG (%)": 1.5, "FDI VN": 20.0, "Xuất khẩu tăng (%)": 3.0, "Xác suất": 0.20},
    "s4": {"Tăng trưởng TG (%)": 0.2, "FDI VN": 12.0, "Xuất khẩu tăng (%)": -5.0, "Xác suất": 0.05},
}

BETA = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}

BETA_S = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}

BUDGET_STAGE1 = 65000.0
BUDGET_STAGE2 = 15000.0
TOTAL_BUDGET = 80000.0


def scenario_df():
    rows = []
    for s in S:
        row = {"Kịch bản": s, "Tên": S_NAME[s]}
        row.update(SCENARIO_INFO[s])
        rows.append(row)
    return pd.DataFrame(rows)


def beta_df():
    rows = []
    for j in J:
        row = {
            "Hạng mục": j,
            "Tên": J_NAME[j],
            "β cơ bản": BETA[j],
        }
        for s in S:
            row[f"β {s} - {S_NAME[s]}"] = BETA_S[(s, j)]
        rows.append(row)
    return pd.DataFrame(rows)


def expected_beta_s():
    return {j: sum(PROB[s] * BETA_S[(s, j)] for s in S) for j in J}


def build_solution_df(x, y):
    first = pd.DataFrame({
        "Hạng mục": J,
        "Tên": [J_NAME[j] for j in J],
        "x_j first-stage": [x[j] for j in J],
        "β cơ bản": [BETA[j] for j in J],
        "Lợi ích first-stage": [BETA[j] * x[j] for j in J]
    })

    rows = []
    for s in S:
        for j in J:
            rows.append({
                "Kịch bản": s,
                "Tên kịch bản": S_NAME[s],
                "Hạng mục": j,
                "Tên hạng mục": J_NAME[j],
                "p_s": PROB[s],
                "y_sj second-stage": y[(s, j)],
                "β_sj": BETA_S[(s, j)],
                "Lợi ích recourse": BETA_S[(s, j)] * y[(s, j)],
                "Lợi ích kỳ vọng recourse": PROB[s] * BETA_S[(s, j)] * y[(s, j)]
            })
    second = pd.DataFrame(rows)

    return first, second


def payoff_given_xy(x, y, weighted=True):
    first = sum(BETA[j] * x[j] for j in J)
    if weighted:
        second = sum(PROB[s] * sum(BETA_S[(s, j)] * y[(s, j)] for j in J) for s in S)
    else:
        second = sum(sum(BETA_S[(s, j)] * y[(s, j)] for j in J) for s in S)
    return first + second


def solve_sp_scipy(fixed_x=None, beta_second=None):
    """
    Giải stochastic program:
    max sum beta*x + sum_s p_s sum beta_s*y_s
    constraints:
        sum x <= 65000
        sum_j y_sj <= 15000 for all s
        y_s,AI <= 0.5*x_H
        x,y >=0

    Nếu fixed_x khác None: chỉ tối ưu recourse y_s theo x cố định.
    Nếu beta_second khác None: dùng hệ số recourse trung bình cho tất cả scenario.
    """
    if not SCIPY_AVAILABLE:
        return None, "Chưa cài scipy"

    n_x = len(J)
    n_y = len(S) * len(J)
    n = n_x + n_y

    def ix(j):
        return J.index(j)

    def iy(s, j):
        return n_x + S.index(s) * len(J) + J.index(j)

    c = np.zeros(n)

    # linprog minimize, nên dùng -objective
    for j in J:
        c[ix(j)] = -BETA[j]

    for s in S:
        for j in J:
            if beta_second is None:
                coef = PROB[s] * BETA_S[(s, j)]
            else:
                coef = PROB[s] * beta_second[j]
            c[iy(s, j)] = -coef

    A_ub = []
    b_ub = []

    # stage 1 budget
    row = np.zeros(n)
    for j in J:
        row[ix(j)] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE1)

    # stage 2 budgets
    for s in S:
        row = np.zeros(n)
        for j in J:
            row[iy(s, j)] = 1
        A_ub.append(row)
        b_ub.append(BUDGET_STAGE2)

    # AI recourse capacity: y_s,AI <= 0.5*x_H -> y_s,AI - 0.5*x_H <= 0
    for s in S:
        row = np.zeros(n)
        row[iy(s, "AI")] = 1
        row[ix("H")] = -0.5
        A_ub.append(row)
        b_ub.append(0)

    # fixed x if needed
    bounds = [(0, None)] * n
    if fixed_x is not None:
        for j in J:
            val = fixed_x[j]
            bounds[ix(j)] = (val, val)

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        return None, res.message

    x = {j: res.x[ix(j)] for j in J}
    y = {(s, j): res.x[iy(s, j)] for s in S for j in J}
    obj = -res.fun

    return {"x": x, "y": y, "objective": obj, "status": "Optimal"}, "Optimal"


def solve_single_scenario(s, fixed_x=None):
    """
    Wait-and-see hoặc deterministic riêng theo một kịch bản.
    max beta*x + beta_s*y_s
    """
    if not SCIPY_AVAILABLE:
        return None, "Chưa cài scipy"

    n_x = len(J)
    n_y = len(J)
    n = n_x + n_y

    def ix(j):
        return J.index(j)

    def iy(j):
        return n_x + J.index(j)

    c = np.zeros(n)
    for j in J:
        c[ix(j)] = -BETA[j]
        c[iy(j)] = -BETA_S[(s, j)]

    A_ub = []
    b_ub = []

    row = np.zeros(n)
    for j in J:
        row[ix(j)] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE1)

    row = np.zeros(n)
    for j in J:
        row[iy(j)] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE2)

    row = np.zeros(n)
    row[iy("AI")] = 1
    row[ix("H")] = -0.5
    A_ub.append(row)
    b_ub.append(0)

    bounds = [(0, None)] * n
    if fixed_x is not None:
        for j in J:
            bounds[ix(j)] = (fixed_x[j], fixed_x[j])

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        return None, res.message

    x = {j: res.x[ix(j)] for j in J}
    y_single = {j: res.x[iy(j)] for j in J}
    obj = -res.fun

    return {"x": x, "y_single": y_single, "objective": obj, "status": "Optimal"}, "Optimal"


def solve_expected_value_solution():
    """
    EV: dùng beta_s kỳ vọng để giải mô hình xác định trung bình.
    Sau đó EEV: cố định x_EV và đánh giá lại trong SP thật.
    """
    avg_beta = expected_beta_s()
    ev_model, ev_status = solve_sp_scipy(beta_second=avg_beta)

    if ev_model is None:
        return None, None, ev_status

    x_ev = ev_model["x"]

    # Evaluate EV solution in stochastic world by optimizing recourse for each scenario with fixed x_EV
    eev_model, eev_status = solve_sp_scipy(fixed_x=x_ev)

    return ev_model, eev_model, eev_status


def solve_wait_and_see():
    """
    WS: nếu biết trước kịch bản, giải riêng từng kịch bản.
    EVPI dùng expected WS objective.
    """
    rows = []
    expected_ws = 0.0

    solutions = {}

    for s in S:
        sol, status = solve_single_scenario(s)
        if sol is None:
            rows.append({"Kịch bản": s, "Trạng thái": status})
            continue

        solutions[s] = sol
        expected_ws += PROB[s] * sol["objective"]

        row = {
            "Kịch bản": s,
            "Tên": S_NAME[s],
            "p_s": PROB[s],
            "Objective nếu biết trước": sol["objective"],
            "p_s × Objective": PROB[s] * sol["objective"],
        }
        for j in J:
            row[f"x_{j}"] = sol["x"][j]
            row[f"y_{j}"] = sol["y_single"][j]
        rows.append(row)

    return pd.DataFrame(rows), expected_ws, solutions


def solve_robust_regret(ws_solutions):
    """
    Robust optimization: chọn x,y_s để minimize max_s regret_s.
    regret_s = z_s^WS - [beta*x + beta_s*y_s]
    """
    if not SCIPY_AVAILABLE:
        return None, "Chưa cài scipy"

    ws_obj = {s: ws_solutions[s]["objective"] for s in S}

    n_x = len(J)
    n_y = len(S) * len(J)
    idx_R = n_x + n_y
    n = idx_R + 1

    def ix(j):
        return J.index(j)

    def iy(s, j):
        return n_x + S.index(s) * len(J) + J.index(j)

    c = np.zeros(n)
    c[idx_R] = 1  # minimize R

    A_ub = []
    b_ub = []

    # budget1
    row = np.zeros(n)
    for j in J:
        row[ix(j)] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE1)

    # budget2
    for s in S:
        row = np.zeros(n)
        for j in J:
            row[iy(s, j)] = 1
        A_ub.append(row)
        b_ub.append(BUDGET_STAGE2)

    # yAI <= 0.5 xH
    for s in S:
        row = np.zeros(n)
        row[iy(s, "AI")] = 1
        row[ix("H")] = -0.5
        A_ub.append(row)
        b_ub.append(0)

    # regret constraints:
    # ws_obj - payoff_s <= R
    # -payoff_s - R <= -ws_obj
    for s in S:
        row = np.zeros(n)
        for j in J:
            row[ix(j)] = -BETA[j]
            row[iy(s, j)] = -BETA_S[(s, j)]
        row[idx_R] = -1
        A_ub.append(row)
        b_ub.append(-ws_obj[s])

    bounds = [(0, None)] * n

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        return None, res.message

    x = {j: res.x[ix(j)] for j in J}
    y = {(s, j): res.x[iy(s, j)] for s in S for j in J}
    R = res.x[idx_R]

    # stochastic expected objective of robust decision
    expected_obj = sum(PROB[s] * (sum(BETA[j] * x[j] for j in J) + sum(BETA_S[(s, j)] * y[(s, j)] for j in J)) for s in S)

    return {"x": x, "y": y, "max_regret": R, "expected_objective": expected_obj, "status": "Optimal"}, "Optimal"


def first_stage_compare_df(sp, ev_model, robust, ws_df):
    rows = []

    def add_row(label, xdict, obj=None):
        row = {"Mô hình": label, "Objective": obj}
        for j in J:
            row[f"x_{j}"] = xdict[j]
        row["Tổng x"] = sum(xdict[j] for j in J)
        rows.append(row)

    add_row("SP - stochastic solution", sp["x"], sp["objective"])

    if ev_model is not None:
        add_row("EV - expected value deterministic", ev_model["x"], ev_model["objective"])

    if robust is not None:
        add_row("Robust regret", robust["x"], robust["expected_objective"])

    for _, r in ws_df.iterrows():
        xdict = {j: r[f"x_{j}"] for j in J}
        add_row(f"Wait-and-see {r['Kịch bản']}", xdict, r["Objective nếu biết trước"])

    return pd.DataFrame(rows)


def plot_first_stage(compare_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = compare_df["Mô hình"].tolist()
    bottom = np.zeros(len(compare_df))

    for j in J:
        vals = compare_df[f"x_{j}"].values
        ax.bar(labels, vals, bottom=bottom, label=J_NAME[j])
        bottom += vals

    ax.set_title("So sánh quyết định first-stage x")
    ax.set_ylabel("Tỷ VND")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    return fig


def plot_second_stage(second_df):
    pivot = second_df.pivot_table(index="Tên kịch bản", columns="Tên hạng mục", values="y_sj second-stage", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Quyết định recourse second-stage theo kịch bản")
    ax.set_ylabel("Tỷ VND")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.4)
    return fig


def run():
    st.title("🌦️ Bài 10 – Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")

    st.write("""
    Bài 10 xây dựng mô hình two-stage stochastic programming cho phân bổ ngân sách số 2026–2030.
    Giai đoạn một là quyết định here-and-now trước khi biết kịch bản; giai đoạn hai là quyết định recourse sau khi kịch bản xảy ra.
    """)

    sp, sp_status = solve_sp_scipy()
    ev_model, eev_model, ev_status = solve_expected_value_solution()
    ws_df, expected_ws, ws_solutions = solve_wait_and_see()

    robust, robust_status = solve_robust_regret(ws_solutions) if ws_solutions else (None, "Không có WS")

    if sp is not None and eev_model is not None:
        vss = sp["objective"] - eev_model["objective"]
    else:
        vss = np.nan

    if sp is not None:
        evpi = expected_ws - sp["objective"]
    else:
        evpi = np.nan

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 Mô hình",
        "📋 Dữ liệu",
        "10.5.1 SP",
        "10.5.2 EV & kịch bản",
        "10.5.3 VSS/EVPI",
        "🤖 10.5.4 & 10.6"
    ])

    with tab1:
        st.subheader("10.1–10.3. Cấu trúc mô hình")

        st.markdown("""
        **Giai đoạn 1 – quyết định here-and-now:**
        """)
        st.latex(r"x=(x_I,x_D,x_{AI},x_H), \quad \sum_j x_j \leq 65.000")

        st.markdown("""
        **Giai đoạn 2 – quyết định recourse theo kịch bản:**
        """)
        st.latex(r"y_s=(y^s_I,y^s_D,y^s_{AI},y^s_H), \quad \sum_j y^s_j \leq 15.000")
        st.latex(r"y^s_{AI} \leq 0.5x_H")

        st.markdown("""
        **Hàm mục tiêu đơn giản hóa:**
        """)
        st.latex(r"\max \sum_j \beta_jx_j+\sum_s p_s\sum_j \beta^s_jy^s_j")

        st.info(
            "Module này tính toán bằng scipy.linprog để bảo đảm chạy ổn định trên máy. "
            "Cấu trúc Set/Param/Var của Pyomo được trình bày trong đề; nếu muốn chạy Pyomo thật cần cài pyomo và solver GLPK/CBC."
        )

        if PYOMO_AVAILABLE:
            st.success("Đã import được Pyomo.")
        else:
            st.warning("Chưa cài Pyomo. Cài nếu cần: python -m pip install pyomo")

        if SCIPY_AVAILABLE:
            st.success("Đã có scipy.linprog để giải LP.")
        else:
            st.error("Chưa cài scipy. Cài: python -m pip install scipy")

    with tab2:
        st.subheader("10.2. Scenario tree")
        st.dataframe(scenario_df(), use_container_width=True)

        st.subheader("10.4. Hệ số β theo kịch bản")
        st.dataframe(beta_df(), use_container_width=True)

        st.subheader("β kỳ vọng theo xác suất")
        exp_beta = expected_beta_s()
        st.dataframe(pd.DataFrame({
            "Hạng mục": J,
            "Tên": [J_NAME[j] for j in J],
            "β kỳ vọng recourse": [exp_beta[j] for j in J]
        }), use_container_width=True)

    with tab3:
        st.subheader("Câu 10.5.1 – Lời giải stochastic programming")

        if sp is None:
            st.error(sp_status)
        else:
            first_df, second_df = build_solution_df(sp["x"], sp["y"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", sp_status)
            c2.metric("Objective SP", f"{sp['objective']:,.2f}")
            c3.metric("Tổng x first-stage", f"{first_df['x_j first-stage'].sum():,.0f}")

            st.markdown("### Quyết định first-stage tối ưu")
            st.dataframe(first_df, use_container_width=True)

            st.markdown("### Quyết định second-stage recourse theo kịch bản")
            st.dataframe(second_df, use_container_width=True, height=420)

            fig = plot_second_stage(second_df)
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.subheader("Câu 10.5.2 – Deterministic từng kịch bản và Expected Value")

        st.markdown("### Wait-and-see: giải từng kịch bản riêng rẽ")
        st.dataframe(ws_df, use_container_width=True, height=420)

        st.markdown("### EV solution và SP solution")

        compare_df = first_stage_compare_df(sp, ev_model, robust, ws_df)
        st.dataframe(compare_df, use_container_width=True)

        fig = plot_first_stage(compare_df)
        st.pyplot(fig)
        plt.close(fig)

        if ev_model is not None and eev_model is not None:
            st.markdown("### Đánh giá lời giải EV trong thế giới stochastic")
            ev_first, ev_second = build_solution_df(ev_model["x"], ev_model["y"])
            eev_first, eev_second = build_solution_df(eev_model["x"], eev_model["y"])

            ev_compare = pd.DataFrame({
                "Chỉ tiêu": ["Objective EV deterministic", "EEV - EV solution evaluated under stochastic model", "SP objective"],
                "Giá trị": [ev_model["objective"], eev_model["objective"], sp["objective"] if sp else np.nan]
            })
            st.dataframe(ev_compare, use_container_width=True)

    with tab5:
        st.subheader("Câu 10.5.3 – VSS và EVPI")

        metrics = pd.DataFrame({
            "Chỉ tiêu": [
                "SP - Stochastic Solution",
                "EEV - Expected result of EV solution",
                "WS - Wait-and-see expected value",
                "VSS = SP - EEV",
                "EVPI = WS - SP"
            ],
            "Giá trị": [
                sp["objective"] if sp else np.nan,
                eev_model["objective"] if eev_model else np.nan,
                expected_ws,
                vss,
                evpi
            ],
            "Ý nghĩa": [
                "Giá trị kỳ vọng khi tối ưu có xét xác suất kịch bản",
                "Giá trị khi dùng lời giải kỳ vọng EV rồi đem vào môi trường bất định thật",
                "Giá trị nếu biết trước kịch bản tương lai",
                "Lợi ích của việc dùng mô hình stochastic thay vì EV",
                "Giá trị tối đa của thông tin hoàn hảo"
            ]
        })

        st.dataframe(metrics, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("VSS", f"{vss:,.4f}" if pd.notna(vss) else "Không tính được")
        c2.metric("EVPI", f"{evpi:,.4f}" if pd.notna(evpi) else "Không tính được")

        if pd.notna(vss) and abs(vss) < 1e-6:
            st.warning(
                "VSS bằng 0 hoặc rất gần 0 trong mô hình đơn giản này. Điều đó xảy ra vì bài toán tuyến tính và không có ràng buộc scenario-specific phức tạp ngoài hệ số β, "
                "nên lời giải EV có thể trùng với lời giải SP. Đây là kết quả tính toán hợp lệ, không phải lỗi."
            )

        if pd.notna(evpi) and evpi > 0:
            st.success(
                "EVPI dương cho thấy nếu biết trước kịch bản tương lai, Chính phủ có thể phân bổ ngân sách tốt hơn. "
                "Đây là giá trị kinh tế của thông tin hoàn hảo."
            )

    with tab6:
        st.subheader("Câu 10.5.4 – Robust optimization cực tiểu hóa regret xấu nhất")

        if robust is None:
            st.error(robust_status)
        else:
            robust_first, robust_second = build_solution_df(robust["x"], robust["y"])

            c1, c2 = st.columns(2)
            c1.metric("Max regret tối thiểu", f"{robust['max_regret']:,.2f}")
            c2.metric("Expected objective robust", f"{robust['expected_objective']:,.2f}")

            st.markdown("### Quyết định first-stage robust")
            st.dataframe(robust_first, use_container_width=True)

            st.markdown("### So sánh SP và Robust")
            if sp is not None:
                compare_robust = pd.DataFrame({
                    "Hạng mục": J,
                    "SP x_j": [sp["x"][j] for j in J],
                    "Robust x_j": [robust["x"][j] for j in J],
                    "Chênh lệch Robust - SP": [robust["x"][j] - sp["x"][j] for j in J]
                })
                st.dataframe(compare_robust, use_container_width=True)

        st.subheader("🤖 Câu hỏi thảo luận chính sách 10.6")

        if sp is not None:
            h_sp = sp["x"]["H"]
            if ev_model is not None:
                h_ev = ev_model["x"]["H"]
            else:
                h_ev = np.nan
        else:
            h_sp = h_ev = np.nan

        with st.container(border=True):
            st.markdown("#### a) So với lời giải xác định, SP đầu tư H nhiều hơn hay ít hơn? Vì sao?")
            st.write(
                f"Trong kết quả hiện tại, x_H của SP là **{h_sp:,.0f}**, còn x_H của EV là **{h_ev:,.0f}**."
            )
            st.write(
                "SP có xu hướng coi H như một năng lực nền tảng để mở rộng AI trong giai đoạn hai, vì ràng buộc y_AI^s ≤ 0,5x_H làm cho nhân lực số trở thành điều kiện hấp thụ công nghệ."
            )

        with st.container(border=True):
            st.markdown("#### b) VSS dương nói lên điều gì?")
            if pd.notna(vss) and vss > 1e-6:
                st.write(
                    f"VSS = **{vss:,.2f}** > 0, cho thấy việc xét bất định giúp tăng giá trị kỳ vọng so với cách ra quyết định theo kịch bản trung bình."
                )
            else:
                st.write(
                    "Trong mô hình tuyến tính đơn giản này, VSS có thể bằng 0 do lời giải EV trùng SP. "
                    "Điều đó không phủ nhận giá trị của tư duy xác suất; nó chỉ cho thấy bộ ràng buộc hiện tại chưa đủ tạo khác biệt lớn giữa EV và SP."
                )
            st.write(
                "Về chính sách, VSS dương sẽ hàm ý rằng hoạch định ngân sách Việt Nam nên dùng tư duy xác suất thay vì chỉ dựa vào một kịch bản cơ sở."
            )

        with st.container(border=True):
            st.markdown("#### c) COVID-19 và bão Yagi cho thấy Việt Nam có dưới đầu tư vào nhân lực số như hàng hóa bảo hiểm không?")
            st.write(
                "Các cú sốc như COVID-19 và bão Yagi cho thấy khả năng thích ứng của lao động, doanh nghiệp và bộ máy công quyền rất quan trọng. "
                "Nhân lực số có thể xem như một loại hàng hóa bảo hiểm vì giúp nền kinh tế chuyển đổi việc làm, vận hành dịch vụ số và duy trì chuỗi cung ứng khi có cú sốc."
            )
            st.write(
                "Nếu mô hình cho thấy H có vai trò lớn trong SP hoặc robust solution, điều đó củng cố lập luận rằng Việt Nam không nên chỉ đầu tư vào hạ tầng và AI, "
                "mà cần đầu tư bền bỉ vào nhân lực số, kỹ năng dữ liệu, an ninh mạng và năng lực thích ứng."
            )


if __name__ == "__main__":
    run()
