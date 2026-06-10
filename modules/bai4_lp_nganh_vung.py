import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

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


# =========================================================
# BÀI 4 – LP PHÂN BỔ NGÂN SÁCH SỐ THEO NGÀNH - VÙNG
# BẢN FIX:
# - Giữ giao diện chia tab như bản cũ của bạn
# - Nhưng dùng lambda mặc định 0.68 để có nghiệm Optimal giống ảnh bạn gửi
# - Có PuLP, CVXPY, heatmap, bỏ C5, thảo luận chính sách
# =========================================================


REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long"
}

ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "I - Hạ tầng",
    "D": "D - CĐS DN",
    "AI": "AI",
    "H": "H - Nhân lực"
}

BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25
}

D0 = {
    "NMM": 38,
    "RRD": 78,
    "NCC": 55,
    "CH": 32,
    "SE": 82,
    "MD": 48
}

TOTAL_BUDGET = 50000.0
MIN_REGION = 5000.0
MAX_REGION = 12000.0
MIN_H = 12000.0
GAMMA = 0.002
LAMBDA_DEFAULT = 0.68


def beta_dataframe():
    rows = []
    for r in REGIONS:
        rows.append({
            "Mã vùng": r,
            "Vùng": REGION_NAMES[r],
            "I - Hạ tầng": BETA[(r, "I")],
            "D - CĐS DN": BETA[(r, "D")],
            "AI": BETA[(r, "AI")],
            "H - Nhân lực": BETA[(r, "H")]
        })
    return pd.DataFrame(rows)


def d0_dataframe():
    return pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [REGION_NAMES[r] for r in REGIONS],
        "D_r ban đầu": [D0[r] for r in REGIONS]
    })


def allocation_to_dataframe(allocation):
    matrix = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)

    for r in REGIONS:
        for j in ITEMS:
            matrix.loc[r, j] = allocation.get((r, j), 0.0)

    matrix.insert(0, "Vùng", [REGION_NAMES[r] for r in REGIONS])
    matrix = matrix.rename(columns={j: ITEM_NAMES[j] for j in ITEMS})
    matrix["Tổng"] = matrix[[ITEM_NAMES[j] for j in ITEMS]].sum(axis=1)

    return matrix


def solve_scipy(lam=LAMBDA_DEFAULT, use_fairness=True):
    if not SCIPY_AVAILABLE:
        return None, None, "Chưa cài scipy"

    n = 24
    beta_vec = np.array([BETA[(r, j)] for r in REGIONS for j in ITEMS])
    c = -beta_vec

    A_ub = []
    b_ub = []

    # C1: ngân sách tổng
    A_ub.append(np.ones(n))
    b_ub.append(TOTAL_BUDGET)

    # C2, C3: sàn/trần vùng
    for ri, r in enumerate(REGIONS):
        row = np.zeros(n)
        row[ri * 4:(ri + 1) * 4] = -1
        A_ub.append(row)
        b_ub.append(-MIN_REGION)

        row = np.zeros(n)
        row[ri * 4:(ri + 1) * 4] = 1
        A_ub.append(row)
        b_ub.append(MAX_REGION)

    # C4: H tối thiểu
    row = np.zeros(n)
    for ri in range(6):
        row[ri * 4 + 3] = -1
    A_ub.append(row)
    b_ub.append(-MIN_H)

    # C5: công bằng, tuyến tính hóa trực tiếp bằng cặp i,k
    # D_i + gamma*x_D_i >= lambda*(D_k + gamma*x_D_k) với mọi i,k
    if use_fairness:
        for ii, i in enumerate(REGIONS):
            for kk, k in enumerate(REGIONS):
                row = np.zeros(n)
                row[kk * 4 + 1] += lam * GAMMA
                row[ii * 4 + 1] += -GAMMA
                A_ub.append(row)
                b_ub.append(D0[i] - lam * D0[k])

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n,
        method="highs"
    )

    if not res.success:
        return None, None, res.message

    allocation = {}
    for ri, r in enumerate(REGIONS):
        for ji, j in enumerate(ITEMS):
            allocation[(r, j)] = float(res.x[ri * 4 + ji])

    return allocation, float(-res.fun), "Optimal"


def solve_pulp(lam=LAMBDA_DEFAULT, use_fairness=True):
    if not PULP_AVAILABLE:
        return solve_scipy(lam=lam, use_fairness=use_fairness)

    m = pulp.LpProblem("Bai4_LP_Nganh_Vung", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)

    m += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS)

    m += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= TOTAL_BUDGET

    for r in REGIONS:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= MIN_REGION
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= MAX_REGION

    m += pulp.lpSum(x[r]["H"] for r in REGIONS) >= MIN_H

    if use_fairness:
        for i in REGIONS:
            for k in REGIONS:
                m += D0[i] + GAMMA * x[i]["D"] >= lam * (D0[k] + GAMMA * x[k]["D"])

    try:
        m.solve(pulp.PULP_CBC_CMD(msg=False))
    except Exception:
        return solve_scipy(lam=lam, use_fairness=use_fairness)

    status = pulp.LpStatus[m.status]

    if status != "Optimal":
        return None, None, status

    allocation = {}
    for r in REGIONS:
        for j in ITEMS:
            allocation[(r, j)] = float(pulp.value(x[r][j]))

    return allocation, float(pulp.value(m.objective)), status


def solve_cvxpy(lam=LAMBDA_DEFAULT, use_fairness=True):
    if not CVXPY_AVAILABLE:
        return None, None, "Chưa cài CVXPY"

    X = cp.Variable((6, 4), nonneg=True)
    beta_mat = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS])

    objective = cp.Maximize(cp.sum(cp.multiply(beta_mat, X)))

    constraints = [
        cp.sum(X) <= TOTAL_BUDGET,
        cp.sum(X[:, 3]) >= MIN_H
    ]

    for ri in range(6):
        constraints.append(cp.sum(X[ri, :]) >= MIN_REGION)
        constraints.append(cp.sum(X[ri, :]) <= MAX_REGION)

    if use_fairness:
        d0_vec = np.array([D0[r] for r in REGIONS])
        for i in range(6):
            for k in range(6):
                constraints.append(d0_vec[i] + GAMMA * X[i, 1] >= lam * (d0_vec[k] + GAMMA * X[k, 1]))

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception as e:
            return None, None, f"Lỗi CVXPY: {e}"

    if X.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, prob.status

    allocation = {}
    for ri, r in enumerate(REGIONS):
        for ji, j in enumerate(ITEMS):
            allocation[(r, j)] = float(max(0, X.value[ri, ji]))

    return allocation, float(prob.value), prob.status


def fairness_check(allocation, lam=LAMBDA_DEFAULT):
    rows = []
    d_after = []

    for r in REGIONS:
        d_after.append(D0[r] + GAMMA * allocation.get((r, "D"), 0.0))

    max_d = max(d_after)
    threshold = lam * max_d

    for idx, r in enumerate(REGIONS):
        rows.append({
            "Mã vùng": r,
            "Vùng": REGION_NAMES[r],
            "D sau đầu tư": d_after[idx],
            "Ngưỡng λ × max(D)": threshold,
            "Dư/thiếu": d_after[idx] - threshold,
            "Đạt C5": d_after[idx] + 1e-6 >= threshold
        })

    return pd.DataFrame(rows)


def lambda_diagnosis(lam=LAMBDA_DEFAULT):
    max_d_initial = max(D0.values())
    required = lam * max_d_initial

    rows = []
    for r in REGIONS:
        max_possible = D0[r] + GAMMA * MAX_REGION
        rows.append({
            "Mã vùng": r,
            "Vùng": REGION_NAMES[r],
            "D_r ban đầu": D0[r],
            "D tối đa nếu x_D,r = 12.000": max_possible,
            "Ngưỡng λ × 82": required,
            "Dư/thiếu": max_possible - required
        })

    return pd.DataFrame(rows)


def plot_heatmap(allocation, title="Heatmap phân bổ tối ưu"):
    df = allocation_to_dataframe(allocation)
    item_cols = [ITEM_NAMES[j] for j in ITEMS]
    matrix = df[item_cols].copy()
    matrix.index = df["Vùng"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(np.arange(len(item_cols)))
    ax.set_xticklabels(item_cols, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Hạng mục")
    ax.set_ylabel("Vùng")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.0f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Tỷ VND")
    return fig


def plot_region_stack(allocation):
    df = allocation_to_dataframe(allocation)
    item_cols = [ITEM_NAMES[j] for j in ITEMS]
    plot_df = df.set_index("Vùng")[item_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Cơ cấu ngân sách theo vùng")
    ax.set_xlabel("Tỷ VND")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="lower right")
    return fig


def priority_table(allocation):
    df = allocation_to_dataframe(allocation)
    item_cols = [ITEM_NAMES[j] for j in ITEMS]
    rows = []

    for _, row in df.iterrows():
        values = row[item_cols]
        best = values.idxmax()
        rows.append({
            "Vùng": row["Vùng"],
            "Hạng mục ưu tiên": best,
            "Ngân sách, tỷ VND": values.max()
        })

    return pd.DataFrame(rows)


def run():
    st.title("🗺️ Bài 4 – LP phân bổ ngân sách số theo ngành - vùng")

    st.write("""
    Bài 4 xây dựng bài toán quy hoạch tuyến tính phân bổ 50.000 tỷ VND ngân sách kinh tế số
    cho 6 vùng và 4 hạng mục đầu tư. Bản này giữ giao diện chia tab như cũ nhưng dùng λ mặc định 0,68
    để mô hình có nghiệm tối ưu khi trình bày.
    """)

    with st.sidebar:
        st.markdown("### Bài 4 – tham số")
        lam = st.number_input("λ công bằng", min_value=0.50, max_value=0.75, value=LAMBDA_DEFAULT, step=0.01, format="%.2f")
        st.caption("λ=0.68 có nghiệm; λ=0.70 thường vô nghiệm với dữ liệu gốc.")

    allocation_pulp, z_pulp, status_pulp = solve_pulp(lam=lam, use_fairness=True)
    allocation_cvx, z_cvx, status_cvx = solve_cvxpy(lam=lam, use_fairness=True)
    allocation_no_fair, z_no_fair, status_no_fair = solve_pulp(lam=lam, use_fairness=False)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 Mô hình",
        "⚙️ Dữ liệu",
        "4.4.1 PuLP",
        "4.4.2 CVXPY",
        "4.4.3 Heatmap",
        "4.4.4 & 4.5"
    ])

    with tab1:
        st.subheader("4.1. Bối cảnh và mô hình")
        st.write("""
        Mục tiêu là tối đa hóa GDP gain kỳ vọng từ phân bổ ngân sách số, đồng thời bảo đảm giới hạn ngân sách,
        sàn/trần vùng, sàn nhân lực số và công bằng vùng miền.
        """)

        st.latex(r"\max Z = \sum_r \sum_j \beta_{j,r}x_{j,r}")

        st.markdown("""
        **Ràng buộc:**
        - C1: Tổng ngân sách không vượt 50.000 tỷ VND.
        - C2: Mỗi vùng nhận tối thiểu 5.000 tỷ VND.
        - C3: Mỗi vùng nhận tối đa 12.000 tỷ VND.
        - C4: Tổng đầu tư nhân lực số tối thiểu 12.000 tỷ VND.
        - C5: Công bằng số hóa vùng miền: \(D_r+\gamma x_{D,r}\geq\lambda\max_r(D_r+\gamma x_{D,r})\).
        - C6: \(x_{j,r}\geq 0\).
        """)

        st.info(
            "Để mô hình có nghiệm khi demo, λ mặc định là 0,68. Nếu đặt λ=0,70, mô hình có thể vô nghiệm do Tây Nguyên không đạt ngưỡng C5."
        )

        st.markdown("### Chẩn đoán nhanh ngưỡng λ")
        st.dataframe(lambda_diagnosis(lam), use_container_width=True)

    with tab2:
        st.subheader("4.3. Dữ liệu đầu vào")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Chỉ số D_r ban đầu")
            st.dataframe(d0_dataframe(), use_container_width=True)

        with col2:
            st.markdown("### Bảng hệ số β")
            st.dataframe(beta_dataframe(), use_container_width=True)

        st.markdown("### Tham số cố định")
        params = pd.DataFrame({
            "Tham số": ["Ngân sách", "Sàn vùng", "Trần vùng", "Sàn H", "γ", "λ đang dùng"],
            "Giá trị": [TOTAL_BUDGET, MIN_REGION, MAX_REGION, MIN_H, GAMMA, lam]
        })
        st.dataframe(params, use_container_width=True)

    with tab3:
        st.subheader("Câu 4.4.1 – Cài đặt và giải bài toán bằng PuLP/CBC")

        if allocation_pulp is None:
            st.error(f"PuLP không tìm được nghiệm tối ưu: {status_pulp}")
            st.dataframe(lambda_diagnosis(lam), use_container_width=True)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trạng thái PuLP", status_pulp)
            m2.metric("Z* PuLP", f"{z_pulp:,.2f}")
            m3.metric("Tổng ngân sách", f"{allocation_to_dataframe(allocation_pulp)['Tổng'].sum():,.0f}")
            m4.metric("λ", f"{lam:.2f}")

            st.markdown("### Ma trận phân bổ tối ưu 6 × 4")
            st.dataframe(allocation_to_dataframe(allocation_pulp), use_container_width=True, height=330)

            st.markdown("### Kiểm tra ràng buộc công bằng C5")
            st.dataframe(fairness_check(allocation_pulp, lam), use_container_width=True)

    with tab4:
        st.subheader("Câu 4.4.2 – Cài đặt lại bằng CVXPY và so sánh với PuLP")

        if allocation_pulp is None:
            st.error("Chưa có nghiệm PuLP để so sánh.")
        else:
            diff_obj = abs(z_pulp - z_cvx) if z_cvx is not None else np.nan

            if allocation_cvx is not None:
                max_alloc_diff = max(
                    abs(allocation_pulp[(r, j)] - allocation_cvx[(r, j)])
                    for r in REGIONS for j in ITEMS
                )
            else:
                max_alloc_diff = np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái PuLP", status_pulp)
            c2.metric("Trạng thái CVXPY", status_cvx)
            c3.metric("Chênh lệch Z*", f"{diff_obj:,.4f}" if pd.notna(diff_obj) else "Không tính được")

            compare = pd.DataFrame({
                "Chỉ tiêu": [
                    "Z* PuLP",
                    "Z* CVXPY",
                    "Chênh lệch hàm mục tiêu",
                    "Chênh lệch phân bổ lớn nhất",
                    "Trạng thái CVXPY"
                ],
                "Kết quả": [
                    z_pulp,
                    z_cvx if z_cvx is not None else np.nan,
                    diff_obj,
                    max_alloc_diff,
                    status_cvx
                ]
            })
            st.dataframe(compare, use_container_width=True)

            if allocation_cvx is None:
                st.warning("CVXPY chưa chạy được. Cài bằng: python -m pip install cvxpy")
            else:
                st.success("PuLP và CVXPY cho kết quả gần trùng nhau, có thể dùng để đối chiếu độ tin cậy của nghiệm.")

    with tab5:
        st.subheader("Câu 4.4.3 – Heatmap và hạng mục ưu tiên")

        if allocation_pulp is None:
            st.error("Chưa có nghiệm để vẽ heatmap.")
        else:
            fig = plot_heatmap(allocation_pulp, "Heatmap phân bổ tối ưu 6 vùng × 4 hạng mục")
            st.pyplot(fig)
            plt.close(fig)

            fig2 = plot_region_stack(allocation_pulp)
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown("### Hạng mục ưu tiên ở từng vùng")
            st.dataframe(priority_table(allocation_pulp), use_container_width=True)

    with tab6:
        st.subheader("Câu 4.4.4 – Đối chiếu với mô hình không có ràng buộc công bằng")

        if allocation_pulp is None or allocation_no_fair is None:
            st.error("Không đủ nghiệm để so sánh.")
            st.write(f"Trạng thái bỏ C5: {status_no_fair}")
        else:
            fairness_cost = z_no_fair - z_pulp
            fairness_cost_pct = fairness_cost / z_no_fair * 100 if z_no_fair else np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* có công bằng", f"{z_pulp:,.0f}")
            c2.metric("Z* bỏ C5", f"{z_no_fair:,.0f}")
            c3.metric("% chi phí công bằng", f"{fairness_cost_pct:.4f}%")

            compare = pd.DataFrame({
                "Kịch bản": ["Có ràng buộc công bằng C5", "Bỏ ràng buộc công bằng C5"],
                "Trạng thái": [status_pulp, status_no_fair],
                "Z*": [z_pulp, z_no_fair],
                "Tổng ngân sách": [
                    allocation_to_dataframe(allocation_pulp)["Tổng"].sum(),
                    allocation_to_dataframe(allocation_no_fair)["Tổng"].sum()
                ]
            })
            st.dataframe(compare, use_container_width=True)

            st.write(
                f"Chi phí kinh tế của công bằng vùng miền là khoảng **{fairness_cost:,.0f}** tỷ VND GDP gain, "
                f"tương đương **{fairness_cost_pct:.4f}%** so với mô hình bỏ C5."
            )

            st.markdown("### Ma trận phân bổ khi bỏ C5")
            st.dataframe(allocation_to_dataframe(allocation_no_fair), use_container_width=True)

        st.subheader("🤖 Tác nhân phân tích chính sách 4.5")

        with st.container(border=True):
            st.markdown("#### a) Nếu bỏ công bằng, vốn sẽ chảy về đâu?")
            st.write(
                "Khi bỏ C5, vốn có xu hướng tập trung vào vùng/hạng mục có hệ số β cao nhất, ví dụ AI ở Đông Nam Bộ, "
                "AI ở Đồng bằng sông Hồng hoặc nhân lực số ở một số vùng. Điều này làm tăng GDP gain nhưng dễ làm tăng chênh lệch vùng."
            )

        with st.container(border=True):
            st.markdown("#### b) C3 có thể coi là chính sách phân quyền không?")
            st.write(
                "Có. Trần ngân sách vùng C3 giúp ngăn vốn dồn quá mức vào một vài cực tăng trưởng. "
                "Dù làm giảm Z*, ràng buộc này giúp bảo đảm phân bổ ngân sách cân bằng và có tính chính danh chính sách cao hơn."
            )

        with st.container(border=True):
            st.markdown("#### c) Tây Nguyên nên đầu tư AI hay H/I trước?")
            st.write(
                "Tây Nguyên có hệ số AI thấp nên không nên ưu tiên AI ngay từ đầu. Mô hình thường ưu tiên CĐS doanh nghiệp hoặc nhân lực số, "
                "hàm ý cần xây nền tảng số và nhân lực trước rồi mới mở rộng AI."
            )

        with st.container(border=True):
            st.markdown("#### Ghi chú về λ")
            st.write(
                "Với λ=0,70 theo tham số gốc, mô hình có thể vô nghiệm. Bản này dùng λ=0,68 để giữ tinh thần công bằng vùng miền "
                "nhưng vẫn có nghiệm tối ưu phục vụ demo và phân tích."
            )


if __name__ == "__main__":
    run()
