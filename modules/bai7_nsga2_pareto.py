import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except Exception:
    PYMOO_AVAILABLE = False


# =========================================================
# BÀI 7 – TỐI ƯU ĐA MỤC TIÊU PARETO VỚI NSGA-II

# - Có chọn 4 mục tiêu
# - Có pop_size, số thế hệ, seed, nút chạy
# - Có mặt Pareto 3D + Parallel coordinates
# - Có TOPSIS chọn nghiệm thỏa hiệp với slider trọng số
# - Có heatmap phân bổ của nghiệm thỏa hiệp
# - Có chi phí cơ hội: GDP max hy sinh bao trùm/môi trường/an ninh bao nhiêu
# - Có câu hỏi thảo luận chính sách
# =========================================================


REGIONS = ["TDMNPB", "DBSH", "BTB-DHMT", "Tay Nguyen", "DNB", "DBSCL"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long"
]
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = ["Hạ tầng", "Dữ liệu", "AI", "Nhân lực"]

BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25]
], dtype=float)

# Bảng tham số bổ sung Bài 7
E = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38], dtype=float)
RHO = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22], dtype=float)
SIGMA = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30], dtype=float)

TOTAL_BUDGET = 50000.0
MIN_REGION = 5000.0
MAX_REGION = 12000.0
MIN_H = 12000.0


def evaluate_x(x):
    X = np.array(x, dtype=float).reshape(6, 4)
    region_sum = X.sum(axis=1)

    # f1: GDP gain, maximize
    gdp_gain = float((BETA * X).sum())

    # f2: bất bình đẳng phân bổ vùng, minimize
    inequality = float(np.abs(region_sum - region_sum.mean()).mean())

    # f3: phát thải, minimize
    emission = float((E * (X[:, 0] + X[:, 2])).sum())

    # f4: rủi ro an ninh dữ liệu ròng, minimize
    security_risk = float((RHO * X[:, 2]).sum() - (SIGMA * X[:, 3]).sum())

    return gdp_gain, inequality, emission, security_risk


def constraints_g(x):
    """
    G <= 0 cho pymoo.
    Giữ các ràng buộc ngân sách, sàn/trần vùng, H tối thiểu.
    Không ép C5 cứng của Bài 4 vì C5 lambda=0.70 từng gây vô nghiệm; trong Bài 7 bao trùm được đưa vào f2.
    """
    X = np.array(x, dtype=float).reshape(6, 4)
    region_sum = X.sum(axis=1)

    G = []

    # C1 tổng ngân sách
    G.append(X.sum() - TOTAL_BUDGET)

    # C2 sàn vùng
    for s in region_sum:
        G.append(MIN_REGION - s)

    # C3 trần vùng
    for s in region_sum:
        G.append(s - MAX_REGION)

    # C4 H tối thiểu
    G.append(MIN_H - X[:, 3].sum())

    return np.array(G, dtype=float)


if PYMOO_AVAILABLE:
    class VietnamDigitalParetoProblem(ElementwiseProblem):
        def __init__(self, objective_flags):
            self.objective_flags = objective_flags
            super().__init__(
                n_var=24,
                n_obj=sum(objective_flags),
                n_ieq_constr=14,
                xl=np.zeros(24),
                xu=np.ones(24) * MAX_REGION
            )

        def _evaluate(self, x, out, *args, **kwargs):
            gdp, inequality, emission, risk = evaluate_x(x)
            all_f = [
                -gdp,       # maximize GDP => minimize -GDP
                inequality, # minimize inequality
                emission,   # minimize emission
                risk        # minimize risk
            ]
            out["F"] = [all_f[i] for i, flag in enumerate(self.objective_flags) if flag]
            out["G"] = constraints_g(x)


def random_feasible_solution(rng):
    """
    Fallback tạo nghiệm khả thi nếu pymoo chưa import được.
    Vẫn đảm bảo ngân sách, sàn/trần vùng, H tối thiểu.
    """
    # region budgets sum exactly 50k and each in [5k, 12k]
    base = np.ones(6) * MIN_REGION
    remain = TOTAL_BUDGET - base.sum()

    cap_extra = MAX_REGION - MIN_REGION
    extra = rng.dirichlet(np.ones(6)) * remain
    extra = np.minimum(extra, cap_extra)

    # điều chỉnh phần dư do clip
    region_budget = base + extra
    for _ in range(100):
        diff = TOTAL_BUDGET - region_budget.sum()
        if abs(diff) < 1e-7:
            break
        if diff > 0:
            idx = np.where(region_budget < MAX_REGION - 1e-8)[0]
            if len(idx) == 0:
                break
            add = min(diff / len(idx), 400)
            region_budget[idx] += np.minimum(add, MAX_REGION - region_budget[idx])
        else:
            idx = np.where(region_budget > MIN_REGION + 1e-8)[0]
            if len(idx) == 0:
                break
            sub = min(-diff / len(idx), 400)
            region_budget[idx] -= np.minimum(sub, region_budget[idx] - MIN_REGION)

    X = np.zeros((6, 4))
    for r in range(6):
        mix = rng.dirichlet(np.ones(4))
        X[r] = region_budget[r] * mix

    # đảm bảo H >= 12k bằng cách chuyển một phần từ I/D/AI sang H
    h_total = X[:, 3].sum()
    if h_total < MIN_H:
        need = MIN_H - h_total
        for r in range(6):
            if need <= 1e-8:
                break
            transferable = X[r, :3].sum()
            shift = min(need / (6 - r), transferable)
            if shift > 0:
                share = X[r, :3] / max(X[r, :3].sum(), 1e-9)
                X[r, :3] -= share * shift
                X[r, 3] += shift
                need -= shift

    return X.flatten()


def nondominated_filter(F):
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated[i] = True
                break
    return ~dominated


@st.cache_data(show_spinner=False)
def run_nsga2(objective_flags, pop_size=100, n_gen=200, seed=42):
    objective_flags = tuple(objective_flags)

    if PYMOO_AVAILABLE:
        problem = VietnamDigitalParetoProblem(objective_flags)
        algo = NSGA2(pop_size=pop_size)
        res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)

        if res.X is not None:
            X = np.atleast_2d(res.X)
            return X, "NSGA-II bằng pymoo"

    # Fallback: random search + lọc phi trội
    rng = np.random.default_rng(seed)
    n_sample = max(2500, pop_size * n_gen // 2)
    X_list, F_list = [], []

    for _ in range(n_sample):
        x = random_feasible_solution(rng)
        gdp, ine, emi, risk = evaluate_x(x)
        full_F = np.array([-gdp, ine, emi, risk])
        F_selected = full_F[[i for i, flag in enumerate(objective_flags) if flag]]
        X_list.append(x)
        F_list.append(F_selected)

    X_arr = np.array(X_list)
    F_arr = np.array(F_list)
    keep = nondominated_filter(F_arr)
    return X_arr[keep], "Fallback phi trội do pymoo chưa chạy trong môi trường"


def make_pareto_df(X):
    rows = []
    for idx, x in enumerate(X):
        gdp, inequality, emission, risk = evaluate_x(x)
        rows.append({
            "id": idx,
            "GDP_gain": gdp,
            "Bất_bình_đẳng": inequality,
            "Phát_thải": emission,
            "Rủi_ro_an_ninh": risk,
            "Bao_trùm_score": 1 / (1 + inequality)
        })
    return pd.DataFrame(rows)


def topsis_rank(pareto_df, weights):
    """
    TOPSIS:
    - GDP_gain: benefit
    - Bất_bình_đẳng: cost
    - Phát_thải: cost
    - Rủi_ro_an_ninh: cost
    """
    cols = ["GDP_gain", "Bất_bình_đẳng", "Phát_thải", "Rủi_ro_an_ninh"]
    benefit = np.array([True, False, False, False])
    X = pareto_df[cols].values.astype(float)

    # Min-max normalize thành điểm tốt 0-1
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        mn, mx = X[:, j].min(), X[:, j].max()
        if mx == mn:
            Z[:, j] = 1.0
        elif benefit[j]:
            Z[:, j] = (X[:, j] - mn) / (mx - mn)
        else:
            Z[:, j] = (mx - X[:, j]) / (mx - mn)

    w = np.array(weights, dtype=float)
    w = w / max(w.sum(), 1e-9)
    V = Z * w

    ideal = V.max(axis=0)
    anti = V.min(axis=0)

    s_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    c_star = s_neg / (s_pos + s_neg + 1e-12)

    out = pareto_df.copy()
    out["TOPSIS_score"] = c_star
    out["rank"] = out["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)


def allocation_table(x):
    X = np.array(x).reshape(6, 4)
    df = pd.DataFrame(X, columns=ITEM_NAMES)
    df.insert(0, "Vùng", REGION_NAMES)
    df["Tổng"] = df[ITEM_NAMES].sum(axis=1)
    return df


def opportunity_cost(pareto_df, comp_row):
    growth_best = pareto_df.sort_values("GDP_gain", ascending=False).iloc[0]

    # Hy sinh bao trùm = bất bình đẳng tăng bao nhiêu % ở nghiệm GDP max so với compromise
    inc_cost = (growth_best["Bất_bình_đẳng"] - comp_row["Bất_bình_đẳng"]) / max(abs(comp_row["Bất_bình_đẳng"]), 1e-9) * 100

    # Môi trường xấu hơn = phát thải tăng bao nhiêu %
    emi_cost = (growth_best["Phát_thải"] - comp_row["Phát_thải"]) / max(abs(comp_row["Phát_thải"]), 1e-9) * 100

    # An ninh xấu hơn = risk tăng bao nhiêu %, dùng abs mẫu để tránh dấu âm
    risk_cost = (growth_best["Rủi_ro_an_ninh"] - comp_row["Rủi_ro_an_ninh"]) / max(abs(comp_row["Rủi_ro_an_ninh"]), 1e-9) * 100

    return growth_best, inc_cost, emi_cost, risk_cost


def plot_3d(pareto_df, comp_id=None):
    if PLOTLY_AVAILABLE:
        fig = px.scatter_3d(
            pareto_df,
            x="GDP_gain",
            y="Bất_bình_đẳng",
            z="Phát_thải",
            color="Rủi_ro_an_ninh",
            hover_data=["id"],
            title="Pareto 3D: GDP - Bất bình đẳng - Phát thải"
        )
        if comp_id is not None:
            row = pareto_df[pareto_df["id"] == comp_id].iloc[0]
            fig.add_trace(go.Scatter3d(
                x=[row["GDP_gain"]],
                y=[row["Bất_bình_đẳng"]],
                z=[row["Phát_thải"]],
                mode="markers",
                marker=dict(size=8, symbol="diamond"),
                name="Nghiệm TOPSIS"
            ))
        fig.update_layout(height=620)
        return fig

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pareto_df["GDP_gain"],
        pareto_df["Bất_bình_đẳng"],
        pareto_df["Phát_thải"],
        c=pareto_df["Rủi_ro_an_ninh"]
    )
    ax.set_xlabel("GDP gain")
    ax.set_ylabel("Bất bình đẳng")
    ax.set_zlabel("Phát thải")
    ax.set_title("Pareto 3D")
    fig.colorbar(sc, ax=ax, label="Rủi ro an ninh")
    return fig


def plot_parallel(pareto_df):
    cols = ["GDP_gain", "Bất_bình_đẳng", "Phát_thải", "Rủi_ro_an_ninh"]
    labels = ["GDP", "Bất bình đẳng", "Phát thải", "An ninh"]

    X = pareto_df[cols].values.astype(float)
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        mn, mx = X[:, j].min(), X[:, j].max()
        Z[:, j] = 0.5 if mx == mn else (X[:, j] - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(Z.shape[0]):
        ax.plot(range(4), Z[i], alpha=0.20, linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Chuẩn hóa 0-1")
    ax.set_title("Parallel coordinates cho 4 mục tiêu")
    ax.grid(alpha=0.4)
    return fig


def plot_heatmap_allocation(x):
    df = allocation_table(x)
    matrix = df[ITEM_NAMES].values

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(ITEM_NAMES)))
    ax.set_xticklabels(ITEM_NAMES)
    ax.set_yticks(np.arange(len(REGION_NAMES)))
    ax.set_yticklabels(REGION_NAMES)
    ax.set_title("Phân bổ ngân sách nghiệm thỏa hiệp (tỷ VND)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Tỷ VND")
    return fig


def run():
    st.title("🧬 Bài 7 – Tối ưu đa mục tiêu Pareto với NSGA-II")

    st.write("""
    Bài 7 giải bài toán phân bổ ngân sách số dưới 4 mục tiêu xung đột:
    tăng trưởng GDP, bao trùm vùng, môi trường và an ninh dữ liệu. Kết quả là tập nghiệm Pareto,
    sau đó dùng TOPSIS để chọn một nghiệm thỏa hiệp.
    """)

    st.subheader("1. Chọn mục tiêu và tham số NSGA-II")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_f1 = st.checkbox("max GDP gain (f1)", value=True)
    with c2:
        use_f2 = st.checkbox("min bất bình đẳng (f2)", value=True)
    with c3:
        use_f3 = st.checkbox("min phát thải (f3)", value=True)
    with c4:
        use_f4 = st.checkbox("min rủi ro an ninh (f4)", value=True)

    objective_flags = [use_f1, use_f2, use_f3, use_f4]
    if not any(objective_flags):
        st.error("Cần chọn ít nhất một mục tiêu.")
        return

    p1, p2, p3 = st.columns(3)
    with p1:
        pop_size = st.slider("Pop size", 50, 200, 100, 10)
    with p2:
        n_gen = st.slider("Số thế hệ", 50, 300, 200, 10)
    with p3:
        seed = st.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)

    if not PYMOO_AVAILABLE:
        st.warning("Chưa import được pymoo. App sẽ dùng fallback phi trội để vẫn có kết quả. Cài lại: python -m pip install pymoo")

    run_button = st.button("🚀 Chạy NSGA-II", type="primary")

    if "bai7_ran" not in st.session_state:
        st.session_state["bai7_ran"] = False

    if run_button:
        st.session_state["bai7_ran"] = True
        st.session_state["bai7_params"] = {
            "objective_flags": objective_flags,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "seed": int(seed)
        }

    if not st.session_state["bai7_ran"]:
        st.info("Bấm **Chạy NSGA-II** để bắt đầu.")
        return

    params = st.session_state["bai7_params"]

    with st.spinner("Đang chạy NSGA-II / lọc Pareto..."):
        X, status = run_nsga2(
            params["objective_flags"],
            pop_size=params["pop_size"],
            n_gen=params["n_gen"],
            seed=params["seed"]
        )

    pareto_df = make_pareto_df(X)

    st.success(f"Trạng thái chạy: {status}. Số nghiệm Pareto/phi trội: {len(pareto_df)}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "7.4.1–7.4.2 Pareto",
        "7.4.3 TOPSIS",
        "7.4.4 Chi phí cơ hội",
        "7.5 Thảo luận"
    ])

    with tab1:
        st.subheader("Mặt Pareto và parallel coordinates")

        fig = plot_3d(pareto_df)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)
            plt.close(fig)

        fig2 = plot_parallel(pareto_df)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("### Bảng nghiệm Pareto")
        st.dataframe(pareto_df, use_container_width=True, height=360)

    with tab2:
        st.subheader("🎯 Chọn nghiệm thỏa hiệp bằng TOPSIS")

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            wgdp = st.slider("w GDP", 0.0, 1.0, 0.40, 0.05)
        with w2:
            winc = st.slider("w Bao trùm", 0.0, 1.0, 0.25, 0.05)
        with w3:
            wenv = st.slider("w Môi trường", 0.0, 1.0, 0.20, 0.05)
        with w4:
            wrisk = st.slider("w An ninh", 0.0, 1.0, 0.15, 0.05)

        topsis_df = topsis_rank(pareto_df, [wgdp, winc, wenv, wrisk])
        comp = topsis_df.iloc[0]
        comp_id = int(comp["id"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GDP gain", f"{comp['GDP_gain']:,.0f} tỷ")
        m2.metric("Bất bình đẳng", f"{comp['Bất_bình_đẳng']:,.0f}")
        m3.metric("Phát thải", f"{comp['Phát_thải']:,.0f}")
        m4.metric("Rủi ro ròng", f"{comp['Rủi_ro_an_ninh']:,.1f}")

        st.markdown("### Top 10 nghiệm theo TOPSIS")
        st.dataframe(topsis_df.head(10), use_container_width=True)

        st.markdown("### Phân bổ ngân sách nghiệm thỏa hiệp")
        st.dataframe(allocation_table(X[comp_id]), use_container_width=True)

        fig_h = plot_heatmap_allocation(X[comp_id])
        st.pyplot(fig_h)
        plt.close(fig_h)

        st.markdown("### Pareto 3D có đánh dấu nghiệm TOPSIS")
        fig3 = plot_3d(pareto_df, comp_id=comp_id)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.pyplot(fig3)
            plt.close(fig3)

    with tab3:
        st.subheader("⚖️ Chi phí cơ hội của các mục tiêu")

        topsis_df_default = topsis_rank(pareto_df, [0.40, 0.25, 0.20, 0.15])
        comp_default = topsis_df_default.iloc[0]
        growth_best, inc_cost, emi_cost, risk_cost = opportunity_cost(pareto_df, comp_default)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP gain max", f"{growth_best['GDP_gain']:,.0f}")
        c2.metric("+ Bất bình đẳng hơn", f"{inc_cost:+.1f}%")
        c3.metric("+ Phát thải hơn", f"{emi_cost:+.1f}%")
        c4.metric("+ Rủi ro hơn", f"{risk_cost:+.1f}%")

        compare = pd.DataFrame({
            "Chỉ tiêu": ["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro an ninh"],
            "Nghiệm TOPSIS": [
                comp_default["GDP_gain"],
                comp_default["Bất_bình_đẳng"],
                comp_default["Phát_thải"],
                comp_default["Rủi_ro_an_ninh"]
            ],
            "Nghiệm GDP max": [
                growth_best["GDP_gain"],
                growth_best["Bất_bình_đẳng"],
                growth_best["Phát_thải"],
                growth_best["Rủi_ro_an_ninh"]
            ],
            "Chênh lệch": [
                growth_best["GDP_gain"] - comp_default["GDP_gain"],
                growth_best["Bất_bình_đẳng"] - comp_default["Bất_bình_đẳng"],
                growth_best["Phát_thải"] - comp_default["Phát_thải"],
                growth_best["Rủi_ro_an_ninh"] - comp_default["Rủi_ro_an_ninh"]
            ]
        })
        st.dataframe(compare, use_container_width=True)

        st.info(
            "Cách đọc: nghiệm GDP max có thể làm GDP cao hơn, nhưng thường phải đánh đổi bằng bất bình đẳng vùng cao hơn, phát thải cao hơn hoặc rủi ro an ninh cao hơn."
        )

    with tab4:
        st.subheader("💬 Câu hỏi thảo luận chính sách")

        with st.expander("a) Đánh đổi giữa tăng trưởng và bao trùm có rõ không?", expanded=True):
            st.write(
                "Có. Khi mô hình ưu tiên GDP gain, vốn thường dồn vào vùng/hạng mục có hệ số β cao, làm chênh lệch phân bổ vùng tăng. "
                "Điều này phản ánh thực tế cơ cấu kinh tế Việt Nam có các cực tăng trưởng mạnh, nhưng khoảng cách vùng và năng lực số vẫn đáng kể."
            )

        with st.expander("b) Trọng số (0.40, 0.25, 0.20, 0.15) có phù hợp không?", expanded=False):
            st.write(
                "Bộ trọng số này phù hợp nếu tăng trưởng vẫn là ưu tiên lớn nhất, nhưng đã dành trọng số đáng kể cho bao trùm, môi trường và an ninh dữ liệu. "
                "Nếu nhấn mạnh COP26, có thể tăng trọng số môi trường. Nếu nhấn mạnh chủ quyền số và an ninh dữ liệu, có thể tăng trọng số an ninh."
            )

        with st.expander("c) NSGA-II khác LP đơn mục tiêu như thế nào? Có thay thế quyết định chính trị không?", expanded=False):
            st.write(
                "LP đơn mục tiêu cho một nghiệm tối ưu theo một hàm mục tiêu. NSGA-II cho một tập nghiệm Pareto, giúp nhìn rõ đánh đổi giữa các mục tiêu. "
                "Tuy nhiên, NSGA-II không thay thế quyết định chính trị; nó chỉ cung cấp bằng chứng định lượng để nhà hoạch định chính sách thảo luận và lựa chọn."
            )


if __name__ == "__main__":
    run()
