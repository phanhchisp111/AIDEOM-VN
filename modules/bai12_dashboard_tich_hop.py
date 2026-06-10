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


SCENARIOS = [
    "S1. Truyền thống",
    "S2. Số hóa nhanh",
    "S3. AI dẫn dắt",
    "S4. Bao trùm số",
    "S5. Tối ưu cân bằng"
]

SCENARIO_DESC = {
    "S1. Truyền thống": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu",
    "S2. Số hóa nhanh": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số",
    "S3. AI dẫn dắt": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu",
    "S4. Bao trùm số": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số",
    "S5. Tối ưu cân bằng": "Kết quả mô hình AIDEOM-VN cân bằng nhiều mục tiêu"
}

BASE_ALLOC = pd.DataFrame({
    "Kịch bản": SCENARIOS,
    "K": [0.70, 0.25, 0.20, 0.30, 0.35],
    "D": [0.10, 0.45, 0.20, 0.20, 0.25],
    "AI": [0.10, 0.15, 0.45, 0.10, 0.20],
    "H": [0.10, 0.15, 0.15, 0.40, 0.20],
})

MODULES = pd.DataFrame({
    "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
    "Tên": [
        "Dự báo kinh tế",
        "Đánh giá sẵn sàng số",
        "Tối ưu phân bổ",
        "Mô phỏng lao động",
        "Đánh giá rủi ro",
        "Dashboard ra quyết định"
    ],
    "Đầu vào": [
        "Macro 2020–2025",
        "Sectors, Regions",
        "Budget, β-matrix",
        "AI plans, H plans",
        "Risk parameters",
        "Outputs M1–M5"
    ],
    "Đầu ra": [
        "GDP, TFP, lao động 2026–2030",
        "Digital Index + AI Readiness",
        "Phân bổ ngành-vùng-thời gian",
        "NetJob từng ngành",
        "Cyber, environmental, dependency risk",
        "Trực quan kịch bản, cảnh báo, khuyến nghị"
    ],
    "Kỹ thuật chính": [
        "Cobb-Douglas + Bài 1",
        "TOPSIS + Entropy + Bài 6",
        "LP Bài 4 + Dynamic Bài 8",
        "Bài 9 + Markov chain mở rộng",
        "NSGA-II Bài 7 + SP Bài 10",
        "Streamlit/Plotly Dashboard"
    ]
})


def minmax_benefit(x):
    x = np.array(x, dtype=float)
    if np.nanmax(x) == np.nanmin(x):
        return np.ones_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def minmax_cost(x):
    x = np.array(x, dtype=float)
    if np.nanmax(x) == np.nanmin(x):
        return np.ones_like(x)
    return (np.nanmax(x) - x) / (np.nanmax(x) - np.nanmin(x))


def rebalance(values):
    arr = np.maximum(np.array(values, dtype=float), 0)
    if arr.sum() == 0:
        arr = np.array([0.35, 0.25, 0.20, 0.20])
    return arr / arr.sum()


def compute_dashboard(total_budget, shock, cyber_weight, emission_weight, s5_alloc, weights):
    df = BASE_ALLOC.copy()
    df.loc[df["Kịch bản"].eq("S5. Tối ưu cân bằng"), ["K", "D", "AI", "H"]] = s5_alloc
    df["Mô tả"] = df["Kịch bản"].map(SCENARIO_DESC)

    df["GDP_2030"] = 12847.6 * (
        1 + 0.18*df["K"] + 0.26*df["D"] + 0.30*df["AI"] + 0.16*df["H"]
    ) * (total_budget / 80000) * (1 - 0.35*shock)

    df["GDP_gain"] = total_budget * (
        0.78*df["K"] + 0.92*df["D"] + 1.05*df["AI"] + 0.72*df["H"]
    ) * (1 - 0.25*shock)

    df["TFP_2030"] = 1 + 0.10*df["D"] + 0.16*df["AI"] + 0.12*df["H"]
    df["Digital_Index"] = np.clip(45 + 70*df["D"] + 35*df["AI"] + 20*df["H"], 0, 100)
    df["AI_Readiness"] = np.clip(35 + 90*df["AI"] + 25*df["H"] + 15*df["D"], 0, 100)

    df["K_budget"] = df["K"] * total_budget
    df["D_budget"] = df["D"] * total_budget
    df["AI_budget"] = df["AI"] * total_budget
    df["H_budget"] = df["H"] * total_budget

    df["NetJob"] = total_budget * (
        3.2*df["H"] + 2.1*df["D"] + 1.2*df["AI"] + 0.9*df["K"]
        - 0.65*np.maximum(df["AI"] - 0.25, 0)
    ) * (1 - 0.20*shock)

    df["JobRisk"] = np.clip(100 * (0.45*df["AI"] + 0.15*df["D"] - 0.30*df["H"]), 0, 100)
    df["Phat_thai"] = total_budget * (
        0.34*df["K"] + 0.18*df["D"] + 0.31*df["AI"] + 0.10*df["H"]
    ) * 0.55 * emission_weight

    df["CyberRisk"] = np.clip(100 * (0.55*df["AI"] + 0.22*df["D"] - 0.28*df["H"]) * cyber_weight, 0, 100)
    df["DependencyRisk"] = np.clip(100 * (0.30*df["AI"] + 0.20*df["D"] - 0.15*df["H"]), 0, 100)
    df["Rui_ro"] = 0.50*df["CyberRisk"] + 0.30*df["DependencyRisk"] + 0.20*df["JobRisk"]
    df["Bao_trum"] = np.clip(100 * (0.35*df["H"] + 0.25*df["D"] + 0.15*df["K"] + 0.10*(1-df["AI"])) * 2.2, 0, 100)
    df["Resilience"] = np.clip(45 + 50*df["H"] + 30*df["D"] - 8*df["AI"] + 12*(df["K"] < 0.4), 0, 100)

    df["score_GDP"] = minmax_benefit(df["GDP_gain"])
    df["score_Emission"] = minmax_cost(df["Phat_thai"])
    df["score_Risk"] = minmax_cost(df["Rui_ro"])
    df["score_NetJob"] = minmax_benefit(df["NetJob"])
    df["score_Inclusion"] = minmax_benefit(df["Bao_trum"])
    df["score_Digital"] = minmax_benefit(df["Digital_Index"])
    df["score_AI"] = minmax_benefit(df["AI_Readiness"])
    df["score_Resilience"] = minmax_benefit(df["Resilience"])

    w = np.array(weights, dtype=float)
    w = w / max(w.sum(), 1e-9)
    df["Composite_score"] = (
        w[0]*df["score_GDP"] + w[1]*df["score_Emission"] + w[2]*df["score_Risk"]
        + w[3]*df["score_NetJob"] + w[4]*df["score_Inclusion"]
    )
    df["Xếp hạng"] = df["Composite_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("Composite_score", ascending=False).reset_index(drop=True)


def metric_for_radar(metric_name):
    mapping = {
        "GDP gain": "score_GDP",
        "Phát thải thấp": "score_Emission",
        "Rủi ro thấp": "score_Risk",
        "NetJob": "score_NetJob",
        "Bao trùm": "score_Inclusion",
        "Digital Index": "score_Digital",
        "AI Readiness": "score_AI",
        "Resilience": "score_Resilience",
    }
    return mapping.get(metric_name, "score_GDP")


def risk_alerts(df):
    rows = []
    for _, r in df.iterrows():
        alerts, score = [], 0
        if r["Phat_thai"] >= df["Phat_thai"].quantile(0.70):
            alerts.append("Phát thải cao"); score += 1
        if r["CyberRisk"] >= 55:
            alerts.append("CyberRisk cao"); score += 2
        if r["JobRisk"] >= 25:
            alerts.append("Rủi ro dịch chuyển lao động"); score += 1
        if r["Bao_trum"] < 60:
            alerts.append("Bao trùm xã hội yếu"); score += 2

        if score >= 3:
            level = "Cao"
        elif score >= 1:
            level = "Trung bình"
        else:
            level = "Thấp"; alerts = ["Không có cảnh báo lớn"]

        rows.append({
            "Kịch bản": r["Kịch bản"],
            "Mức cảnh báo": level,
            "Cảnh báo": "; ".join(alerts),
            "Gợi ý": "Tăng H/D, bổ sung an sinh, kiểm soát AI/cyber" if level != "Thấp" else "Có thể dùng làm tham chiếu"
        })
    return pd.DataFrame(rows)


def plot_bar(df, y, title):
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.bar(df["Kịch bản"], df[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=17)
    ax.grid(axis="y", alpha=0.35)
    for i, v in enumerate(df[y]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    return fig


def plot_alloc(df, mode="share"):
    if mode == "share":
        cols, title, ylabel = ["K", "D", "AI", "H"], "Cơ cấu phân bổ theo kịch bản", "Tỷ trọng"
    else:
        cols, title, ylabel = ["K_budget", "D_budget", "AI_budget", "H_budget"], "Ngân sách quy đổi theo kịch bản", "Tỷ VND"
    fig, ax = plt.subplots(figsize=(9, 5))
    df.set_index("Kịch bản")[cols].plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=17)
    ax.grid(axis="y", alpha=0.35)
    return fig


def plot_bubble(df):
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            df, x="GDP_gain", y="Phat_thai", size="NetJob", color="Rui_ro",
            hover_name="Kịch bản",
            hover_data=["CyberRisk", "Bao_trum", "AI_Readiness", "Composite_score"],
            title="Bubble chart: GDP gain - Phát thải - NetJob - Rủi ro"
        )
        fig.update_layout(height=520)
        return fig

    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = df["NetJob"] / max(df["NetJob"].max(), 1) * 800
    sc = ax.scatter(df["GDP_gain"], df["Phat_thai"], s=sizes, c=df["Rui_ro"])
    ax.set_xlabel("GDP gain")
    ax.set_ylabel("Phát thải")
    ax.set_title("Bubble chart")
    fig.colorbar(sc, ax=ax, label="Rủi ro")
    return fig


def plot_radar(df, selected_metrics, selected_scenarios):
    if not selected_metrics:
        selected_metrics = ["GDP gain", "Phát thải thấp", "Rủi ro thấp", "NetJob"]
    if len(selected_metrics) < 3:
        selected_metrics = selected_metrics + ["GDP gain", "Phát thải thấp", "Rủi ro thấp"]
        selected_metrics = list(dict.fromkeys(selected_metrics))[:3]

    radar_df = df[df["Kịch bản"].isin(selected_scenarios)].copy()
    if radar_df.empty:
        radar_df = df.copy()

    cols = [metric_for_radar(m) for m in selected_metrics]

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for _, row in radar_df.iterrows():
            vals = [float(row[c]) for c in cols]
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1],
                theta=selected_metrics + selected_metrics[:1],
                fill="toself",
                name=row["Kịch bản"]
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar KPI tùy chọn",
            height=600,
            showlegend=True
        )
        return fig

    angles = np.linspace(0, 2*np.pi, len(selected_metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in radar_df.iterrows():
        vals = [float(row[c]) for c in cols]
        ax.plot(angles, vals + vals[:1], label=row["Kịch bản"])
        ax.fill(angles, vals + vals[:1], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_metrics)
    ax.set_title("Radar KPI tùy chọn")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    return fig


def test_table(df):
    use = df[df["Kịch bản"].isin(["S1. Truyền thống", "S3. AI dẫn dắt", "S5. Tối ưu cân bằng"])].copy()
    use["Test tổng tỷ trọng = 1"] = np.isclose(use[["K", "D", "AI", "H"]].sum(axis=1), 1.0)
    use["Test KPI không âm"] = use[["GDP_gain", "Phat_thai", "NetJob"]].min(axis=1) >= 0
    use["Test có score"] = use["Composite_score"].notna()
    return use[["Kịch bản", "GDP_gain", "Phat_thai", "Rui_ro", "NetJob", "Composite_score", "Test tổng tỷ trọng = 1", "Test KPI không âm", "Test có score"]]


def run():
    st.title("🇻🇳 Bài 12 – AIDEOM-VN Dashboard tích hợp")
    st.caption("Dashboard tương tác: chọn module, chọn kịch bản, chỉnh ngân sách/S5/trọng số, xem radar KPI và cảnh báo rủi ro.")

    with st.sidebar:
        st.markdown("## ⚙️ Điều khiển Dashboard")
        selected_modules = st.multiselect("1) Chọn module M1-M6", MODULES["Module"].tolist(), default=MODULES["Module"].tolist())
        selected_scenarios = st.multiselect("2) Chọn kịch bản", SCENARIOS, default=SCENARIOS)

        st.markdown("### 3) Tham số mô phỏng")
        total_budget = st.slider("Tổng ngân sách kịch bản, tỷ VND", 50000, 100000, 80000, 5000)
        shock = st.slider("Shock kinh tế", 0.0, 0.40, 0.00, 0.05)
        cyber_weight = st.slider("Độ nhạy CyberRisk", 0.5, 2.0, 1.0, 0.1)
        emission_weight = st.slider("Độ nhạy phát thải", 0.5, 2.0, 1.0, 0.1)

        st.markdown("### 4) Tùy chỉnh S5")
        s5_k = st.slider("S5: K", 0.0, 1.0, 0.35, 0.05)
        s5_d = st.slider("S5: D", 0.0, 1.0, 0.25, 0.05)
        s5_ai = st.slider("S5: AI", 0.0, 1.0, 0.20, 0.05)
        s5_h = st.slider("S5: H", 0.0, 1.0, 0.20, 0.05)

        st.markdown("### 5) Trọng số score")
        w_gdp = st.slider("w GDP", 0.0, 1.0, 0.30, 0.05)
        w_em = st.slider("w phát thải thấp", 0.0, 1.0, 0.20, 0.05)
        w_risk = st.slider("w rủi ro thấp", 0.0, 1.0, 0.15, 0.05)
        w_job = st.slider("w NetJob", 0.0, 1.0, 0.20, 0.05)
        w_inc = st.slider("w bao trùm", 0.0, 1.0, 0.15, 0.05)

        st.button("🚀 So sánh 5 kịch bản", type="primary")

    if not selected_scenarios:
        st.error("Bạn cần chọn ít nhất một kịch bản.")
        return

    full_df = compute_dashboard(
        total_budget=total_budget,
        shock=shock,
        cyber_weight=cyber_weight,
        emission_weight=emission_weight,
        s5_alloc=rebalance([s5_k, s5_d, s5_ai, s5_h]),
        weights=[w_gdp, w_em, w_risk, w_job, w_inc]
    )
    df = full_df[full_df["Kịch bản"].isin(selected_scenarios)].copy()
    df = df.sort_values("Composite_score", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    alerts = risk_alerts(df)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Tổng quan", "🧩 Module M1-M6", "💰 Phân bổ", "📊 So sánh kịch bản",
        "🕸️ Radar KPI", "🚨 Cảnh báo & test", "🤖 Khuyến nghị"
    ])

    with tab1:
        st.subheader("So sánh 5 kịch bản chính sách")
        st.success(f"Kịch bản dẫn đầu: {best['Kịch bản']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP gain", f"{best['GDP_gain']:,.0f}")
        c2.metric("Phát thải", f"{best['Phat_thai']:,.0f}")
        c3.metric("Rủi ro", f"{best['Rui_ro']:,.1f}")
        c4.metric("NetJob", f"{best['NetJob']:,.0f}")

        st.dataframe(
            df[["Kịch bản", "GDP_gain", "Phat_thai", "Rui_ro", "CyberRisk", "NetJob", "Bao_trum", "Digital_Index", "AI_Readiness", "Composite_score"]],
            use_container_width=True,
            height=330
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_bar(df, "GDP_gain", "GDP gain theo kịch bản")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig = plot_bubble(df)
            if PLOTLY_AVAILABLE:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
                plt.close(fig)

    with tab2:
        st.subheader("6 module AIDEOM-VN")
        st.dataframe(MODULES[MODULES["Module"].isin(selected_modules)], use_container_width=True, height=320)
        status = MODULES.copy()
        status["Trạng thái"] = np.where(status["Module"].isin(selected_modules), "Hiển thị", "Tạm ẩn")
        st.dataframe(status[["Module", "Tên", "Trạng thái", "Kỹ thuật chính"]], use_container_width=True)

    with tab3:
        st.subheader("Phân bổ ngân sách")
        st.dataframe(df[["Kịch bản", "K", "D", "AI", "H", "K_budget", "D_budget", "AI_budget", "H_budget"]], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_alloc(df, "share")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig = plot_alloc(df, "budget")
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.subheader("Kịch bản so sánh")
        kpi_options = st.multiselect(
            "Chọn KPI muốn hiển thị trong bảng",
            ["GDP_2030", "GDP_gain", "TFP_2030", "Digital_Index", "AI_Readiness", "Phat_thai", "Rui_ro", "CyberRisk", "DependencyRisk", "NetJob", "Bao_trum", "Resilience"],
            default=["GDP_gain", "Phat_thai", "Rui_ro", "NetJob"],
            key="kpi_table_select"
        )
        if kpi_options:
            st.dataframe(df[["Kịch bản"] + kpi_options + ["Composite_score"]], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_bar(df, "Phat_thai", "Phát thải theo kịch bản")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig = plot_bar(df, "NetJob", "NetJob theo kịch bản")
            st.pyplot(fig)
            plt.close(fig)

    with tab5:
        st.subheader("Radar KPI tùy chọn")
        left, right = st.columns([1, 2])
        with left:
            radar_metrics = st.multiselect(
                "Chọn trục radar",
                ["GDP gain", "Phát thải thấp", "Rủi ro thấp", "NetJob", "Bao trùm", "Digital Index", "AI Readiness", "Resilience"],
                default=["GDP gain", "Phát thải thấp", "Rủi ro thấp", "NetJob"],
                key="radar_metric_select"
            )
            radar_scenarios = st.multiselect(
                "Chọn kịch bản trên radar",
                df["Kịch bản"].tolist(),
                default=df["Kịch bản"].tolist(),
                key="radar_scenario_select"
            )
            st.info("Chọn được trục KPI và kịch bản trên radar.")
        with right:
            fig = plot_radar(df, radar_metrics, radar_scenarios)
            if PLOTLY_AVAILABLE:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
                plt.close(fig)
        st.dataframe(df[["Kịch bản", "score_GDP", "score_Emission", "score_Risk", "score_NetJob", "score_Inclusion", "score_Digital", "score_AI", "score_Resilience"]], use_container_width=True)

    with tab6:
        st.subheader("Cảnh báo rủi ro")
        st.dataframe(alerts, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_bar(df, "CyberRisk", "CyberRisk theo kịch bản")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig = plot_bar(df, "JobRisk", "Job displacement risk theo kịch bản")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Bộ test S1, S3, S5")
        tests = test_table(full_df)
        st.dataframe(tests, use_container_width=True)
        st.metric("Tỷ lệ test pass", f"{tests[['Test tổng tỷ trọng = 1', 'Test KPI không âm', 'Test có score']].values.mean()*100:.1f}%")

    with tab7:
        st.subheader("Khuyến nghị chính sách")
        rec = pd.DataFrame({
            "Ưu tiên": ["Tổng hợp cân bằng", "Tăng trưởng GDP", "Việc làm và bao trùm", "Rủi ro thấp", "Phát thải thấp"],
            "Kịch bản gợi ý": [
                df.iloc[0]["Kịch bản"],
                df.sort_values("GDP_gain", ascending=False).iloc[0]["Kịch bản"],
                df.sort_values("NetJob", ascending=False).iloc[0]["Kịch bản"],
                df.sort_values("Rui_ro").iloc[0]["Kịch bản"],
                df.sort_values("Phat_thai").iloc[0]["Kịch bản"],
            ],
            "Diễn giải": [
                "Điểm tổng hợp cao nhất theo trọng số đang chọn.",
                "Phù hợp khi ưu tiên tăng trưởng ngắn hạn.",
                "Phù hợp khi lo ngại thất nghiệp công nghệ.",
                "Phù hợp khi ưu tiên an ninh dữ liệu và chủ quyền số.",
                "Phù hợp với chuyển đổi xanh và cam kết khí hậu."
            ]
        })
        st.dataframe(rec, use_container_width=True)
        st.success(f"Theo cấu hình hiện tại, dashboard khuyến nghị **{best['Kịch bản']}**.")


if __name__ == "__main__":
    run()
