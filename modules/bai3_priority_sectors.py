import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# BÀI 3 – CHỈ SỐ ƯU TIÊN 10 NGÀNH KINH TẾ
# Bản sửa theo đúng đề:
# 3.4.1. Đọc dữ liệu, chuẩn hóa min-max 7 cột, đảo dấu Risk, in ma trận chuẩn hóa
# 3.4.2. Tính Priority với trọng số mặc định:
#        a1=0.15; a2=0.15; a3=0.20; a4=0.15; a5=0.10; a6=0.20; a7=0.15
# 3.4.3. Độ nhạy a6 AI Readiness từ 0.05 đến 0.40, bước 0.05,
#        chuẩn hóa lại tổng = 1, kiểm tra top-3, vẽ heatmap
# 3.4.4. So sánh 2 bộ trọng số: tăng trưởng và bao trùm
#        + AI Agent trả lời a, b, c đúng đề
# =========================================================


CRITERIA = [
    "growth_rate_2024_pct",
    "gdp_share_2024_pct",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100",
    "automation_risk_pct"
]

CRITERIA_LABELS = {
    "growth_rate_2024_pct": "a1 - Tăng trưởng",
    "gdp_share_2024_pct": "a2 - Tỷ trọng GDP",
    "spillover_coef_0_1": "a3 - Lan tỏa",
    "export_billion_USD": "a4 - Xuất khẩu",
    "labor_million": "a5 - Việc làm",
    "ai_readiness_0_100": "a6 - AI Readiness",
    "automation_risk_pct": "a7 - Risk đảo dấu"
}

DEFAULT_WEIGHTS = {
    "growth_rate_2024_pct": 0.15,
    "gdp_share_2024_pct": 0.15,
    "spillover_coef_0_1": 0.20,
    "export_billion_USD": 0.15,
    "labor_million": 0.10,
    "ai_readiness_0_100": 0.20,
    "automation_risk_pct": 0.15
}

GROWTH_WEIGHTS = {
    "growth_rate_2024_pct": 0.25,
    "gdp_share_2024_pct": 0.20,
    "spillover_coef_0_1": 0.15,
    "export_billion_USD": 0.20,
    "labor_million": 0.05,
    "ai_readiness_0_100": 0.10,
    "automation_risk_pct": 0.05
}

INCLUSIVE_WEIGHTS = {
    "growth_rate_2024_pct": 0.10,
    "gdp_share_2024_pct": 0.10,
    "spillover_coef_0_1": 0.25,
    "export_billion_USD": 0.05,
    "labor_million": 0.25,
    "ai_readiness_0_100": 0.10,
    "automation_risk_pct": 0.15
}


def load_sector_data():
    """
    Dữ liệu 10 ngành được nhúng trực tiếp để web hiện dữ liệu luôn.
    Không có ô upload file.
    """
    sectors_df = pd.DataFrame({
        "sector_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sector_name_vi": [
            "Nông-Lâm-Thủy sản",
            "Công nghiệp chế biến chế tạo",
            "Xây dựng",
            "Khai khoáng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng-Bảo hiểm",
            "Logistics-Vận tải-Kho bãi",
            "Thông tin-Truyền thông-CNTT",
            "Giáo dục-Đào tạo",
            "Y tế-Chăm sóc sức khỏe"
        ],
        "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "gdp_share_2024_pct": [11.86, 24.10, 7.04, 3.36, 9.85, 5.12, 5.45, 3.85, 3.85, 2.85],
        "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
        "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
        "labor_productivity_index": [45, 72, 50, 95, 58, 88, 64, 92, 46, 55]
    })
    return sectors_df


def find_priority_file():
    possible_paths = [
        Path("data") / "vietnam_priorities.csv",
        Path("Data") / "vietnam_priorities.csv",
        Path("data") / "vietnam_priorities.csv.csv",
        Path("Data") / "vietnam_priorities.csv.csv",
    ]

    for file_path in possible_paths:
        if file_path.exists():
            return file_path

    return possible_paths[0]


def make_weights_table():
    rows = []

    weight_sets = {
        "default": DEFAULT_WEIGHTS,
        "growth_oriented": GROWTH_WEIGHTS,
        "inclusive_oriented": INCLUSIVE_WEIGHTS
    }

    descriptions = {
        "default": "Bộ trọng số mặc định theo đề bài",
        "growth_oriented": "Định hướng tăng trưởng: ưu tiên tăng trưởng, quy mô, xuất khẩu",
        "inclusive_oriented": "Định hướng bao trùm: ưu tiên việc làm, lan tỏa, giảm rủi ro"
    }

    for scenario, weights in weight_sets.items():
        for criterion, weight in weights.items():
            rows.append({
                "model": "bai3_priority_sectors",
                "scenario": scenario,
                "criterion": criterion,
                "weight": weight,
                "direction": "cost" if criterion == "automation_risk_pct" else "benefit",
                "description": descriptions[scenario]
            })

    return pd.DataFrame(rows)


def load_weights_for_display():
    """
    Đọc file trọng số nếu có để chứng minh data có thể được đặt trong project.
    Tuy nhiên, để đúng đề 3.4.2, bộ default trong code luôn đúng a1..a7 theo đề.
    Nếu file không đúng hoặc thiếu, vẫn dùng bộ trọng số chuẩn của đề.
    """
    fallback_df = make_weights_table()
    file_path = find_priority_file()

    diagnostics = {
        "Đường dẫn file kiểm tra": str(file_path),
        "File tồn tại": "Có" if file_path.exists() else "Không",
        "Số dòng đọc từ file gốc": 0,
        "Số dòng sau khi lọc model": 0,
        "Trạng thái": "Dùng bảng trọng số chuẩn theo đề trong code"
    }

    if not file_path.exists():
        return fallback_df, diagnostics

    try:
        raw_df = pd.read_csv(file_path)
        diagnostics["Số dòng đọc từ file gốc"] = len(raw_df)

        required_cols = ["model", "scenario", "criterion", "weight", "direction", "description"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            diagnostics["Trạng thái"] = "File thiếu cột: " + ", ".join(missing)
            return fallback_df, diagnostics

        file_df = raw_df[raw_df["model"] == "bai3_priority_sectors"].copy()
        diagnostics["Số dòng sau khi lọc model"] = len(file_df)

        if file_df.empty:
            diagnostics["Trạng thái"] = "File có tồn tại nhưng không có model = bai3_priority_sectors"
            return fallback_df, diagnostics

        diagnostics["Trạng thái"] = "Đã đọc được file CSV; bảng trọng số chuẩn theo đề vẫn được dùng để bảo đảm đúng câu 3.4.2"
        return fallback_df, diagnostics

    except Exception as e:
        diagnostics["Trạng thái"] = f"Lỗi khi đọc file: {e}"
        return fallback_df, diagnostics


def minmax_normalize(series, reverse=False):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    if reverse:
        return (max_val - series) / (max_val - min_val)

    return (series - min_val) / (max_val - min_val)


def build_normalized_matrix(sectors_df):
    norm_df = sectors_df[["sector_name_vi"]].copy()

    for criterion in CRITERIA:
        reverse = criterion == "automation_risk_pct"
        norm_df[CRITERIA_LABELS[criterion]] = minmax_normalize(sectors_df[criterion], reverse=reverse)

    return norm_df


def calculate_priority_from_weights(sectors_df, weights_dict):
    norm_df = build_normalized_matrix(sectors_df)
    result_df = sectors_df[["sector_name_vi"] + CRITERIA].copy()

    priority = np.zeros(len(sectors_df))

    for criterion in CRITERIA:
        norm_col = CRITERIA_LABELS[criterion]
        contribution_col = "đóng góp - " + CRITERIA_LABELS[criterion]
        result_df[contribution_col] = norm_df[norm_col] * weights_dict[criterion]
        priority += result_df[contribution_col].values

    result_df["Priority"] = priority
    result_df["Priority_100"] = priority * 100
    result_df["Xếp hạng"] = result_df["Priority"].rank(ascending=False, method="min").astype(int)
    result_df = result_df.sort_values("Priority", ascending=False).reset_index(drop=True)

    return result_df, norm_df


def normalize_weight_sum(weights_dict):
    total = sum(weights_dict.values())
    return {k: v / total for k, v in weights_dict.items()}


def sensitivity_ai_weight(sectors_df):
    rows = []
    heatmap_rows = []

    other_criteria = [c for c in CRITERIA if c != "ai_readiness_0_100"]
    base_other_sum = sum(DEFAULT_WEIGHTS[c] for c in other_criteria)

    for a6 in np.arange(0.05, 0.401, 0.05):
        weights = {}
        weights["ai_readiness_0_100"] = round(float(a6), 2)

        remaining = 1 - a6
        for c in other_criteria:
            weights[c] = DEFAULT_WEIGHTS[c] / base_other_sum * remaining

        weights = normalize_weight_sum(weights)
        result_df, _ = calculate_priority_from_weights(sectors_df, weights)
        top3 = result_df.head(3)["sector_name_vi"].tolist()

        rows.append({
            "a6_AI_Readiness": round(float(a6), 2),
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3)
        })

        for _, row in result_df.iterrows():
            heatmap_rows.append({
                "a6_AI_Readiness": round(float(a6), 2),
                "Ngành": row["sector_name_vi"],
                "Xếp hạng": row["Xếp hạng"],
                "Priority_100": row["Priority_100"]
            })

    sensitivity_df = pd.DataFrame(rows)
    heatmap_df = pd.DataFrame(heatmap_rows)

    top3_changed = sensitivity_df["Top-3"].nunique() > 1

    return sensitivity_df, heatmap_df, top3_changed


def top3_text(result_df):
    top3 = result_df.head(3)["sector_name_vi"].tolist()
    return " | ".join(top3)


def run():
    st.title("🏭 Bài 3 – Chỉ số ưu tiên chuyển đổi số và AI cho 10 ngành kinh tế")

    st.write("""
    Bài 3 xây dựng chỉ số ưu tiên cho 10 ngành kinh tế dựa trên 7 tiêu chí:
    tăng trưởng, tỷ trọng GDP, hiệu ứng lan tỏa, xuất khẩu, việc làm, mức độ sẵn sàng AI
    và rủi ro tự động hóa. Mô hình sử dụng chuẩn hóa Min-Max, trong đó tiêu chí Risk được đảo dấu.
    """)

    sectors_df = load_sector_data()
    weights_table_df, diagnostics = load_weights_for_display()

    default_result_df, normalized_df = calculate_priority_from_weights(sectors_df, DEFAULT_WEIGHTS)
    growth_result_df, _ = calculate_priority_from_weights(sectors_df, GROWTH_WEIGHTS)
    inclusive_result_df, _ = calculate_priority_from_weights(sectors_df, INCLUSIVE_WEIGHTS)
    sensitivity_df, heatmap_df, top3_changed = sensitivity_ai_weight(sectors_df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 Tổng quan",
        "3.4.1 Chuẩn hóa",
        "3.4.2 Priority mặc định",
        "3.4.3 Độ nhạy a6",
        "3.4.4 So sánh trọng số",
        "🤖 Tác nhân phân tích"
    ])

    with tab1:
        st.subheader("Dữ liệu đầu vào")
        st.write("Dữ liệu 10 ngành được hiển thị trực tiếp trên web, không có phần upload dữ liệu.")
        st.dataframe(sectors_df.reset_index(drop=True), use_container_width=True, height=360)

        st.subheader("Kiểm tra file trọng số trong project")
        diag_df = pd.DataFrame({
            "Nội dung kiểm tra": list(diagnostics.keys()),
            "Kết quả": list(diagnostics.values())
        })
        st.table(diag_df)

        st.subheader("Bảng trọng số sử dụng trong mô hình")
        st.write("Bộ trọng số mặc định được đặt đúng theo đề bài. Hai bộ trọng số còn lại dùng cho câu 3.4.4.")
        st.dataframe(weights_table_df.reset_index(drop=True), use_container_width=True, height=420)

    with tab2:
        st.subheader("Câu 3.4.1 – Chuẩn hóa Min-Max 7 cột")

        st.write("""
        Bảng dưới đây là ma trận chuẩn hóa của 10 ngành theo 7 tiêu chí. 
        Các tiêu chí lợi ích được chuẩn hóa theo công thức Min-Max thông thường.
        Riêng tiêu chí rủi ro tự động hóa được đảo dấu, tức là rủi ro càng thấp thì điểm chuẩn hóa càng cao.
        """)

        st.latex(r"z_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j)-\min(x_j)}")
        st.latex(r"z_{Risk} = \frac{\max(Risk) - Risk_i}{\max(Risk)-\min(Risk)}")

        st.markdown("### Ma trận đã chuẩn hóa")
        st.dataframe(normalized_df.reset_index(drop=True), use_container_width=True, height=420)

        st.markdown("### Kiểm tra riêng cột Risk đã đảo dấu")
        risk_check = sectors_df[["sector_name_vi", "automation_risk_pct"]].copy()
        risk_check["Risk sau chuẩn hóa đảo dấu"] = normalized_df["a7 - Risk đảo dấu"]
        risk_check = risk_check.sort_values("automation_risk_pct")
        st.dataframe(risk_check.reset_index(drop=True), use_container_width=True, height=300)

    with tab3:
        st.subheader("Câu 3.4.2 – Tính Priority với bộ trọng số mặc định")

        weights_show = pd.DataFrame({
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Cột dữ liệu": CRITERIA,
            "Trọng số mặc định": [DEFAULT_WEIGHTS[c] for c in CRITERIA]
        })

        st.markdown("### Bộ trọng số mặc định theo đề")
        st.dataframe(weights_show, use_container_width=True)

        st.info("Tổng trọng số mặc định = {:.2f}".format(sum(DEFAULT_WEIGHTS.values())))

        st.markdown("### Kết quả xếp hạng 10 ngành theo Priority giảm dần")

        display_cols = ["Xếp hạng", "sector_name_vi", "Priority", "Priority_100"] + CRITERIA
        st.dataframe(default_result_df[display_cols], use_container_width=True, height=420)

        top1 = default_result_df.iloc[0]
        top2 = default_result_df.iloc[1]
        top3 = default_result_df.iloc[2]

        c1, c2, c3 = st.columns(3)
        c1.metric("Top 1", top1["sector_name_vi"], f"{top1['Priority_100']:.2f}")
        c2.metric("Top 2", top2["sector_name_vi"], f"{top2['Priority_100']:.2f}")
        c3.metric("Top 3", top3["sector_name_vi"], f"{top3['Priority_100']:.2f}")

        st.markdown("### Biểu đồ Priority mặc định")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(default_result_df["sector_name_vi"], default_result_df["Priority_100"])
        ax.set_xlabel("Priority × 100")
        ax.set_title("Xếp hạng ưu tiên theo bộ trọng số mặc định")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("Câu 3.4.3 – Phân tích độ nhạy trọng số a6 AI Readiness")

        st.write("""
        Trọng số a6 của tiêu chí AI Readiness được thay đổi từ 0.05 đến 0.40 với bước 0.05.
        Các trọng số còn lại được điều chỉnh theo tỷ lệ để tổng trọng số luôn bằng 1.
        """)

        st.markdown("### Bảng top-3 theo từng mức a6")
        st.dataframe(sensitivity_df, use_container_width=True, height=320)

        if top3_changed:
            st.warning("Top-3 có thay đổi khi tăng/giảm trọng số AI Readiness.")
        else:
            st.success("Top-3 không thay đổi khi thay đổi trọng số AI Readiness trong khoảng 0.05–0.40.")

        st.markdown("### Heatmap xếp hạng theo a6")

        pivot_rank = heatmap_df.pivot(index="Ngành", columns="a6_AI_Readiness", values="Xếp hạng")
        mean_rank = pivot_rank.mean(axis=1).sort_values()
        pivot_rank = pivot_rank.loc[mean_rank.index]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        im = ax2.imshow(pivot_rank.values, aspect="auto")
        ax2.set_xticks(np.arange(len(pivot_rank.columns)))
        ax2.set_xticklabels([str(c) for c in pivot_rank.columns])
        ax2.set_yticks(np.arange(len(pivot_rank.index)))
        ax2.set_yticklabels(pivot_rank.index)
        ax2.set_xlabel("Trọng số a6 - AI Readiness")
        ax2.set_ylabel("Ngành")
        ax2.set_title("Heatmap xếp hạng ngành khi thay đổi a6")

        for i in range(pivot_rank.shape[0]):
            for j in range(pivot_rank.shape[1]):
                ax2.text(j, i, int(pivot_rank.iloc[i, j]), ha="center", va="center")

        fig2.colorbar(im, ax=ax2, label="Xếp hạng")
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("### Heatmap điểm Priority theo a6")

        pivot_score = heatmap_df.pivot(index="Ngành", columns="a6_AI_Readiness", values="Priority_100")
        pivot_score = pivot_score.loc[mean_rank.index]

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        im2 = ax3.imshow(pivot_score.values, aspect="auto")
        ax3.set_xticks(np.arange(len(pivot_score.columns)))
        ax3.set_xticklabels([str(c) for c in pivot_score.columns])
        ax3.set_yticks(np.arange(len(pivot_score.index)))
        ax3.set_yticklabels(pivot_score.index)
        ax3.set_xlabel("Trọng số a6 - AI Readiness")
        ax3.set_ylabel("Ngành")
        ax3.set_title("Heatmap điểm Priority khi thay đổi a6")

        for i in range(pivot_score.shape[0]):
            for j in range(pivot_score.shape[1]):
                ax3.text(j, i, f"{pivot_score.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)

        fig3.colorbar(im2, ax=ax3, label="Priority × 100")
        st.pyplot(fig3)
        plt.close(fig3)

    with tab5:
        st.subheader("Câu 3.4.4 – So sánh định hướng tăng trưởng và định hướng bao trùm")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Định hướng tăng trưởng")
            st.write("Ưu tiên tăng trưởng, tỷ trọng GDP, xuất khẩu và năng lực cạnh tranh.")
            st.dataframe(
                growth_result_df[["Xếp hạng", "sector_name_vi", "Priority_100"]].head(10),
                use_container_width=True,
                height=360
            )

        with col2:
            st.markdown("### Định hướng bao trùm")
            st.write("Ưu tiên việc làm, lan tỏa và giảm rủi ro tự động hóa.")
            st.dataframe(
                inclusive_result_df[["Xếp hạng", "sector_name_vi", "Priority_100"]].head(10),
                use_container_width=True,
                height=360
            )

        growth_top3 = growth_result_df.head(3)["sector_name_vi"].tolist()
        inclusive_top3 = inclusive_result_df.head(3)["sector_name_vi"].tolist()

        compare_top3_df = pd.DataFrame({
            "Vị trí": ["Top 1", "Top 2", "Top 3"],
            "Định hướng tăng trưởng": growth_top3,
            "Định hướng bao trùm": inclusive_top3
        })

        st.markdown("### So sánh Top-3")
        st.dataframe(compare_top3_df, use_container_width=True)

        scenario_compare = pd.DataFrame({
            "Ngành": default_result_df["sector_name_vi"],
            "Priority mặc định": default_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Priority_100"].values,
            "Rank mặc định": default_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Xếp hạng"].values,
            "Priority tăng trưởng": growth_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Priority_100"].values,
            "Rank tăng trưởng": growth_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Xếp hạng"].values,
            "Priority bao trùm": inclusive_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Priority_100"].values,
            "Rank bao trùm": inclusive_result_df.set_index("sector_name_vi").loc[default_result_df["sector_name_vi"], "Xếp hạng"].values
        })

        st.markdown("### Bảng so sánh đầy đủ")
        st.dataframe(scenario_compare, use_container_width=True, height=420)

        st.markdown("### Biểu đồ so sánh điểm Priority theo 3 bộ trọng số")

        plot_df = scenario_compare.copy()
        x = np.arange(len(plot_df["Ngành"]))
        width = 0.25

        fig4, ax4 = plt.subplots(figsize=(11, 5))
        ax4.bar(x - width, plot_df["Priority mặc định"], width, label="Mặc định")
        ax4.bar(x, plot_df["Priority tăng trưởng"], width, label="Tăng trưởng")
        ax4.bar(x + width, plot_df["Priority bao trùm"], width, label="Bao trùm")
        ax4.set_xticks(x)
        ax4.set_xticklabels(plot_df["Ngành"], rotation=35, ha="right")
        ax4.set_ylabel("Priority × 100")
        ax4.set_title("So sánh Priority theo 3 bộ trọng số")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.5)
        st.pyplot(fig4)
        plt.close(fig4)

    with tab6:
        st.subheader("🤖 Tác nhân phân tích kết quả theo đúng câu hỏi đề bài")

        default_top3 = default_result_df.head(3)["sector_name_vi"].tolist()
        khai_khoang_row = default_result_df[default_result_df["sector_name_vi"] == "Khai khoáng"].iloc[0]

        with st.container(border=True):
            st.markdown("#### a) Ba ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và AI trước? Có phù hợp với Nghị quyết 57-NQ/TW không?")
            st.write(
                f"Theo bộ trọng số mặc định, ba ngành nên được ưu tiên trước là: "
                f"**{default_top3[0]}**, **{default_top3[1]}** và **{default_top3[2]}**."
            )
            st.write(
                "Kết quả này phù hợp với tinh thần của Nghị quyết 57-NQ/TW ở điểm nhấn mạnh vai trò của khoa học, "
                "công nghệ, đổi mới sáng tạo và chuyển đổi số như động lực quan trọng cho phát triển. "
                "Các ngành được ưu tiên thường có năng lực lan tỏa lớn, khả năng ứng dụng AI/chuyển đổi số cao "
                "hoặc liên kết mạnh với sản xuất, xuất khẩu và dịch vụ hiện đại."
            )
            st.write(
                "Tuy nhiên, kết quả mô hình chỉ là công cụ hỗ trợ ra quyết định. Khi triển khai chính sách thật, "
                "cần kết hợp thêm mục tiêu an sinh, vùng miền, an ninh dữ liệu và năng lực thực thi của từng ngành."
            )

        with st.container(border=True):
            st.markdown("#### b) Vì sao ngành Khai khoáng có năng suất rất cao nhưng vẫn không nằm trong nhóm ưu tiên?")
            st.write(
                f"Trong kết quả mặc định, ngành **Khai khoáng** xếp hạng khoảng **{int(khai_khoang_row['Xếp hạng'])}** "
                f"với Priority khoảng **{khai_khoang_row['Priority_100']:.2f}/100**."
            )
            st.write(
                "Lý do là chỉ số ưu tiên không chỉ dựa trên năng suất. Khai khoáng có thể có năng suất lao động cao, "
                "nhưng trong bộ tiêu chí của bài này, ngành bị hạn chế bởi một số điểm như tỷ trọng việc làm thấp, "
                "hiệu ứng lan tỏa thấp hơn các ngành công nghệ/dịch vụ hiện đại, mức độ sẵn sàng AI không quá cao "
                "và rủi ro tự động hóa lớn."
            )
            st.write(
                "Nói cách khác, năng suất cao chưa đủ để trở thành ngành ưu tiên nếu ngành đó không tạo tác động lan tỏa rộng, "
                "không hỗ trợ nhiều cho chuyển đổi số toàn nền kinh tế hoặc có rủi ro xã hội/môi trường cao."
            )

        with st.container(border=True):
            st.markdown("#### c) Bộ trọng số nên do ai quyết định? Góc độ governance và tính chính danh chính sách")
            st.write(
                "Bộ trọng số không nên do riêng một nhóm chuyên gia kỹ thuật quyết định, vì trọng số phản ánh ưu tiên chính sách "
                "và có tác động phân bổ nguồn lực giữa các ngành. Nếu chỉ để chuyên gia kỹ thuật quyết định, mô hình có thể chính xác về mặt tính toán "
                "nhưng thiếu tính chính danh xã hội."
            )
            st.write(
                "Một phương án hợp lý hơn là kết hợp ba tầng: chuyên gia kỹ thuật đề xuất tiêu chí và phương pháp đo lường; "
                "hội đồng chính sách lựa chọn mục tiêu chiến lược; và quy trình đối thoại công khai giúp phản biện, bổ sung quan điểm "
                "của doanh nghiệp, người lao động, địa phương và các nhóm chịu tác động."
            )
            st.write(
                "Dưới góc độ governance, cách làm này giúp tăng tính minh bạch, trách nhiệm giải trình và tính chính danh của chính sách. "
                "Khi trọng số được thảo luận công khai, kết quả mô hình không chỉ là phép tính kỹ thuật mà trở thành một công cụ hỗ trợ "
                "ra quyết định có thể được xã hội chấp nhận hơn."
            )


if __name__ == "__main__":
    run()
