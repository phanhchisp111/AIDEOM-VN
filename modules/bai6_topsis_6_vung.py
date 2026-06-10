import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# BÀI 6 – TOPSIS XẾP HẠNG 6 VÙNG KINH TẾ VIỆT NAM
# Bám đúng đề:
# 6.4.1 TOPSIS bằng numpy với trọng số chuyên gia
# 6.4.2 Entropy weights và so sánh xếp hạng
# 6.4.3 Độ nhạy w_AI từ 0.10 đến 0.40
# 6.4.4 AHP đơn giản để so sánh với TOPSIS
# 6.5 AI Agent trả lời a, b, c, d
# Không upload file; dữ liệu hiện trực tiếp trên web.
# =========================================================


CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef"
]

CRITERIA_LABELS = {
    "grdp_per_capita_million_VND": "GRDP/người",
    "fdi_registered_billion_USD": "FDI",
    "digital_index_0_100": "Digital Index",
    "ai_readiness_0_100": "AI Readiness",
    "trained_labor_pct": "LĐ đào tạo",
    "rd_intensity_pct": "R&D/GRDP",
    "internet_penetration_pct": "Internet",
    "gini_coef": "Gini"
}

IS_BENEFIT = np.array([True, True, True, True, True, True, True, False])

EXPERT_WEIGHTS = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])


def find_regions_file():
    possible_paths = [
        Path("data") / "vietnam_regions_2024.csv",
        Path("Data") / "vietnam_regions_2024.csv",
        Path("data") / "vietnam_regions_2024.csv.csv",
        Path("Data") / "vietnam_regions_2024.csv.csv",
    ]

    for file_path in possible_paths:
        if file_path.exists():
            return file_path

    return possible_paths[0]


def default_regions_data():
    return pd.DataFrame({
        "region_name_vi": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long"
        ],
        "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index_0_100": [38, 78, 55, 32, 82, 48],
        "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
        "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
        "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392]
    })


def load_regions_data():
    """
    Đọc file vietnam_regions_2024.csv nếu có, nhưng không có ô upload.
    Nếu file thiếu cột, dùng dữ liệu mặc định đúng theo đề.
    """
    file_path = find_regions_file()
    fallback_df = default_regions_data()

    diagnostics = {
        "Đường dẫn file kiểm tra": str(file_path),
        "File tồn tại": "Có" if file_path.exists() else "Không",
        "Số dòng đọc được": 0,
        "Số cột yêu cầu có đủ": "Chưa kiểm tra",
        "Trạng thái": "Dùng dữ liệu mặc định đúng theo đề"
    }

    if not file_path.exists():
        return fallback_df, diagnostics

    try:
        raw_df = pd.read_csv(file_path)
        diagnostics["Số dòng đọc được"] = len(raw_df)

        required_cols = ["region_name_vi"] + CRITERIA
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            diagnostics["Số cột yêu cầu có đủ"] = "Không"
            diagnostics["Trạng thái"] = "File thiếu cột: " + ", ".join(missing) + ". Dùng dữ liệu mặc định theo đề."
            return fallback_df, diagnostics

        diagnostics["Số cột yêu cầu có đủ"] = "Có"

        df = raw_df[required_cols].head(6).copy()

        for c in CRITERIA:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if df[CRITERIA].isna().any().any():
            diagnostics["Trạng thái"] = "File có giá trị không đọc được. Dùng dữ liệu mặc định theo đề."
            return fallback_df, diagnostics

        diagnostics["Trạng thái"] = "Đã đọc dữ liệu từ file vietnam_regions_2024.csv"
        return df.reset_index(drop=True), diagnostics

    except Exception as e:
        diagnostics["Trạng thái"] = f"Lỗi khi đọc file: {e}. Dùng dữ liệu mặc định theo đề."
        return fallback_df, diagnostics


def vector_normalize(X):
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1, denom)
    return X / denom


def topsis(df, weights):
    X = df[CRITERIA].values.astype(float)

    # Bước 1: chuẩn hóa vector
    R = vector_normalize(X)

    # Bước 2: ma trận chuẩn hóa có trọng số
    V = R * weights

    # Bước 3: ideal và anti-ideal
    A_star = np.where(IS_BENEFIT, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(IS_BENEFIT, V.min(axis=0), V.max(axis=0))

    # Bước 4: khoảng cách Euclide
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # Bước 5: hệ số gần gũi
    C_star = S_neg / (S_star + S_neg)

    result = df.copy()
    result["S_star"] = S_star
    result["S_neg"] = S_neg
    result["TOPSIS_score"] = C_star
    result["rank"] = result["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)

    norm_df = pd.DataFrame(R, columns=[CRITERIA_LABELS[c] for c in CRITERIA])
    norm_df.insert(0, "Vùng", df["region_name_vi"])

    weighted_df = pd.DataFrame(V, columns=[CRITERIA_LABELS[c] for c in CRITERIA])
    weighted_df.insert(0, "Vùng", df["region_name_vi"])

    ideal_df = pd.DataFrame({
        "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
        "Loại": ["Benefit" if b else "Cost" for b in IS_BENEFIT],
        "A* lý tưởng tốt": A_star,
        "A- lý tưởng xấu": A_neg
    })

    return result, norm_df, weighted_df, ideal_df


def entropy_weights(df):
    X = df[CRITERIA].values.astype(float).copy()

    # Với Gini là cost, đảo chiều để Entropy hiểu là giá trị càng cao càng tốt
    for j, is_benefit in enumerate(IS_BENEFIT):
        if not is_benefit:
            max_val = X[:, j].max()
            min_val = X[:, j].min()
            X[:, j] = max_val + min_val - X[:, j]

    col_sum = X.sum(axis=0)
    col_sum = np.where(col_sum == 0, 1, col_sum)

    P = X / col_sum
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    weights = d / d.sum()

    entropy_df = pd.DataFrame({
        "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
        "Entropy E_j": E,
        "Độ phân tán d_j": d,
        "Trọng số Entropy": weights
    })

    return weights, entropy_df


def sensitivity_ai(df):
    rows = []
    heatmap_rows = []

    ai_idx = CRITERIA.index("ai_readiness_0_100")
    base_other = EXPERT_WEIGHTS.copy()
    base_other[ai_idx] = 0
    base_other_sum = base_other.sum()

    for w_ai in np.arange(0.10, 0.401, 0.05):
        weights = EXPERT_WEIGHTS.copy()
        remaining = 1 - w_ai

        for j in range(len(weights)):
            if j == ai_idx:
                weights[j] = w_ai
            else:
                weights[j] = EXPERT_WEIGHTS[j] / base_other_sum * remaining

        weights = weights / weights.sum()

        result, _, _, _ = topsis(df, weights)
        top3 = result.head(3)["region_name_vi"].tolist()

        rows.append({
            "w_AI": round(float(w_ai), 2),
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3)
        })

        for _, row in result.iterrows():
            heatmap_rows.append({
                "w_AI": round(float(w_ai), 2),
                "Vùng": row["region_name_vi"],
                "Rank": row["rank"],
                "TOPSIS_score": row["TOPSIS_score"]
            })

    sensitivity_df = pd.DataFrame(rows)
    heatmap_df = pd.DataFrame(heatmap_rows)
    top3_stable = sensitivity_df["Top-3"].nunique() == 1

    return sensitivity_df, heatmap_df, top3_stable


def ahp_weights_simple():
    """
    AHP đơn giản: dùng ma trận so sánh cặp được xây dựng nhất quán gần đúng từ mức ưu tiên.
    Mức ưu tiên phản ánh AI Readiness cao nhất, tiếp đến Digital, R&D, trained labor, GRDP/FDI, Internet, Gini.
    """
    priority = np.array([0.10, 0.10, 0.15, 0.22, 0.14, 0.15, 0.06, 0.08])
    priority = priority / priority.sum()

    n = len(priority)
    pairwise = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pairwise[i, j] = priority[i] / priority[j]

    eigvals, eigvecs = np.linalg.eig(pairwise)
    max_idx = np.argmax(eigvals.real)
    weights = eigvecs[:, max_idx].real
    weights = np.abs(weights)
    weights = weights / weights.sum()

    lam_max = eigvals[max_idx].real
    ci = (lam_max - n) / (n - 1)
    ri_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = ri_dict[n]
    cr = ci / ri if ri != 0 else 0

    ahp_df = pd.DataFrame({
        "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
        "Trọng số AHP": weights
    })

    return weights, pairwise, ahp_df, ci, cr


def compare_rank(expert_result, entropy_result, ahp_result=None):
    base = expert_result[["region_name_vi", "rank", "TOPSIS_score"]].rename(columns={
        "rank": "Rank chuyên gia",
        "TOPSIS_score": "Score chuyên gia"
    })

    entropy = entropy_result[["region_name_vi", "rank", "TOPSIS_score"]].rename(columns={
        "rank": "Rank Entropy",
        "TOPSIS_score": "Score Entropy"
    })

    compare = base.merge(entropy, on="region_name_vi", how="left")
    compare["Thay đổi rank Entropy - chuyên gia"] = compare["Rank Entropy"] - compare["Rank chuyên gia"]

    if ahp_result is not None:
        ahp = ahp_result[["region_name_vi", "rank", "TOPSIS_score"]].rename(columns={
            "rank": "Rank AHP",
            "TOPSIS_score": "Score AHP"
        })

        compare = compare.merge(ahp, on="region_name_vi", how="left")
        compare["Thay đổi rank AHP - chuyên gia"] = compare["Rank AHP"] - compare["Rank chuyên gia"]

    return compare.sort_values("Rank chuyên gia")


def plot_rank_bar(result, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = result.sort_values("TOPSIS_score", ascending=True)

    ax.barh(plot_df["region_name_vi"], plot_df["TOPSIS_score"])
    ax.set_xlabel("C* TOPSIS")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.5)

    for i, value in enumerate(plot_df["TOPSIS_score"]):
        ax.text(value, i, f"{value:.3f}", va="center")

    return fig


def run():
    st.title("🌏 Bài 6 – TOPSIS xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI")

    st.write("""
    Bài 6 áp dụng phương pháp TOPSIS để xếp hạng 6 vùng kinh tế - xã hội Việt Nam theo mức độ sẵn sàng
    triển khai các trung tâm AI và sandbox dữ liệu. Bài cũng sử dụng Entropy để xác định trọng số khách quan
    và AHP đơn giản để so sánh kết quả.
    """)

    df, diagnostics = load_regions_data()

    expert_result, norm_df, weighted_df, ideal_df = topsis(df, EXPERT_WEIGHTS)
    ent_w, entropy_detail_df = entropy_weights(df)
    entropy_result, _, _, _ = topsis(df, ent_w)

    sensitivity_df, heatmap_df, top3_stable = sensitivity_ai(df)

    ahp_w, pairwise, ahp_weight_df, ci, cr = ahp_weights_simple()
    ahp_result, _, _, _ = topsis(df, ahp_w)

    compare_df = compare_rank(expert_result, entropy_result, ahp_result)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 Mô hình",
        "📋 Dữ liệu",
        "6.4.1 TOPSIS",
        "6.4.2 Entropy",
        "6.4.3 Độ nhạy",
        "🤖 6.4.4 & 6.5"
    ])

    with tab1:
        st.subheader("6.1. Bối cảnh Việt Nam")
        st.write("""
        Theo Quyết định 127/QĐ-TTg về Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI đến năm 2030,
        Việt Nam đặt mục tiêu trở thành trung tâm AI của ASEAN. Do nguồn lực có hạn, cần xếp hạng các vùng để
        xác định nơi ưu tiên triển khai trung tâm AI trước.
        """)

        st.subheader("6.2. Lý thuyết TOPSIS")
        st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}")
        st.latex(r"v_{ij}=w_j r_{ij}")
        st.latex(r"C_i^*=\frac{S_i^-}{S_i^*+S_i^-}")

        st.write("""
        TOPSIS ưu tiên phương án gần lời giải lý tưởng dương nhất và xa lời giải lý tưởng âm nhất.
        Với tiêu chí Gini, đây là tiêu chí chi phí nên giá trị thấp hơn được xem là tốt hơn.
        """)

    with tab2:
        st.subheader("6.3. Dữ liệu 6 vùng kinh tế - xã hội")

        if diagnostics["Trạng thái"] == "Đã đọc dữ liệu từ file vietnam_regions_2024.csv":
            st.success("Đã đọc dữ liệu từ file vietnam_regions_2024.csv")
        else:
            st.warning("Đang dùng dữ liệu mặc định đúng theo đề.")

        diag_df = pd.DataFrame({
            "Nội dung kiểm tra": list(diagnostics.keys()),
            "Kết quả": list(diagnostics.values())
        })
        st.markdown("### Kiểm tra file dữ liệu")
        st.table(diag_df)

        st.markdown("### Bảng dữ liệu đang sử dụng")
        st.dataframe(df, use_container_width=True, height=360)

        st.markdown("### Phân loại tiêu chí")
        criteria_df = pd.DataFrame({
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Cột dữ liệu": CRITERIA,
            "Loại": ["Benefit" if b else "Cost" for b in IS_BENEFIT],
            "Trọng số chuyên gia": EXPERT_WEIGHTS
        })
        st.dataframe(criteria_df, use_container_width=True)

    with tab3:
        st.subheader("Câu 6.4.1 – Cài đặt TOPSIS từ đầu bằng numpy với trọng số chuyên gia")

        st.markdown("### Trọng số chuyên gia")
        expert_weight_df = pd.DataFrame({
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Trọng số": EXPERT_WEIGHTS
        })
        st.dataframe(expert_weight_df, use_container_width=True)

        st.markdown("### Ma trận chuẩn hóa vector R")
        st.dataframe(norm_df, use_container_width=True, height=300)

        st.markdown("### Ma trận chuẩn hóa có trọng số V")
        st.dataframe(weighted_df, use_container_width=True, height=300)

        st.markdown("### Lời giải lý tưởng tốt/xấu")
        st.dataframe(ideal_df, use_container_width=True)

        st.markdown("### Kết quả TOPSIS")
        result_cols = ["rank", "region_name_vi", "S_star", "S_neg", "TOPSIS_score"]
        st.dataframe(expert_result[result_cols], use_container_width=True)

        top_region = expert_result.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top 1", top_region["region_name_vi"], f"{top_region['TOPSIS_score']:.3f}")
        c2.metric("Top 2", expert_result.iloc[1]["region_name_vi"], f"{expert_result.iloc[1]['TOPSIS_score']:.3f}")
        c3.metric("Top 3", expert_result.iloc[2]["region_name_vi"], f"{expert_result.iloc[2]['TOPSIS_score']:.3f}")

        fig = plot_rank_bar(expert_result, "Xếp hạng TOPSIS với trọng số chuyên gia")
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("Câu 6.4.2 – Tính trọng số khách quan bằng Entropy")

        st.write("""
        Phương pháp Entropy cho trọng số cao hơn với những tiêu chí có độ phân tán thông tin lớn hơn giữa các vùng.
        Với Gini là tiêu chí chi phí, giá trị được đảo chiều trước khi tính Entropy.
        """)

        st.markdown("### Bảng trọng số Entropy")
        st.dataframe(entropy_detail_df, use_container_width=True)

        st.markdown("### Kết quả TOPSIS với trọng số Entropy")
        result_cols = ["rank", "region_name_vi", "S_star", "S_neg", "TOPSIS_score"]
        st.dataframe(entropy_result[result_cols], use_container_width=True)

        st.markdown("### So sánh xếp hạng chuyên gia và Entropy")
        st.dataframe(compare_df[[
            "region_name_vi", "Rank chuyên gia", "Score chuyên gia",
            "Rank Entropy", "Score Entropy", "Thay đổi rank Entropy - chuyên gia"
        ]], use_container_width=True)

        largest_change = compare_df.iloc[
            compare_df["Thay đổi rank Entropy - chuyên gia"].abs().argmax()
        ]

        st.info(
            f"Vùng có thay đổi xếp hạng lớn nhất khi dùng Entropy là "
            f"**{largest_change['region_name_vi']}**, với thay đổi rank = "
            f"{largest_change['Thay đổi rank Entropy - chuyên gia']}."
        )

        fig = plot_rank_bar(entropy_result, "Xếp hạng TOPSIS với trọng số Entropy")
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        st.subheader("Câu 6.4.3 – Phân tích độ nhạy w_AI từ 0.10 đến 0.40")

        st.write("""
        Trọng số AI Readiness được thay đổi từ 0.10 đến 0.40. Các trọng số còn lại được điều chỉnh theo tỷ lệ
        để tổng trọng số luôn bằng 1.
        """)

        st.markdown("### Bảng Top-3 theo từng mức w_AI")
        st.dataframe(sensitivity_df, use_container_width=True)

        if top3_stable:
            st.success("Top-3 ổn định khi w_AI thay đổi từ 0.10 đến 0.40.")
        else:
            st.warning("Top-3 có thay đổi khi w_AI thay đổi từ 0.10 đến 0.40.")

        st.markdown("### Heatmap xếp hạng theo w_AI")

        pivot_rank = heatmap_df.pivot(index="Vùng", columns="w_AI", values="Rank")
        mean_rank = pivot_rank.mean(axis=1).sort_values()
        pivot_rank = pivot_rank.loc[mean_rank.index]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(pivot_rank.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot_rank.columns)))
        ax.set_xticklabels([str(c) for c in pivot_rank.columns])
        ax.set_yticks(np.arange(len(pivot_rank.index)))
        ax.set_yticklabels(pivot_rank.index)
        ax.set_xlabel("w_AI")
        ax.set_ylabel("Vùng")
        ax.set_title("Heatmap xếp hạng khi thay đổi trọng số AI Readiness")

        for i in range(pivot_rank.shape[0]):
            for j in range(pivot_rank.shape[1]):
                ax.text(j, i, int(pivot_rank.iloc[i, j]), ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Xếp hạng")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Heatmap điểm TOPSIS theo w_AI")

        pivot_score = heatmap_df.pivot(index="Vùng", columns="w_AI", values="TOPSIS_score")
        pivot_score = pivot_score.loc[mean_rank.index]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        im2 = ax2.imshow(pivot_score.values, aspect="auto")
        ax2.set_xticks(np.arange(len(pivot_score.columns)))
        ax2.set_xticklabels([str(c) for c in pivot_score.columns])
        ax2.set_yticks(np.arange(len(pivot_score.index)))
        ax2.set_yticklabels(pivot_score.index)
        ax2.set_xlabel("w_AI")
        ax2.set_ylabel("Vùng")
        ax2.set_title("Heatmap điểm TOPSIS khi thay đổi w_AI")

        for i in range(pivot_score.shape[0]):
            for j in range(pivot_score.shape[1]):
                ax2.text(j, i, f"{pivot_score.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

        fig2.colorbar(im2, ax=ax2, label="TOPSIS score")
        st.pyplot(fig2)
        plt.close(fig2)

    with tab6:
        st.subheader("Câu 6.4.4 – AHP đơn giản và câu hỏi chính sách 6.5")

        st.markdown("### Trọng số AHP đơn giản")
        st.dataframe(ahp_weight_df, use_container_width=True)

        st.write(f"Chỉ số nhất quán gần đúng: CI = {ci:.6f}; CR = {cr:.6f}")

        st.markdown("### Kết quả TOPSIS dùng trọng số AHP")
        st.dataframe(ahp_result[["rank", "region_name_vi", "TOPSIS_score"]], use_container_width=True)

        st.markdown("### So sánh TOPSIS chuyên gia, Entropy và AHP")
        st.dataframe(compare_df, use_container_width=True)

        expert_top3 = expert_result.head(3)["region_name_vi"].tolist()
        entropy_largest_change = compare_df.iloc[
            compare_df["Thay đổi rank Entropy - chuyên gia"].abs().argmax()
        ]
        top_region = expert_result.iloc[0]["region_name_vi"]

        st.subheader("🤖 Tác nhân phân tích chính sách 6.5")

        with st.container(border=True):
            st.markdown("#### a) Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia? Có nên triển khai trung tâm AI quốc gia đầu tiên không?")
            st.write(
                f"Theo TOPSIS với trọng số chuyên gia, vùng dẫn đầu là **{top_region}**. "
                "Vùng này có điểm gần phương án lý tưởng nhất xét trên các tiêu chí như GRDP/người, FDI, Digital Index, "
                "AI Readiness, lao động đào tạo, R&D, Internet và Gini."
            )
            st.write(
                "Tuy nhiên, đây chưa chắc là vùng duy nhất nên triển khai trung tâm AI đầu tiên. TOPSIS phản ánh mức độ sẵn sàng, "
                "nhưng quyết định chính sách còn cần xét thêm yếu tố địa - chính trị, an ninh dữ liệu, lan tỏa vùng và cân bằng phát triển."
            )

        with st.container(border=True):
            st.markdown("#### b) Khi dùng Entropy, vùng nào thay đổi xếp hạng lớn nhất? Vì sao?")
            st.write(
                f"Vùng có thay đổi xếp hạng lớn nhất khi dùng Entropy là **{entropy_largest_change['region_name_vi']}**. "
                f"Mức thay đổi rank là **{entropy_largest_change['Thay đổi rank Entropy - chuyên gia']}**."
            )
            st.write(
                "Nguyên nhân là Entropy không dùng đánh giá chủ quan mà dựa vào độ phân tán dữ liệu. "
                "Tiêu chí nào có khác biệt lớn giữa các vùng sẽ được trọng số cao hơn. Do đó, những vùng mạnh hoặc yếu rõ rệt "
                "ở các tiêu chí phân tán mạnh có thể thay đổi thứ hạng so với bộ trọng số chuyên gia."
            )

        with st.container(border=True):
            st.markdown("#### c) AI Readiness và Internet penetration có thể tương quan cao. Điều này ảnh hưởng thế nào? Đề xuất xử lý.")
            st.write(
                "Nếu AI Readiness và Internet penetration tương quan cao, TOPSIS có thể vô tình đếm hai lần cùng một khía cạnh năng lực số. "
                "Khi đó, các vùng có hạ tầng số tốt sẽ được cộng lợi thế quá mức, làm kết quả nghiêng về các vùng đã phát triển."
            )
            st.write(
                "Có thể xử lý bằng cách kiểm tra ma trận tương quan, loại bớt tiêu chí trùng lặp, gộp các tiêu chí thành một chỉ số tổng hợp, "
                "hoặc sử dụng PCA/factor analysis để giảm chiều dữ liệu trước khi áp dụng TOPSIS."
            )

        with st.container(border=True):
            st.markdown("#### d) Chọn 3 vùng nào để xây dựng 3 trung tâm AI lớn? Có cần điều chỉnh địa - chính trị không?")
            st.write(
                f"Dựa trên TOPSIS trọng số chuyên gia, ba vùng ưu tiên là: "
                f"**{expert_top3[0]}**, **{expert_top3[1]}** và **{expert_top3[2]}**."
            )
            st.write(
                "Tuy nhiên, để triển khai 3 trung tâm AI lớn theo tinh thần Quyết định 127/QĐ-TTg, không nên chỉ chọn máy móc theo điểm TOPSIS. "
                "Cần điều chỉnh thêm tiêu chí địa - chính trị, bảo đảm phân bố Bắc - Trung - Nam, an ninh dữ liệu, khả năng liên kết đại học - doanh nghiệp "
                "và mục tiêu thu hẹp khoảng cách số giữa các vùng."
            )


if __name__ == "__main__":
    run()
