import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

try:
    from pulp import (
        LpProblem, LpMaximize, LpVariable, lpSum,
        PULP_CBC_CMD, LpStatus, value
    )
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


# =========================================================
# BÀI 5 – MIP LỰA CHỌN 15 DỰ ÁN CHUYỂN ĐỔI SỐ
# BẢN FIX THEO CODE GỢI Ý GIẢNG VIÊN
# Kết quả chuẩn:
# selected = P2, P4, P6, P7, P8, P9, P12, P14, P15
# Z* = 115,400 tỷ
# C = 59,600; C1 = 39,800; số dự án = 9
# =========================================================


P = list(range(1, 16))

C = {
    1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200,
    6: 5800, 7: 6500, 8: 15000, 9: 2500, 10: 7200,
    11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500
}

C1 = {
    1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500,
    6: 4000, 7: 4500, 8: 9000, 9: 1800, 10: 5000,
    11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200
}

B = {
    1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800,
    6: 11400, 7: 12200, 8: 28500, 9: 5800, 10: 13800,
    11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800
}

PROJECT_NAMES = {
    1: "P1 - Hạ tầng số quốc gia",
    2: "P2 - Trung tâm dữ liệu vùng",
    3: "P3 - Siêu dự án AI/bán dẫn",
    4: "P4 - Dịch vụ công trực tuyến",
    5: "P5 - Thanh toán số",
    6: "P6 - Nông nghiệp số",
    7: "P7 - SME chuyển đổi số",
    8: "P8 - AI trong y tế",
    9: "P9 - Giáo dục số",
    10: "P10 - Du lịch số",
    11: "P11 - Logistics số",
    12: "P12 - An ninh mạng",
    13: "P13 - Cloud chủ quyền",
    14: "P14 - Đào tạo kỹ năng số",
    15: "P15 - Dữ liệu mở"
}


def data_table():
    df = pd.DataFrame({
        "Dự án": [f"P{i}" for i in P],
        "Tên dự án": [PROJECT_NAMES[i] for i in P],
        "C - Tổng chi phí": [C[i] for i in P],
        "C1 - Chi phí năm 1-2": [C1[i] for i in P],
        "B - Lợi ích": [B[i] for i in P],
        "B/C": [B[i] / C[i] for i in P]
    })
    return df


def check_constraints(sel, budget=80000, budget12=40000, both_p1_p2=False):
    sel = set(sel)

    checks = {
        "C1 Tổng chi phí ≤ ngân sách": sum(C[i] for i in sel) <= budget,
        "C2 Chi phí năm 1-2 ≤ 40,000": sum(C1[i] for i in sel) <= budget12,
        "C3 Không chọn đồng thời P1 và P2": not ({1, 2}.issubset(sel)),
        "C4 Nếu chọn P8 thì phải chọn P12": (8 not in sel) or (12 in sel),
        "C5 Nếu chọn P13 thì phải chọn P12": (13 not in sel) or (12 in sel),
        "C6 Chọn ít nhất một trong P4, P5": (4 in sel) or (5 in sel),
        "C7 Bắt buộc chọn P14": 14 in sel,
        "C8 Số dự án từ 7 đến 11": 7 <= len(sel) <= 11,
    }

    if both_p1_p2:
        checks["Mở rộng: bắt buộc chọn cả P1 và P2"] = ({1, 2}.issubset(sel))

    return checks


def brute_force_solve(budget=80000, budget12=40000, force_p1_p2=False, expected=False):
    """
    Fallback chắc chắn đúng, không phụ thuộc PuLP.
    expected=True: tối đa hóa expected benefit với xác suất rủi ro.
    """
    p_risk = {}
    for i in P:
        if i in [1, 2, 8, 12, 13]:
            p_risk[i] = 0.85
        elif i in [4, 5, 9, 14, 15]:
            p_risk[i] = 0.75
        elif i in [3]:
            p_risk[i] = 0.65
        else:
            p_risk[i] = 0.80

    best_z = -1
    best_sel = []

    for r in range(7, 12):
        for comb in combinations(P, r):
            sel = set(comb)

            if sum(C[i] for i in sel) > budget:
                continue
            if sum(C1[i] for i in sel) > budget12:
                continue
            if 1 in sel and 2 in sel and not force_p1_p2:
                continue
            if force_p1_p2 and not ({1, 2}.issubset(sel)):
                continue
            if 8 in sel and 12 not in sel:
                continue
            if 13 in sel and 12 not in sel:
                continue
            if 4 not in sel and 5 not in sel:
                continue
            if 14 not in sel:
                continue

            if expected:
                z = sum(B[i] * p_risk[i] for i in sel)
            else:
                z = sum(B[i] for i in sel)

            if z > best_z:
                best_z = z
                best_sel = sorted(sel)

    return best_sel, best_z


def pulp_solve(budget=80000, budget12=40000, force_p1_p2=False, expected=False):
    """
    Giải đúng theo mã giảng viên.
    Nếu PuLP/CBC có vấn đề, tự fallback sang brute force.
    """
    if not PULP_AVAILABLE:
        return brute_force_solve(budget, budget12, force_p1_p2, expected), "Brute force fallback"

    p_risk = {}
    for i in P:
        if i in [1, 2, 8, 12, 13]:
            p_risk[i] = 0.85
        elif i in [4, 5, 9, 14, 15]:
            p_risk[i] = 0.75
        elif i in [3]:
            p_risk[i] = 0.65
        else:
            p_risk[i] = 0.80

    try:
        m = LpProblem("VN_Project_Selection", LpMaximize)
        y = LpVariable.dicts("y", P, cat="Binary")

        if expected:
            m += lpSum(B[i] * p_risk[i] * y[i] for i in P)
        else:
            m += lpSum(B[i] * y[i] for i in P)

        m += lpSum(C[i] * y[i] for i in P) <= budget
        m += lpSum(C1[i] * y[i] for i in P) <= budget12
        m += y[1] + y[2] <= 1

        if force_p1_p2:
            # Câu 5.4.3 kiểm tra trường hợp bắt buộc cả P1 và P2.
            # Điều kiện này mâu thuẫn với y1+y2<=1, nên bài toán sẽ vô nghiệm.
            m += y[1] >= 1
            m += y[2] >= 1

        m += y[8] <= y[12]
        m += y[13] <= y[12]
        m += y[4] + y[5] >= 1
        m += y[14] >= 1
        m += lpSum(y[i] for i in P) >= 7
        m += lpSum(y[i] for i in P) <= 11

        m.solve(PULP_CBC_CMD(msg=False))
        status = LpStatus[m.status]

        if status != "Optimal":
            return ([], None), f"PuLP status: {status}"

        selected = [i for i in P if y[i].value() is not None and y[i].value() > 0.5]
        obj = value(m.objective)

        # Nếu vì lỗi môi trường objective trả None/0 bất thường thì dùng brute force.
        if obj is None or (obj == 0 and len(selected) == 0):
            return brute_force_solve(budget, budget12, force_p1_p2, expected), "Brute force fallback do objective PuLP bất thường"

        return (selected, obj), "PuLP CBC Optimal"

    except Exception as e:
        return brute_force_solve(budget, budget12, force_p1_p2, expected), f"Brute force fallback do lỗi PuLP: {e}"


def selected_table(sel):
    df = data_table()
    df["Chọn"] = df["Dự án"].apply(lambda x: int(x.replace("P", "")) in sel)
    return df[df["Chọn"]].copy()


def summary_metrics(sel, z):
    return {
        "Số dự án": len(sel),
        "Tổng lợi ích Z*": z,
        "Tổng chi phí": sum(C[i] for i in sel),
        "Chi phí năm 1-2": sum(C1[i] for i in sel),
        "B/C trung bình": np.mean([B[i] / C[i] for i in sel]) if sel else 0
    }


def plot_selected(sel):
    df = selected_table(sel)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df["Dự án"], df["B - Lợi ích"])
    ax.set_title("Lợi ích các dự án được chọn")
    ax.set_ylabel("Tỷ VND")
    ax.grid(axis="y", alpha=0.35)
    for i, v in enumerate(df["B - Lợi ích"]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    return fig


def run():
    st.title("📌 Bài 5 – MIP lựa chọn 15 dự án chuyển đổi số")

    st.write("""
    Bài toán lựa chọn dự án được mô hình hóa dưới dạng quy hoạch nguyên nhị phân.
    Biến quyết định là \(y_i \\in \\{0,1\\}\), trong đó \(y_i=1\) nếu dự án \(P_i\) được chọn.
    """)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "5.3 Mô hình",
        "Dữ liệu",
        "5.4.1 MIP gốc",
        "5.4.2 Ngân sách 100,000",
        "5.4.3 Bắt buộc P1 & P2",
        "5.4.4 Rủi ro & 5.5"
    ])

    with tab1:
        st.subheader("5.3. Mô hình toán học")
        st.latex(r"y_i \in \{0,1\},\quad i=1,\ldots,15")
        st.latex(r"\max Z=\sum_i B_i y_i")

        st.markdown("**Ràng buộc:**")
        st.latex(r"\sum_i C_i y_i \leq 80{,}000")
        st.latex(r"\sum_i C_{1i} y_i \leq 40{,}000")
        st.latex(r"y_1+y_2 \leq 1")
        st.latex(r"y_8 \leq y_{12},\quad y_{13} \leq y_{12}")
        st.latex(r"y_4+y_5 \geq 1,\quad y_{14}\geq 1")
        st.latex(r"7 \leq \sum_i y_i \leq 11")

        if PULP_AVAILABLE:
            st.success("PuLP đã sẵn sàng. Mô hình được giải bằng CBC solver.")
        else:
            st.warning("Chưa có PuLP. App dùng brute force fallback nên kết quả vẫn đúng.")

    with tab2:
        st.subheader("Bảng dữ liệu 15 dự án")
        st.dataframe(data_table(), use_container_width=True, height=520)

    with tab3:
        st.subheader("5.4.1. Kết quả mô hình MIP gốc")

        (sel, z), status = pulp_solve(budget=80000, budget12=40000)
        st.info(f"Trạng thái: {status}")

        if sel:
            metrics = summary_metrics(sel, z)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Z*", f"{metrics['Tổng lợi ích Z*']:,.0f} tỷ")
            c2.metric("Số dự án", f"{metrics['Số dự án']}")
            c3.metric("Tổng chi phí", f"{metrics['Tổng chi phí']:,.0f}")
            c4.metric("Chi phí năm 1-2", f"{metrics['Chi phí năm 1-2']:,.0f}")

            st.success("Dự án được chọn: " + ", ".join([f"P{i}" for i in sel]))
            st.dataframe(selected_table(sel), use_container_width=True)

            fig = plot_selected(sel)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Kiểm tra ràng buộc")
            checks = check_constraints(sel)
            st.dataframe(pd.DataFrame({
                "Ràng buộc": list(checks.keys()),
                "Đạt": list(checks.values())
            }), use_container_width=True)
        else:
            st.error("Không tìm được nghiệm tối ưu.")

    with tab4:
        st.subheader("5.4.2. Khi ngân sách tăng lên 100,000")

        (sel_80, z_80), _ = pulp_solve(budget=80000, budget12=40000)
        (sel_100, z_100), status_100 = pulp_solve(budget=100000, budget12=40000)

        st.info(f"Trạng thái ngân sách 100,000: {status_100}")

        compare = pd.DataFrame({
            "Trường hợp": ["Ngân sách 80,000", "Ngân sách 100,000"],
            "Dự án chọn": [
                ", ".join([f"P{i}" for i in sel_80]),
                ", ".join([f"P{i}" for i in sel_100])
            ],
            "Số dự án": [len(sel_80), len(sel_100)],
            "Z*": [z_80, z_100],
            "Tổng chi phí": [sum(C[i] for i in sel_80), sum(C[i] for i in sel_100)],
            "Chi phí năm 1-2": [sum(C1[i] for i in sel_80), sum(C1[i] for i in sel_100)]
        })
        st.dataframe(compare, use_container_width=True)

        st.write(
            "Khi tổng ngân sách tăng, nghiệm có thể chọn thêm hoặc thay thế dự án có lợi ích cao hơn. "
            "Tuy nhiên, ràng buộc chi phí năm 1-2 vẫn giới hạn khả năng mở rộng danh mục."
        )

    with tab5:
        st.subheader("5.4.3. Kiểm tra trường hợp bắt buộc chọn cả P1 và P2")

        st.warning(
            "Theo mô hình gốc có ràng buộc y1 + y2 ≤ 1, nên nếu bắt buộc chọn cả P1 và P2 thì bài toán mâu thuẫn và vô nghiệm."
        )

        (sel_force, z_force), status_force = pulp_solve(budget=80000, budget12=40000, force_p1_p2=True)
        st.info(f"Trạng thái: {status_force}")

        if sel_force:
            st.dataframe(selected_table(sel_force), use_container_width=True)
            st.metric("Z*", f"{z_force:,.0f}")
        else:
            st.error("Không có nghiệm khả thi do ràng buộc y1 + y2 ≤ 1 mâu thuẫn với yêu cầu chọn cả P1 và P2.")

    with tab6:
        st.subheader("5.4.4. Mở rộng xét rủi ro xác suất thành công")

        st.write("""
        Với mở rộng rủi ro, hàm mục tiêu được điều chỉnh thành lợi ích kỳ vọng:
        """)
        st.latex(r"\max E(Z)=\sum_i p_i B_i y_i")

        prob_df = pd.DataFrame({
            "Dự án": [f"P{i}" for i in P],
            "B": [B[i] for i in P],
            "p_i": [
                0.85 if i in [1, 2, 8, 12, 13]
                else 0.75 if i in [4, 5, 9, 14, 15]
                else 0.65 if i == 3
                else 0.80
                for i in P
            ]
        })
        prob_df["Expected Benefit"] = prob_df["B"] * prob_df["p_i"]
        st.dataframe(prob_df, use_container_width=True)

        (sel_risk, z_risk), status_risk = pulp_solve(budget=80000, budget12=40000, expected=True)
        st.info(f"Trạng thái: {status_risk}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Z*", f"{z_risk:,.0f}")
        c2.metric("Số dự án", f"{len(sel_risk)}")
        c3.metric("Tổng chi phí", f"{sum(C[i] for i in sel_risk):,.0f}")

        st.success("Dự án được chọn khi xét rủi ro: " + ", ".join([f"P{i}" for i in sel_risk]))
        st.dataframe(selected_table(sel_risk), use_container_width=True)

        st.subheader("5.5. Nhận xét chính sách")
        st.write("""
        Kết quả cho thấy mô hình MIP giúp lựa chọn danh mục dự án thỏa mãn đồng thời nhiều ràng buộc về ngân sách,
        tiến độ triển khai, quan hệ phụ thuộc và yêu cầu tối thiểu về số lượng dự án. Nghiệm tối ưu gốc chọn 9 dự án,
        đạt tổng lợi ích 115,400 tỷ VND trong khi vẫn không vượt ngân sách tổng và ngân sách hai năm đầu.
        Khi bổ sung yếu tố rủi ro, danh mục tối ưu có thể thay đổi do mô hình ưu tiên các dự án có lợi ích kỳ vọng cao hơn,
        thay vì chỉ xét lợi ích danh nghĩa.
        """)


if __name__ == "__main__":
    run()
