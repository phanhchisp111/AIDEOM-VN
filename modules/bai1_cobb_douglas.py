import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run():
    st.title("🧮 Bài 1 – Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")

    st.write("""
    Bài 1 sử dụng hàm sản xuất Cobb-Douglas mở rộng để đánh giá vai trò của vốn vật chất,
    lao động, chuyển đổi số, năng lực AI và vốn nhân lực số đối với tăng trưởng GDP Việt Nam
    giai đoạn 2020–2025. Mô hình cho phép tính TFP, GDP dự báo, MAPE, phân rã tăng trưởng
    và dự báo GDP đến năm 2030.
    """)

    # =========================
    # 1. DỮ LIỆU ĐẦU VÀO
    # =========================
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025])

    Y = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    input_df = pd.DataFrame({
        "Năm": years,
        "Y - GDP thực tế": Y,
        "K - Vốn tích lũy": K,
        "L - Lao động": L,
        "D - Kinh tế số/GDP (%)": D,
        "AI - DN công nghệ số": AI,
        "H - Lao động qua đào tạo (%)": H
    })

    # =========================
    # 2. GIAO DIỆN TAB
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Tổng quan",
        "⚙️ Tham số & dữ liệu",
        "📊 Kết quả mô hình",
        "📈 Biểu đồ",
        "🤖 Tác nhân phân tích"
    ])

    with tab1:
        st.subheader("1.1. Bối cảnh Việt Nam")
        st.write("""
        Theo đề bài, GDP Việt Nam năm 2024 đạt 11.511,9 nghìn tỷ VND và tiếp tục tăng trong năm 2025.
        Đồng thời, kinh tế số, năng lực AI và vốn nhân lực số ngày càng trở thành các nhân tố quan trọng
        trong mô hình tăng trưởng mới. Vì vậy, bài này mở rộng hàm sản xuất Cobb-Douglas truyền thống
        bằng cách bổ sung ba yếu tố mới: số hóa D, năng lực AI và vốn nhân lực số H.
        """)

        st.subheader("1.2. Mô hình toán học")
        st.latex(r"Y_t = A_t \cdot K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta")
        st.latex(r"\alpha + \beta + \gamma + \delta + \theta = 1")

        st.write("""
        Trong đó:
        - **Y**: sản lượng/GDP của nền kinh tế.
        - **A**: năng suất nhân tố tổng hợp, hay TFP.
        - **K**: vốn vật chất.
        - **L**: lao động.
        - **D**: mức độ số hóa, đo bằng tỷ trọng kinh tế số/GDP.
        - **AI**: năng lực trí tuệ nhân tạo.
        - **H**: vốn nhân lực số.
        """)

        st.subheader("Phân rã tăng trưởng")
        st.latex(
            r"\Delta \ln Y_t = \Delta \ln A_t + \alpha \Delta \ln K_t + \beta \Delta \ln L_t + \gamma \Delta \ln D_t + \delta \Delta \ln AI_t + \theta \Delta \ln H_t"
        )

    with tab2:
        st.subheader("Dữ liệu Việt Nam 2020–2025")
        st.dataframe(input_df, use_container_width=True)

        st.subheader("Điều chỉnh tham số đàn hồi")

        col1, col2 = st.columns(2)

        with col1:
            alpha = st.slider("α - Độ co giãn theo vốn vật chất K", 0.00, 0.80, 0.33, 0.01)
            beta = st.slider("β - Độ co giãn theo lao động L", 0.00, 0.80, 0.42, 0.01)

        with col2:
            gamma = st.slider("γ - Độ co giãn theo số hóa D", 0.00, 0.50, 0.10, 0.01)
            delta = st.slider("δ - Độ co giãn theo AI", 0.00, 0.50, 0.08, 0.01)

        theta = round(1.0 - (alpha + beta + gamma + delta), 4)

        st.metric("θ - Độ co giãn theo nhân lực số H", f"{theta:.4f}")

        if theta < 0:
            st.error("Tổng α + β + γ + δ đang lớn hơn 1 nên θ bị âm. Hãy giảm một trong các tham số.")
            return

        st.info(f"Tổng hệ số hiện tại: α + β + γ + δ + θ = {alpha + beta + gamma + delta + theta:.4f}")

    # =========================
    # 3. TÍNH TOÁN CHÍNH
    # =========================
    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_mean = A.mean()

    Y_hat = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    result_df = pd.DataFrame({
        "Năm": years,
        "GDP thực tế": Y,
        "TFP A_t": A,
        "GDP dự báo": Y_hat,
        "Sai số tuyệt đối (%)": np.abs((Y - Y_hat) / Y) * 100
    })

    # =========================
    # 4. PHÂN RÃ TĂNG TRƯỞNG
    # =========================
    dln_Y = np.log(Y[-1]) - np.log(Y[0])
    dln_K = np.log(K[-1]) - np.log(K[0])
    dln_L = np.log(L[-1]) - np.log(L[0])
    dln_D = np.log(D[-1]) - np.log(D[0])
    dln_AI = np.log(AI[-1]) - np.log(AI[0])
    dln_H = np.log(H[-1]) - np.log(H[0])
    dln_A = np.log(A[-1]) - np.log(A[0])

    growth_df = pd.DataFrame({
        "Yếu tố": ["TFP", "Vốn K", "Lao động L", "Số hóa D", "Năng lực AI", "Nhân lực số H"],
        "Đóng góp log-growth": [
            dln_A,
            alpha * dln_K,
            beta * dln_L,
            gamma * dln_D,
            delta * dln_AI,
            theta * dln_H
        ]
    })

    growth_df["Tỷ trọng đóng góp (%)"] = growth_df["Đóng góp log-growth"] / dln_Y * 100

    # =========================
    # 5. DỰ BÁO GDP 2030 THEO ĐỀ
    # =========================
    K_2030 = K[-1] * (1.06 ** 5)
    L_2030 = L[-1] * (1.06 ** 5)
    D_2030 = 30
    AI_2030 = 100
    H_2030 = 35
    A_2030 = A[-1] * (1.012 ** 5)

    GDP_2030 = A_2030 * (
        K_2030**alpha *
        L_2030**beta *
        D_2030**gamma *
        AI_2030**delta *
        H_2030**theta
    )

    scenario_2030_df = pd.DataFrame({
        "Chỉ tiêu": [
            "K năm 2030",
            "L năm 2030",
            "D năm 2030",
            "AI năm 2030",
            "H năm 2030",
            "TFP A năm 2030",
            "GDP dự báo 2030"
        ],
        "Giá trị": [
            K_2030,
            L_2030,
            D_2030,
            AI_2030,
            H_2030,
            A_2030,
            GDP_2030
        ]
    })

    future_years = np.arange(2025, 2031)

    K_future = K[-1] * (1.06 ** np.arange(0, 6))
    L_future = L[-1] * (1.06 ** np.arange(0, 6))
    A_future = A[-1] * (1.012 ** np.arange(0, 6))
    D_future = np.linspace(D[-1], 30, 6)
    AI_future = np.linspace(AI[-1], 100, 6)
    H_future = np.linspace(H[-1], 35, 6)

    GDP_future = A_future * (
        K_future**alpha *
        L_future**beta *
        D_future**gamma *
        AI_future**delta *
        H_future**theta
    )

    future_df = pd.DataFrame({
        "Năm": future_years,
        "K": K_future,
        "L": L_future,
        "D": D_future,
        "AI": AI_future,
        "H": H_future,
        "TFP A": A_future,
        "GDP dự báo": GDP_future
    })
    # =========================
    # 6. MỞ RỘNG: BA KỊCH BẢN DỰ BÁO GDP 2030
    # =========================
    scenario_rows = []

    scenarios = {
        "Chuyển đổi số chậm": {
            "D_2030": 25,
            "AI_2030": 85,
            "H_2030": 32,
            "tfp_growth": 0.008,
            "description": "Kinh tế số và năng lực AI tăng chậm, TFP cải thiện thấp."
        },
        "Cơ sở theo đề": {
            "D_2030": 30,
            "AI_2030": 100,
            "H_2030": 35,
            "tfp_growth": 0.012,
            "description": "Kịch bản đúng theo yêu cầu đề bài."
        },
        "Chuyển đổi số nhanh": {
            "D_2030": 35,
            "AI_2030": 120,
            "H_2030": 40,
            "tfp_growth": 0.018,
            "description": "Kinh tế số, AI và nhân lực số tăng nhanh hơn kỳ vọng."
        }
    }

    for scenario_name, params in scenarios.items():
        K_s = K[-1] * (1.06 ** 5)
        L_s = L[-1] * (1.06 ** 5)
        D_s = params["D_2030"]
        AI_s = params["AI_2030"]
        H_s = params["H_2030"]
        A_s = A[-1] * ((1 + params["tfp_growth"]) ** 5)

        GDP_s = A_s * (
            K_s**alpha *
            L_s**beta *
            D_s**gamma *
            AI_s**delta *
            H_s**theta
        )

        scenario_rows.append({
            "Kịch bản": scenario_name,
            "D 2030 (%)": D_s,
            "AI 2030": AI_s,
            "H 2030 (%)": H_s,
            "Tăng trưởng TFP/năm": params["tfp_growth"] * 100,
            "GDP dự báo 2030": GDP_s,
            "Diễn giải": params["description"]
        })

    scenario_compare_df = pd.DataFrame(scenario_rows)
    # =========================
    # TAB 3: KẾT QUẢ
    # =========================
    with tab3:
        st.subheader("Kết quả chính của mô hình")

        c1, c2, c3 = st.columns(3)
        c1.metric("TFP trung bình", f"{A_mean:.3f}")
        c2.metric("MAPE", f"{mape:.2f}%")
        c3.metric("GDP dự báo 2030", f"{GDP_2030:,.2f}")

        st.subheader("Câu 1.4.1 và 1.4.2: TFP, GDP dự báo và MAPE")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("Câu 1.4.3: Phân rã tăng trưởng 2020–2025")
        st.dataframe(growth_df, use_container_width=True)

        st.subheader("Câu 1.4.4: Kịch bản dự báo GDP năm 2030")
        st.dataframe(scenario_2030_df, use_container_width=True)

        st.subheader("Mô phỏng GDP giai đoạn 2025–2030")
        st.dataframe(future_df, use_container_width=True)
        st.subheader("Mở rộng: So sánh GDP 2030 theo 3 kịch bản")
        st.dataframe(scenario_compare_df, use_container_width=True)

    # =========================
    # TAB 4: BIỂU ĐỒ
    # =========================
    with tab4:
        st.subheader("Biểu đồ TFP theo năm")

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(years, A, marker="o", linewidth=2)
        ax1.set_title("Xu hướng TFP A_t giai đoạn 2020–2025")
        ax1.set_xlabel("Năm")
        ax1.set_ylabel("TFP A_t")
        ax1.grid(True, alpha=0.5)
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("GDP thực tế và GDP dự báo")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(years, Y, marker="o", linewidth=2, label="GDP thực tế")
        ax2.plot(years, Y_hat, marker="s", linewidth=2, label="GDP dự báo")
        ax2.set_title("So sánh GDP thực tế và GDP dự báo")
        ax2.set_xlabel("Năm")
        ax2.set_ylabel("GDP, nghìn tỷ VND")
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Biểu đồ phân rã tăng trưởng")

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        ax3.bar(growth_df["Yếu tố"], growth_df["Tỷ trọng đóng góp (%)"])
        ax3.set_title("Tỷ trọng đóng góp vào tăng trưởng GDP 2020–2025")
        ax3.set_xlabel("Yếu tố")
        ax3.set_ylabel("Tỷ trọng đóng góp (%)")
        ax3.tick_params(axis="x", rotation=30)
        ax3.grid(axis="y", alpha=0.5)
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Biểu đồ dự báo GDP đến năm 2030")

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(future_df["Năm"], future_df["GDP dự báo"], marker="o", linewidth=2)
        ax4.set_title("Mô phỏng GDP Việt Nam đến năm 2030")
        ax4.set_xlabel("Năm")
        ax4.set_ylabel("GDP dự báo, nghìn tỷ VND")
        ax4.grid(True, alpha=0.5)
        st.pyplot(fig4)
        plt.close(fig4)
        st.subheader("Mở rộng: GDP 2030 theo các kịch bản chuyển đổi số")

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.bar(
            scenario_compare_df["Kịch bản"],
            scenario_compare_df["GDP dự báo 2030"]
        )
        ax5.set_title("So sánh GDP dự báo năm 2030 theo 3 kịch bản")
        ax5.set_xlabel("Kịch bản")
        ax5.set_ylabel("GDP dự báo 2030, nghìn tỷ VND")
        ax5.tick_params(axis="x", rotation=15)
        ax5.grid(axis="y", alpha=0.5)
        st.pyplot(fig5)
        plt.close(fig5)

    # =========================
    # TAB 5: AI AGENT
    # =========================
    with tab5:
        st.subheader("🤖 Tác nhân phân tích kết quả")

        A_start = A[0]
        A_end = A[-1]

        if A_end > A_start:
            tfp_trend = "tăng"
            tfp_meaning = (
                "Điều này cho thấy chất lượng tăng trưởng của Việt Nam có xu hướng được cải thiện. "
                "Tăng trưởng GDP không chỉ đến từ mở rộng vốn và lao động, mà còn phản ánh hiệu quả sử dụng nguồn lực, "
                "tiến bộ công nghệ, chuyển đổi số và năng lực quản trị tốt hơn."
            )
        elif A_end < A_start:
            tfp_trend = "giảm"
            tfp_meaning = (
                "Điều này cho thấy chất lượng tăng trưởng có dấu hiệu suy giảm. "
                "Tăng trưởng có thể vẫn phụ thuộc nhiều vào vốn, lao động hoặc các đầu vào truyền thống."
            )
        else:
            tfp_trend = "gần như không đổi"
            tfp_meaning = "Điều này cho thấy chất lượng tăng trưởng chưa có sự thay đổi rõ rệt."

        new_factors = growth_df[
            growth_df["Yếu tố"].isin(["Số hóa D", "Năng lực AI", "Nhân lực số H"])
        ].copy()

        top_new_factor = new_factors.sort_values(
            "Tỷ trọng đóng góp (%)",
            ascending=False
        ).iloc[0]

        top_factor_name = top_new_factor["Yếu tố"]
        top_factor_value = top_new_factor["Tỷ trọng đóng góp (%)"]

        if top_factor_name == "Số hóa D":
            reason = (
                "Số hóa D đóng góp lớn nhất vì tỷ trọng kinh tế số/GDP tăng nhanh trong giai đoạn 2020–2025. "
                "Điều này phản ánh quá trình chuyển đổi số lan rộng trong thương mại, dịch vụ, thanh toán số, "
                "chính phủ số và hoạt động của doanh nghiệp."
            )
        elif top_factor_name == "Năng lực AI":
            reason = (
                "Năng lực AI đóng góp lớn nhất vì số lượng doanh nghiệp công nghệ số và khả năng ứng dụng AI tăng nhanh, "
                "góp phần cải thiện năng suất và đổi mới mô hình sản xuất."
            )
        else:
            reason = (
                "Nhân lực số H đóng góp lớn nhất vì chất lượng lao động và tỷ lệ lao động qua đào tạo được cải thiện, "
                "giúp nền kinh tế hấp thụ công nghệ số và AI hiệu quả hơn."
            )

        if mape < 5:
            mape_comment = "MAPE ở mức thấp, cho thấy mô hình có độ khớp khá tốt với dữ liệu thực tế."
        elif mape < 10:
            mape_comment = "MAPE ở mức trung bình, mô hình có thể dùng để tham khảo nhưng vẫn cần hiệu chỉnh thêm."
        else:
            mape_comment = "MAPE khá cao, mô hình cần được xem xét lại về tham số hoặc dữ liệu đầu vào."

        feasibility = (
            "Dựa trên mô hình mô phỏng, mục tiêu Việt Nam đạt 30% kinh tế số/GDP vào năm 2030 "
            "có thể xem là khả thi về mặt định lượng nếu Việt Nam duy trì được tốc độ chuyển đổi số, "
            "đồng thời cải thiện năng lực AI, nhân lực số, vốn vật chất và TFP."
        )

        constraints = (
            "Tuy nhiên, mục tiêu này cần đi kèm các ràng buộc chính sách quan trọng như: "
            "đầu tư vào hạ tầng số, dữ liệu, điện toán đám mây và Internet tốc độ cao; "
            "nâng cao chất lượng nhân lực số và kỹ năng AI; "
            "tăng khả năng hấp thụ công nghệ của doanh nghiệp, đặc biệt là doanh nghiệp nhỏ và vừa; "
            "bảo đảm tăng trưởng TFP; đồng thời hoàn thiện khung pháp lý về dữ liệu, "
            "an ninh mạng và quản trị kinh tế số."
        )

        st.markdown("### 🧠 AI Agent – Trả lời câu 1.5")

        with st.container(border=True):
            st.markdown("#### a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020–2025? Điều đó nói lên gì về chất lượng tăng trưởng?")
            st.write(
                f"TFP của Việt Nam có xu hướng **{tfp_trend}** trong giai đoạn 2020–2025. "
                f"Cụ thể, TFP thay đổi từ khoảng **{A_start:.2f}** năm 2020 lên khoảng **{A_end:.2f}** năm 2025."
            )
            st.write(tfp_meaning)
            st.write(mape_comment)

        with st.container(border=True):
            st.markdown("#### b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất cho tăng trưởng giai đoạn vừa qua? Vì sao?")
            st.write(
                f"Trong ba yếu tố mới gồm **số hóa D**, **năng lực AI** và **vốn nhân lực số H**, "
                f"yếu tố đóng góp lớn nhất là **{top_factor_name}**, với tỷ trọng đóng góp khoảng "
                f"**{top_factor_value:.2f}%** trong tổng tăng trưởng log GDP giai đoạn 2020–2025."
            )
            st.write(reason)

        with st.container(border=True):
            st.markdown("#### c) Mục tiêu Việt Nam đạt 30% kinh tế số/GDP vào năm 2030 có khả thi không nếu dựa trên mô hình này? Cần ràng buộc gì?")
        st.markdown("### 🚀 Phân tích mở rộng theo kịch bản")

        best_scenario = scenario_compare_df.sort_values(
            "GDP dự báo 2030",
            ascending=False
        ).iloc[0]

        worst_scenario = scenario_compare_df.sort_values(
            "GDP dự báo 2030",
            ascending=True
        ).iloc[0]

        with st.container(border=True):
            st.markdown("#### d) Khi mở rộng mô hình theo 3 kịch bản, kịch bản nào tạo GDP 2030 cao nhất?")

            st.write(
                f"Kết quả mô phỏng cho thấy kịch bản tạo GDP năm 2030 cao nhất là "
                f"**{best_scenario['Kịch bản']}**, với GDP dự báo khoảng "
                f"**{best_scenario['GDP dự báo 2030']:,.2f} nghìn tỷ VND**."
            )

            st.write(
                f"Ngược lại, kịch bản thấp nhất là **{worst_scenario['Kịch bản']}**, "
                f"với GDP dự báo khoảng **{worst_scenario['GDP dự báo 2030']:,.2f} nghìn tỷ VND**."
            )

            st.write(
                "Điều này cho thấy tốc độ phát triển kinh tế số, năng lực AI, nhân lực số và TFP "
                "có ảnh hưởng rõ rệt đến triển vọng tăng trưởng dài hạn. Nếu Việt Nam thúc đẩy nhanh "
                "chuyển đổi số và cải thiện năng suất, GDP năm 2030 có thể cao hơn đáng kể so với kịch bản cơ sở."
            )

        with st.container(border=True):
            st.markdown("#### e) Hàm ý chính sách từ phần mở rộng")

            st.write(
                "Từ phần mở rộng, có thể thấy chính sách tăng trưởng không nên chỉ tập trung vào mở rộng vốn và lao động, "
                "mà cần ưu tiên các yếu tố nâng cao chất lượng tăng trưởng như TFP, kinh tế số, AI và nhân lực số."
            )

            st.write(
                "Trong đó, Chính phủ cần tập trung vào bốn nhóm chính sách: "
                "**phát triển hạ tầng số**, **thúc đẩy doanh nghiệp ứng dụng AI**, "
                "**đào tạo nhân lực số chất lượng cao**, và **cải thiện năng suất nhân tố tổng hợp** thông qua đổi mới sáng tạo."
            )
            st.write(feasibility)
            st.write(
                f"Theo kịch bản đề bài, với **D = 30%**, **AI = 100 nghìn doanh nghiệp số**, "
                f"**H = 35%**, **K và L tăng 6%/năm**, và **TFP tăng 1,2%/năm**, "
                f"GDP Việt Nam năm 2030 được dự báo đạt khoảng **{GDP_2030:,.2f} nghìn tỷ VND**."
            )
            st.write(constraints)
            st.write(
                "**Kết luận:** Mục tiêu 30% kinh tế số/GDP vào năm 2030 là có cơ sở, "
                "nhưng chỉ khả thi nếu Việt Nam đầu tư đồng bộ vào hạ tầng số, AI, "
                "nhân lực số, đổi mới doanh nghiệp và cải thiện năng suất nhân tố tổng hợp."
            )