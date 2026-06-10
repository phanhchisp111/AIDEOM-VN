import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run():
    st.header("📉 Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI & Số hóa")
    
    st.write("""
    Mô hình Cobb-Douglas truyền thống xác định sản lượng đầu ra ($Y$) dựa trên Vốn ($K$) và Lao động ($L$):
    $$Y = A \\cdot K^\\alpha \\cdot L^\\beta$$
    Trong thời đại số, yếu tố **Công nghệ AI & Số hóa ($AI$)** được đưa vào như một nhân tố tác động trực tiếp tới năng suất tổng nhân tố (TFP), làm thay đổi đáng kể sản lượng:
    $$Y = A \\cdot K^\\alpha \\cdot L^\\beta \\cdot (1 + \\gamma \\cdot AI)$$
    *Trong đó $AI \\in [0, 1]$ biểu thị mức độ ứng dụng AI, và $\\gamma$ là hệ số tác động của AI.*
    """)

    # Giao diện chia cột nhập dữ liệu và hiển thị kết quả
    col_input, col_result = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("⚙️ Thông số đầu vào")
        
        # Nhóm thông số truyền thống
        with st.container(border=True):
            st.markdown("**1. Yếu tố truyền thống**")
            A = st.number_input("Hệ số năng suất gốc (A)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            K = st.slider("Quy mô Vốn (K)", min_value=10, max_value=1000, value=100, step=10)
            L = st.slider("Lực lượng Lao động (L)", min_value=10, max_value=1000, value=150, step=10)
            
            alpha = st.slider("Hệ số co giãn của Vốn (α)", min_value=0.05, max_value=0.95, value=0.3, step=0.05)
            beta = st.slider("Hệ số co giãn của Lao động (β)", min_value=0.05, max_value=0.95, value=0.7, step=0.05)
            
        # Nhóm thông số AI & Số hóa
        with st.container(border=True):
            st.markdown("**2. Yếu tố AI & Số hóa**")
            ai_level = st.slider("Mức độ ứng dụng AI (AI)", min_value=0.0, max_value=1.0, value=0.3, step=0.05, format="%d%%")
            gamma = st.slider("Hệ số tác động của AI (γ)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

    with col_result:
        st.subheader("📊 Kết quả tính toán & Phân tích")

        # Tính toán các giá trị
        Y_base = A * (K**alpha) * (L**beta)
        Y_ai = Y_base * (1 + gamma * ai_level)
        delta_Y = Y_ai - Y_base
        growth_pct = (delta_Y / Y_base) * 100 if Y_base > 0 else 0
        scale_sum = alpha + beta

        if scale_sum > 1.001:
            scale_type = "Hiệu suất tăng theo quy mô (Increasing Returns to Scale)"
            scale_desc = "Tăng tỷ lệ vốn và lao động sẽ làm sản lượng tăng với tỷ lệ cao hơn."
        elif scale_sum < 0.999:
            scale_type = "Hiệu suất giảm theo quy mô (Decreasing Returns to Scale)"
            scale_desc = "Tăng tỷ lệ vốn và lao động sẽ làm sản lượng tăng với tỷ lệ thấp hơn."
        else:
            scale_type = "Hiệu suất không đổi theo quy mô (Constant Returns to Scale)"
            scale_desc = "Tăng tỷ lệ vốn và lao động sẽ làm sản lượng tăng với tỷ lệ tương ứng."

        # Hiển thị số liệu trực quan dạng card
        card_base, card_ai, card_gain = st.columns(3)
        with card_base:
            st.metric(label="Sản lượng truyền thống (Y)", value=f"{Y_base:,.2f}")
        with card_ai:
            st.metric(label="Sản lượng khi tích hợp AI (Y_AI)", value=f"{Y_ai:,.2f}", delta=f"+{growth_pct:.2f}%")
        with card_gain:
            st.metric(label="Giá trị thặng dư từ AI (ΔY)", value=f"{delta_Y:,.2f}")

        # Tab kết quả đồ thị và tác nhân phân tích
        tab1, tab2, tab3 = st.tabs(["📈 Đồ thị phân tích", "📋 Bảng số liệu nhạy cảm", "🤖 Tác nhân phân tích AI"])

        with tab1:
            st.write("#### Biểu đồ tác động của mức độ ứng dụng AI đến sản lượng")
            # Tạo dữ liệu vẽ đồ thị
            ai_range = np.linspace(0.0, 1.0, 100)
            Y_range = Y_base * (1 + gamma * ai_range)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ai_range * 100, Y_range, color="#1E88E5", linewidth=2.5, label="Sản lượng Y(AI)")
            ax.axvline(ai_level * 100, color="#FFC107", linestyle="--", label=f"Mức hiện tại ({int(ai_level*100)}%)")
            ax.scatter(ai_level * 100, Y_ai, color="#D81B60", s=80, zorder=5)
            
            ax.set_title("Sự thay đổi sản lượng theo mức độ ứng dụng AI", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mức độ ứng dụng AI (%)")
            ax.set_ylabel("Sản lượng đầu ra (Y)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.write("#### Phân tích độ nhạy cảm của sản lượng theo quy mô Vốn (K) và Lao động (L)")
            
            # Tạo bảng ma trận nhạy cảm
            k_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
            l_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
            
            matrix_data = []
            for k_f in k_factors:
                row = []
                for l_f in l_factors:
                    val = A * ((K * k_f)**alpha) * ((L * l_f)**beta) * (1 + gamma * ai_level)
                    row.append(f"{val:,.1f}")
                matrix_data.append(row)
                
            df_sensitivity = pd.DataFrame(
                matrix_data,
                index=[f"K={int(K*k_f)} ({int(k_f*100)}%)" for k_f in k_factors],
                columns=[f"L={int(L*l_f)} ({int(l_f*100)}%)" for l_f in l_factors]
            )
            st.dataframe(df_sensitivity, use_container_width=True)
            st.caption("Bảng hiển thị sản lượng đầu ra Y khi Vốn và Lao động thay đổi quanh điểm thiết lập hiện tại.")

        with tab3:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5;">
                <h4 style="margin-top: 0; color: #1E88E5;">🤖 Trợ lý phân tích kinh tế AI</h4>
                <p>Dựa trên cấu hình mô hình hiện tại, tôi xin đưa ra các nhận định kinh tế lượng như sau:</p>
                <ul>
                    <li><b>Đánh giá hiệu suất quy mô:</b> Tổng hệ số co giãn là <b>{scale_sum:.2f}</b>, thể hiện nền kinh tế đang ở trạng thái <b>{scale_type}</b>. {scale_desc}</li>
                    <li><b>Đóng góp của AI:</b> Ứng dụng AI ở mức <b>{int(ai_level*100)}%</b> giúp tăng thêm <b>{growth_pct:.1f}%</b> sản lượng so với mô hình kinh tế truyền thống. Điều này cho thấy số hóa có tính chất lan tỏa hiệu quả rất mạnh (hệ số tác động γ = <b>{gamma}</b>).</li>
                    <li><b>Đề xuất chính sách:</b> 
                        <ul>
                            <li>Nếu có thêm 1% ngân sách đầu tư, đầu tư vào <b>{"Lao động (L)" if beta > alpha else "Vốn (K)"}</b> sẽ đem lại tỷ suất sinh lời cận biên cao hơn do hệ số co giãn tương ứng lớn hơn (α={alpha:.2f} vs β={beta:.2f}).</li>
                            <li>Tăng tốc số hóa từ mức {int(ai_level*100)}% lên {min(int(ai_level*100)+20, 100)}% có thể mang lại thêm sản lượng thặng dư đáng kể, đóng vai trò như động lực tăng trưởng mới khi tăng trưởng quy mô lao động hay vốn truyền thống chạm trần.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
