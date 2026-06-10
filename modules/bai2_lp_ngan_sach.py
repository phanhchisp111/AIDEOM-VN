import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# ============================================================
# HÀM GIẢI LP BẰNG SCIPY
# ============================================================
def solve_lp_scipy(
    total_budget=100,
    min_i=25,
    min_ai=15,
    min_h=20,
    min_rd=10,
    tech_share=0.35,
    use_fairness=True,
    max_share=0.45
):
    """
    Bài toán:
    Max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4

    Trong đó:
    x1: Hạ tầng số
    x2: AI và dữ liệu
    x3: Nhân lực số
    x4: R&D công nghệ

    scipy.optimize.linprog chỉ giải bài toán Min,
    nên ta đổi Max Z thành Min -Z.
    """

    # Hệ số hàm mục tiêu dạng Min
    c = [-0.85, -1.20, -0.95, -1.35]

    # Ràng buộc A_ub x <= b_ub
    A_ub = [
        [1, 1, 1, 1],                          # Tổng ngân sách <= B
        [-1, 0, 0, 0],                         # x1 >= min_i
        [0, -1, 0, 0],                         # x2 >= min_ai
        [0, 0, -1, 0],                         # x3 >= min_h
        [0, 0, 0, -1],                         # x4 >= min_rd
        [tech_share, -(1 - tech_share), tech_share, -(1 - tech_share)]
        # x2 + x4 >= tech_share * (x1 + x2 + x3 + x4)
    ]

    b_ub = [
        total_budget,
        -min_i,
        -min_ai,
        -min_h,
        -min_rd,
        0
    ]

    # MỞ RỘNG: ràng buộc công bằng ngân sách
    # Không hạng mục nào được nhận quá max_share tổng ngân sách
    if use_fairness:
        max_amount = max_share * total_budget

        A_ub.extend([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        b_ub.extend([
            max_amount,
            max_amount,
            max_amount,
            max_amount
        ])

    bounds = [(0, None)] * 4

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    return res


# ============================================================
# HÀM GIẢI LP BẰNG PULP ĐỂ LẤY SHADOW PRICE
# ============================================================
def solve_lp_pulp(
    total_budget=100,
    min_i=25,
    min_ai=15,
    min_h=20,
    min_rd=10,
    tech_share=0.35,
    use_fairness=True,
    max_share=0.45
):
    try:
        import pulp
    except ImportError:
        return None, None, None, "PuLP chưa được cài. Hãy chạy: python -m pip install pulp"

    model = pulp.LpProblem("Bai_2_Phan_bo_ngan_sach_so", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_Ha_tang_so", lowBound=0)
    x2 = pulp.LpVariable("x2_AI_va_du_lieu", lowBound=0)
    x3 = pulp.LpVariable("x3_Nhan_luc_so", lowBound=0)
    x4 = pulp.LpVariable("x4_RD_cong_nghe", lowBound=0)

    # Hàm mục tiêu
    model += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4, "GDP_ky_vong"

    # Ràng buộc theo đề
    model += x1 + x2 + x3 + x4 <= total_budget, "Ngan_sach_tong"
    model += x1 >= min_i, "Ha_tang_so_toi_thieu"
    model += x2 >= min_ai, "AI_va_du_lieu_toi_thieu"
    model += x3 >= min_h, "Nhan_luc_so_toi_thieu"
    model += x4 >= min_rd, "RD_toi_thieu"
    model += x2 + x4 >= tech_share * (x1 + x2 + x3 + x4), "Ty_trong_cong_nghe_chien_luoc"

    # Mở rộng: ràng buộc công bằng
    if use_fairness:
        max_amount = max_share * total_budget
        model += x1 <= max_amount, "Cong_bang_x1"
        model += x2 <= max_amount, "Cong_bang_x2"
        model += x3 <= max_amount, "Cong_bang_x3"
        model += x4 <= max_amount, "Cong_bang_x4"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    solution_df = pd.DataFrame({
        "Hạng mục": [
            "Hạ tầng số",
            "AI và dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Biến": ["x1", "x2", "x3", "x4"],
        "Phân bổ tối ưu": [
            x1.varValue,
            x2.varValue,
            x3.varValue,
            x4.varValue
        ],
        "Hệ số tác động GDP": [
            0.85,
            1.20,
            0.95,
            1.35
        ]
    })

    dual_rows = []
    for name, constraint in model.constraints.items():
        dual_rows.append({
            "Ràng buộc": name,
            "Shadow price": constraint.pi,
            "Slack": constraint.slack
        })

    dual_df = pd.DataFrame(dual_rows)
    objective_value = pulp.value(model.objective)

    return solution_df, dual_df, objective_value, None


# ============================================================
# HÀM CHẠY STREAMLIT
# ============================================================
def run():
    st.title("💰 Bài 2 – Quy hoạch tuyến tính phân bổ ngân sách số")

    st.write("""
    Bài 2 xây dựng mô hình quy hoạch tuyến tính để phân bổ ngân sách trung ương cho 4 hạng mục đầu tư số:
    hạ tầng số, AI và dữ liệu, nhân lực số, và R&D công nghệ. Mục tiêu là tối đa hóa mức tăng GDP kỳ vọng,
    đồng thời bảo đảm các ràng buộc tối thiểu, tỷ trọng công nghệ chiến lược và phần mở rộng về công bằng ngân sách.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Tổng quan",
        "⚙️ Tham số mô hình",
        "📊 Kết quả tối ưu",
        "📈 Độ nhạy & mở rộng",
        "🤖 Tác nhân phân tích"
    ])

    # ========================================================
    # TAB 1: TỔNG QUAN
    # ========================================================
    with tab1:
        st.subheader("2.1. Bối cảnh bài toán")

        st.write("""
        Giả sử Chính phủ phân bổ **100.000 tỷ VND** ngân sách trung ương cho chuyển đổi số năm 2026.
        Ngân sách được chia cho 4 hạng mục chính.
        """)

        overview_df = pd.DataFrame({
            "Biến": ["x1", "x2", "x3", "x4"],
            "Hạng mục đầu tư": [
                "Hạ tầng số",
                "AI và dữ liệu",
                "Nhân lực số",
                "R&D công nghệ"
            ],
            "Ý nghĩa": [
                "Đầu tư vào hạ tầng số, mạng, trung tâm dữ liệu, nền tảng số",
                "Đầu tư vào AI, dữ liệu lớn, phân tích dữ liệu và tự động hóa",
                "Đào tạo kỹ năng số, kỹ sư AI, nhân lực công nghệ",
                "Nghiên cứu, đổi mới sáng tạo và công nghệ chiến lược"
            ],
            "Hệ số tác động GDP": [
                0.85,
                1.20,
                0.95,
                1.35
            ]
        })

        st.dataframe(overview_df, use_container_width=True)

        st.subheader("2.2. Mô hình toán học")

        st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")

        st.write("Với các ràng buộc theo đề:")

        st.latex(r"x_1 + x_2 + x_3 + x_4 \leq 100")
        st.latex(r"x_1 \geq 25")
        st.latex(r"x_2 \geq 15")
        st.latex(r"x_3 \geq 20")
        st.latex(r"x_4 \geq 10")
        st.latex(r"x_2 + x_4 \geq 0.35(x_1+x_2+x_3+x_4)")
        st.latex(r"x_1,x_2,x_3,x_4 \geq 0")

        st.subheader("2.3. Phần mở rộng")
        st.write("""
        Ngoài yêu cầu trong đề, mô hình được mở rộng bằng **ràng buộc công bằng ngân sách**:
        không hạng mục nào được nhận quá một tỷ lệ nhất định của tổng ngân sách.
        """)

        st.latex(r"x_i \leq sB,\quad i=1,2,3,4")

        st.write("""
        Trong đó, \(s\) là tỷ lệ tối đa cho mỗi hạng mục. Ví dụ, nếu \(s = 45\%\),
        thì không hạng mục nào được nhận quá 45% tổng ngân sách.
        """)

    # ========================================================
    # TAB 2: THAM SỐ MÔ HÌNH
    # ========================================================
    with tab2:
        st.subheader("Điều chỉnh tham số")

        col1, col2 = st.columns(2)

        with col1:
            total_budget = st.slider("Tổng ngân sách B", 80, 160, 100, 5)
            min_i = st.slider("Hạ tầng số tối thiểu x1", 0, 60, 25, 5)
            min_ai = st.slider("AI và dữ liệu tối thiểu x2", 0, 60, 15, 5)

        with col2:
            min_h = st.slider("Nhân lực số tối thiểu x3", 0, 60, 20, 5)
            min_rd = st.slider("R&D tối thiểu x4", 0, 60, 10, 5)
            tech_share = st.slider("Tỷ trọng công nghệ chiến lược AI + R&D", 0.10, 0.70, 0.35, 0.05)

        st.subheader("Mở rộng mô hình")

        use_fairness = st.checkbox(
            "Bật ràng buộc công bằng ngân sách",
            value=True
        )

        max_share = st.slider(
            "Tỷ lệ tối đa cho mỗi hạng mục",
            0.30,
            0.70,
            0.45,
            0.05
        )

        coef_df = pd.DataFrame({
            "Hạng mục": [
                "Hạ tầng số",
                "AI và dữ liệu",
                "Nhân lực số",
                "R&D công nghệ"
            ],
            "Biến": ["x1", "x2", "x3", "x4"],
            "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
            "Mức tối thiểu": [min_i, min_ai, min_h, min_rd]
        })

        st.subheader("Bảng tham số hiện tại")
        st.dataframe(coef_df, use_container_width=True)

        if use_fairness:
            st.info(
                f"Ràng buộc mở rộng đang bật: không hạng mục nào được nhận quá "
                f"{max_share*100:.0f}% tổng ngân sách."
            )
        else:
            st.warning("Ràng buộc công bằng đang tắt. Mô hình sẽ tối đa hóa hiệu quả GDP thuần túy.")

    # ========================================================
    # GIẢI MÔ HÌNH CHÍNH
    # ========================================================
    res = solve_lp_scipy(
        total_budget=total_budget,
        min_i=min_i,
        min_ai=min_ai,
        min_h=min_h,
        min_rd=min_rd,
        tech_share=tech_share,
        use_fairness=use_fairness,
        max_share=max_share
    )

    # ========================================================
    # TAB 3: KẾT QUẢ TỐI ƯU
    # ========================================================
    with tab3:
        st.subheader("Kết quả giải bằng scipy.optimize.linprog")

        if not res.success:
            st.error("Bài toán không khả thi với bộ tham số hiện tại.")
            st.write(res.message)
        else:
            x = res.x
            z_value = -res.fun

            solution_df = pd.DataFrame({
                "Hạng mục": [
                    "Hạ tầng số",
                    "AI và dữ liệu",
                    "Nhân lực số",
                    "R&D công nghệ"
                ],
                "Biến": ["x1", "x2", "x3", "x4"],
                "Phân bổ tối ưu": x,
                "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
                "Đóng góp vào Z": [
                    0.85*x[0],
                    1.20*x[1],
                    0.95*x[2],
                    1.35*x[3]
                ]
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("Giá trị tối ưu Z*", f"{z_value:,.2f}")
            c2.metric("Tổng ngân sách dùng", f"{x.sum():,.2f}")
            c3.metric("Tỷ trọng AI + R&D", f"{((x[1]+x[3])/x.sum())*100:.2f}%")

            st.dataframe(solution_df, use_container_width=True)

            st.subheader("Biểu đồ phân bổ ngân sách tối ưu")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(solution_df["Hạng mục"], solution_df["Phân bổ tối ưu"])
            ax.set_title("Phân bổ ngân sách tối ưu theo hạng mục")
            ax.set_ylabel("Nghìn tỷ VND")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

            # Mở rộng: so sánh có và không có ràng buộc công bằng
            st.subheader("Mở rộng: So sánh có và không có ràng buộc công bằng")

            res_no_fair = solve_lp_scipy(
                total_budget=total_budget,
                min_i=min_i,
                min_ai=min_ai,
                min_h=min_h,
                min_rd=min_rd,
                tech_share=tech_share,
                use_fairness=False,
                max_share=max_share
            )

            res_fair = solve_lp_scipy(
                total_budget=total_budget,
                min_i=min_i,
                min_ai=min_ai,
                min_h=min_h,
                min_rd=min_rd,
                tech_share=tech_share,
                use_fairness=True,
                max_share=max_share
            )

            if res_no_fair.success and res_fair.success:
                compare_fair_df = pd.DataFrame({
                    "Hạng mục": [
                        "Hạ tầng số",
                        "AI và dữ liệu",
                        "Nhân lực số",
                        "R&D công nghệ"
                    ],
                    "Không ràng buộc công bằng": res_no_fair.x,
                    "Có ràng buộc công bằng": res_fair.x
                })

                z_no_fair = -res_no_fair.fun
                z_fair = -res_fair.fun

                st.dataframe(compare_fair_df, use_container_width=True)

                f1, f2, f3 = st.columns(3)
                f1.metric("Z không công bằng", f"{z_no_fair:,.2f}")
                f2.metric("Z có công bằng", f"{z_fair:,.2f}")
                f3.metric("Đánh đổi hiệu quả", f"{z_fair - z_no_fair:,.2f}")

                fig_fair, ax_fair = plt.subplots(figsize=(8, 4))
                x_axis = np.arange(len(compare_fair_df["Hạng mục"]))
                width = 0.35

                ax_fair.bar(
                    x_axis - width/2,
                    compare_fair_df["Không ràng buộc công bằng"],
                    width,
                    label="Không ràng buộc công bằng"
                )

                ax_fair.bar(
                    x_axis + width/2,
                    compare_fair_df["Có ràng buộc công bằng"],
                    width,
                    label="Có ràng buộc công bằng"
                )

                ax_fair.set_xticks(x_axis)
                ax_fair.set_xticklabels(compare_fair_df["Hạng mục"], rotation=20)
                ax_fair.set_title("So sánh phân bổ ngân sách khi thêm ràng buộc công bằng")
                ax_fair.set_ylabel("Nghìn tỷ VND")
                ax_fair.legend()
                ax_fair.grid(axis="y", alpha=0.5)
                st.pyplot(fig_fair)
                plt.close(fig_fair)

        st.subheader("Kết quả giải lại bằng PuLP và shadow price")

        pulp_solution_df, dual_df, pulp_obj, pulp_error = solve_lp_pulp(
            total_budget=total_budget,
            min_i=min_i,
            min_ai=min_ai,
            min_h=min_h,
            min_rd=min_rd,
            tech_share=tech_share,
            use_fairness=use_fairness,
            max_share=max_share
        )

        if pulp_error is not None:
            st.warning(pulp_error)
        else:
            st.write(f"Giá trị tối ưu theo PuLP: **{pulp_obj:,.2f}**")
            st.dataframe(pulp_solution_df, use_container_width=True)

            st.write("Bảng shadow price và slack của từng ràng buộc:")
            st.dataframe(dual_df, use_container_width=True)

    # ========================================================
    # TAB 4: ĐỘ NHẠY & MỞ RỘNG
    # ========================================================
    with tab4:
        st.subheader("Câu 2.4.3 – Phân tích độ nhạy theo ngân sách tổng")

        budget_list = [100, 120, 140]
        sensitivity_rows = []

        for B in budget_list:
            res_B = solve_lp_scipy(
                total_budget=B,
                min_i=min_i,
                min_ai=min_ai,
                min_h=min_h,
                min_rd=min_rd,
                tech_share=tech_share,
                use_fairness=use_fairness,
                max_share=max_share
            )

            if res_B.success:
                sensitivity_rows.append({
                    "Ngân sách B": B,
                    "Z tối ưu": -res_B.fun,
                    "x1 Hạ tầng số": res_B.x[0],
                    "x2 AI và dữ liệu": res_B.x[1],
                    "x3 Nhân lực số": res_B.x[2],
                    "x4 R&D": res_B.x[3]
                })
            else:
                sensitivity_rows.append({
                    "Ngân sách B": B,
                    "Z tối ưu": np.nan,
                    "x1 Hạ tầng số": np.nan,
                    "x2 AI và dữ liệu": np.nan,
                    "x3 Nhân lực số": np.nan,
                    "x4 R&D": np.nan
                })

        sensitivity_df = pd.DataFrame(sensitivity_rows)
        st.dataframe(sensitivity_df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(sensitivity_df["Ngân sách B"], sensitivity_df["Z tối ưu"], marker="o", linewidth=2)
        ax2.set_title("Đường cong Z*(B) khi ngân sách tăng")
        ax2.set_xlabel("Ngân sách tổng, nghìn tỷ VND")
        ax2.set_ylabel("Giá trị tối ưu Z*")
        ax2.grid(True, alpha=0.5)
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Câu 2.4.4 – Kịch bản ưu tiên nhân lực số x3 ≥ 30")

        res_h30 = solve_lp_scipy(
            total_budget=total_budget,
            min_i=min_i,
            min_ai=min_ai,
            min_h=30,
            min_rd=min_rd,
            tech_share=tech_share,
            use_fairness=use_fairness,
            max_share=max_share
        )

        if not res_h30.success:
            st.error("Kịch bản x3 ≥ 30 không khả thi với bộ tham số hiện tại.")
            st.write(res_h30.message)
        else:
            x_h30 = res_h30.x
            z_h30 = -res_h30.fun
            baseline_z = -res.fun if res.success else np.nan
            delta_z = z_h30 - baseline_z

            h30_df = pd.DataFrame({
                "Hạng mục": [
                    "Hạ tầng số",
                    "AI và dữ liệu",
                    "Nhân lực số",
                    "R&D công nghệ"
                ],
                "Phân bổ khi x3 ≥ 30": x_h30
            })

            c4, c5 = st.columns(2)
            c4.metric("Z* khi x3 ≥ 30", f"{z_h30:,.2f}")
            c5.metric("Thay đổi so với ban đầu", f"{delta_z:,.2f}")

            st.dataframe(h30_df, use_container_width=True)

    # ========================================================
    # TAB 5: AI AGENT
    # ========================================================
    with tab5:
        st.subheader("🤖 Tác nhân phân tích kết quả")

        if not res.success:
            st.error("Chưa thể phân tích vì bài toán hiện tại không khả thi.")
            return

        x = res.x
        z_value = -res.fun
        total_used = x.sum()
        tech_ratio = (x[1] + x[3]) / total_used * 100

        try:
            budget_shadow_price = -res.ineqlin.marginals[0]
        except Exception:
            budget_shadow_price = np.nan

        res_h30 = solve_lp_scipy(
            total_budget=total_budget,
            min_i=min_i,
            min_ai=min_ai,
            min_h=30,
            min_rd=min_rd,
            tech_share=tech_share,
            use_fairness=use_fairness,
            max_share=max_share
        )

        if res_h30.success:
            z_h30 = -res_h30.fun
            delta_z_h30 = z_h30 - z_value
            feasible_h30 = "còn khả thi"
        else:
            z_h30 = np.nan
            delta_z_h30 = np.nan
            feasible_h30 = "không khả thi"

        st.markdown("### 🧠 AI Agent – Trả lời câu 2.5")

        with st.container(border=True):
            st.markdown("#### a) Khi ngân sách tổng tăng thêm 1 nghìn tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu? Đây có phải là cận trên hợp lý của chi phí cơ hội vốn công không?")

            st.write(
                f"Theo kết quả tối ưu, shadow price của ràng buộc ngân sách tổng xấp xỉ **{budget_shadow_price:.2f}**. "
                f"Điều này có nghĩa là nếu ngân sách tăng thêm 1 nghìn tỷ VND, giá trị GDP kỳ vọng trong mô hình "
                f"có thể tăng thêm khoảng **{budget_shadow_price:.2f} nghìn tỷ VND**, với điều kiện cấu trúc ràng buộc hiện tại không thay đổi."
            )

            st.write(
                "Tuy nhiên, đây chỉ nên được xem là một chỉ báo định lượng trong mô hình, không phải cận trên tuyệt đối "
                "của chi phí cơ hội vốn công. Trong thực tế, hiệu quả vốn công còn phụ thuộc vào năng lực thực thi, "
                "độ trễ chính sách, chất lượng dự án, khả năng hấp thụ vốn và các ưu tiên xã hội khác."
            )

        with st.container(border=True):
            st.markdown("#### b) Vì sao R&D có hệ số tác động cao nhất nhưng lại có ràng buộc tối thiểu thấp nhất?")

            st.write(
                "R&D có hệ số tác động cao nhất vì đầu tư vào nghiên cứu, đổi mới sáng tạo và công nghệ chiến lược "
                "có thể tạo ra tác động lan tỏa dài hạn cho toàn nền kinh tế. Tuy nhiên, R&D thường có độ rủi ro cao, "
                "thời gian thu hồi vốn dài và đòi hỏi năng lực nghiên cứu, thể chế, doanh nghiệp công nghệ và nhân lực chất lượng cao."
            )

            st.write(
                "Vì vậy, ràng buộc tối thiểu của R&D thấp hơn không có nghĩa là R&D kém quan trọng, mà phản ánh cách tiếp cận thận trọng: "
                "Nhà nước cần bảo đảm một mức đầu tư nền tảng cho R&D, nhưng không nên dồn quá nhiều ngân sách khi hệ sinh thái đổi mới sáng tạo "
                "chưa đủ khả năng hấp thụ."
            )

        with st.container(border=True):
            st.markdown("#### c) Tỷ lệ 35% công nghệ chiến lược AI + R&D có khả thi không khi ngân sách nhà nước còn ưu tiên hạ tầng giao thông và an sinh xã hội?")

            st.write(
                f"Trong nghiệm tối ưu hiện tại, tỷ trọng AI + R&D đạt khoảng **{tech_ratio:.2f}%** tổng ngân sách phân bổ. "
                "Điều này cho thấy ràng buộc 35% có thể khả thi về mặt mô hình, đặc biệt khi AI và R&D có hệ số tác động cao."
            )

            st.write(
                "Tuy nhiên, trong thực tiễn quản lý ngân sách Việt Nam, tỷ lệ này cần được cân nhắc kỹ vì ngân sách nhà nước "
                "còn phải dành cho hạ tầng giao thông, y tế, giáo dục, an sinh xã hội và quốc phòng. Do đó, thay vì áp dụng cứng nhắc, "
                "tỷ lệ 35% nên được xem là mục tiêu định hướng hoặc tỷ lệ tối thiểu cho các chương trình chuyển đổi số trọng điểm."
            )

            st.write(
                "Một phương án hợp lý là triển khai theo lộ trình: trước hết bảo đảm hạ tầng số và nhân lực số, sau đó tăng dần tỷ trọng AI và R&D "
                "khi năng lực hấp thụ công nghệ của doanh nghiệp và khu vực công được cải thiện."
            )

        st.markdown("### 🔎 Phân tích thêm theo yêu cầu 2.4.4")

        with st.container(border=True):
            st.markdown("#### Kịch bản ưu tiên nhân lực số: x3 ≥ 30")

            st.write(
                f"Khi Chính phủ nâng ràng buộc nhân lực số lên **x3 ≥ 30**, bài toán **{feasible_h30}**."
            )

            if res_h30.success:
                st.write(
                    f"Giá trị tối ưu mới là **{z_h30:,.2f}**, thay đổi **{delta_z_h30:,.2f}** so với nghiệm ban đầu. "
                    "Nếu Z* giảm, điều này phản ánh đánh đổi chính sách: ưu tiên nhân lực số giúp giải quyết thiếu hụt kỹ sư AI, "
                    "nhưng có thể làm giảm một phần GDP kỳ vọng ngắn hạn do ngân sách bị chuyển khỏi hạng mục có hệ số tác động cao hơn."
                )

        st.markdown("### ⚖️ Phân tích mở rộng: ràng buộc công bằng ngân sách")

        res_no_fair = solve_lp_scipy(
            total_budget=total_budget,
            min_i=min_i,
            min_ai=min_ai,
            min_h=min_h,
            min_rd=min_rd,
            tech_share=tech_share,
            use_fairness=False,
            max_share=max_share
        )

        res_fair = solve_lp_scipy(
            total_budget=total_budget,
            min_i=min_i,
            min_ai=min_ai,
            min_h=min_h,
            min_rd=min_rd,
            tech_share=tech_share,
            use_fairness=True,
            max_share=max_share
        )

        with st.container(border=True):
            st.markdown("#### d) Khi thêm ràng buộc công bằng ngân sách, kết quả thay đổi như thế nào?")

            if res_no_fair.success and res_fair.success:
                z_no_fair = -res_no_fair.fun
                z_fair = -res_fair.fun
                delta_fair = z_fair - z_no_fair

                st.write(
                    f"Khi không áp dụng ràng buộc công bằng, giá trị tối ưu đạt **{z_no_fair:,.2f}**. "
                    f"Khi áp dụng ràng buộc công bằng, giá trị tối ưu đạt **{z_fair:,.2f}**."
                )

                st.write(
                    f"Mức thay đổi là **{delta_fair:,.2f}**. Nếu giá trị này âm, điều đó cho thấy việc phân bổ ngân sách cân bằng hơn "
                    "có thể làm giảm một phần hiệu quả GDP kỳ vọng trong ngắn hạn."
                )

                st.write(
                    "Tuy nhiên, ràng buộc công bằng có ý nghĩa chính sách quan trọng vì giúp tránh việc ngân sách bị dồn quá nhiều "
                    "vào một hạng mục duy nhất. Đây là đánh đổi giữa **hiệu quả kinh tế** và **tính cân bằng trong phân bổ ngân sách công**."
                )
            else:
                st.write(
                    "Một trong hai mô hình không khả thi, vì vậy chưa thể so sánh đầy đủ tác động của ràng buộc công bằng."
                )

        with st.container(border=True):
            st.markdown("#### e) Hàm ý chính sách từ phần mở rộng")

            st.write(
                "Phần mở rộng cho thấy bài toán phân bổ ngân sách không chỉ là tối đa hóa GDP kỳ vọng, "
                "mà còn cần xét đến tính cân bằng, khả năng hấp thụ vốn và ưu tiên phát triển dài hạn."
            )

            st.write(
                "Trong bối cảnh Việt Nam, ràng buộc công bằng ngân sách có thể giúp bảo đảm rằng hạ tầng số, AI, nhân lực số "
                "và R&D đều được đầu tư ở mức hợp lý. Điều này phù hợp với định hướng phát triển kinh tế số bền vững, "
                "tránh tình trạng đầu tư lệch quá mạnh vào một lĩnh vực trong khi các điều kiện nền tảng chưa được củng cố."
            )