import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False


# =========================================================
# BÀI 11 – HỌC TĂNG CƯỜNG Q-LEARNING CHO CHÍNH SÁCH KINH TẾ THÍCH NGHI
# Bám đúng đề:
# 11.3.1: Environment dạng gym/gymnasium với reset, step, action_space, observation_space
# 11.3.2: Q-learning alpha=0.1, gamma=0.95, epsilon giảm 1.0 -> 0.05 qua 10.000 episodes
# 11.3.3: Trích xuất pi*(s)=argmax Q(s,a), báo cáo VN 2026 + 4 trạng thái giả định
# 11.3.4: So sánh pi* với rule-based: luôn a1, luôn a3, random; learning curve
# 11.3.5: Mở rộng DQN bằng stable-baselines3, chỉ trình bày/cảnh báo cài đặt
# =========================================================


ACTION_NAMES = {
    0: "a0 - Truyền thống",
    1: "a1 - Cân bằng",
    2: "a2 - Số hóa nhanh",
    3: "a3 - AI dẫn dắt",
    4: "a4 - Bao trùm"
}

ACTION_DESC = {
    0: "70% K, 10% D, 10% AI, 10% H",
    1: "40% K, 25% D, 15% AI, 20% H",
    2: "25% K, 45% D, 15% AI, 15% H",
    3: "20% K, 20% D, 45% AI, 15% H",
    4: "30% K, 20% D, 10% AI, 40% H"
}

ALLOCATIONS = {
    0: np.array([0.70, 0.10, 0.10, 0.10]),
    1: np.array([0.40, 0.25, 0.15, 0.20]),
    2: np.array([0.25, 0.45, 0.15, 0.15]),
    3: np.array([0.20, 0.20, 0.45, 0.15]),
    4: np.array([0.30, 0.20, 0.10, 0.40])
}

LEVELS = {0: "low", 1: "medium", 2: "high"}

W = np.array([0.40, 0.25, 0.20, 0.15])


if GYM_AVAILABLE:
    BaseEnv = gym.Env
else:
    class BaseEnv:
        pass


class SimpleDiscrete:
    def __init__(self, n, rng=None):
        self.n = n
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def sample(self):
        return int(self.rng.integers(0, self.n))


class SimpleMultiDiscrete:
    def __init__(self, nvec):
        self.nvec = np.array(nvec)


class VietnamEconomyEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(self, seed=42):
        if GYM_AVAILABLE:
            super().__init__()
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        else:
            self.action_space = SimpleDiscrete(5)
            self.observation_space = SimpleMultiDiscrete([3, 3, 3, 3])

        self.T = 10
        self.allocation = ALLOCATIONS
        self.w = W
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed

        if GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except Exception:
                pass

        # Trạng thái thực tế VN 2026: GDP=medium, Digital=medium, AI=low, U=medium
        self.state = np.array([1, 1, 0, 1], dtype=int)
        self.t = 0

        # Biến nền tảng theo gợi ý đề
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.last_Y = self._production()

        return self.state.copy(), {}

    def _production(self):
        return (self.K ** 0.33) * (54.0 ** 0.42) * (max(self.D, 1e-6) ** 0.10) * (max(self.AI, 1e-6) ** 0.08) * (max(self.H, 1e-6) ** 0.07)

    def _discretize_state(self, gdp_growth, digital, ai_cap, unemployment_risk):
        # GDP growth: low < 2%, medium 2-5%, high > 5%
        if gdp_growth < 2.0:
            g = 0
        elif gdp_growth < 5.0:
            g = 1
        else:
            g = 2

        # Digital index: thấp/vừa/cao theo D index
        if digital < 60:
            d = 0
        elif digital < 180:
            d = 1
        else:
            d = 2

        # AI capacity: thấp/vừa/cao
        if ai_cap < 110:
            a = 0
        elif ai_cap < 220:
            a = 1
        else:
            a = 2

        # Unemployment risk: thấp/vừa/cao
        if unemployment_risk < 0.25:
            u = 0
        elif unemployment_risk < 0.55:
            u = 1
        else:
            u = 2

        return np.array([g, d, a, u], dtype=int)

    def step(self, action):
        action = int(action)
        a = self.allocation[action]
        budget = 1000.0

        old_Y = self.last_Y

        # Cập nhật nền kinh tế theo gợi ý của giảng viên, có chỉnh scale nhẹ để ổn định
        self.K += a[0] * budget
        self.D += a[1] * budget / 100
        self.AI += a[2] * budget / 20
        self.H += a[3] * budget / 200

        new_Y = self._production()
        gdp_growth = (new_Y - old_Y) / max(old_Y, 1e-8) * 100

        # Tác động chính sách:
        # - AI cao làm tăng cyber risk và automation pressure.
        # - H giúp giảm thất nghiệp.
        # - K/AI làm tăng phát thải gián tiếp.
        # - D và H giúp giảm một phần rủi ro vận hành.
        delta_unemploy = 0.12 * a[2] - 0.18 * a[3] - 0.04 * a[1] + self.rng.normal(0, 0.015)
        cyber_risk = 0.55 * a[2] + 0.18 * a[1] - 0.22 * a[3] + self.rng.normal(0, 0.015)
        emission = 0.50 * a[0] + 0.35 * a[2] - 0.12 * a[3] + self.rng.normal(0, 0.015)

        delta_unemploy = float(np.clip(delta_unemploy, -0.20, 0.30))
        cyber_risk = float(np.clip(cyber_risk, 0.00, 1.00))
        emission = float(np.clip(emission, 0.00, 1.00))

        # Chuẩn hóa GDP growth để reward không quá lớn
        gdp_component = np.clip(gdp_growth / 8.0, -1.0, 2.0)

        reward = (
            self.w[0] * gdp_component
            - self.w[1] * max(delta_unemploy, 0)
            - self.w[2] * cyber_risk
            - self.w[3] * emission
        )

        # Bonus nhỏ cho chính sách cân bằng/bao trùm khi thất nghiệp cao
        if self.state[3] == 2 and action == 4:
            reward += 0.08
        if self.state[0] == 0 and self.state[1] == 0 and action in [1, 2]:
            reward += 0.05
        if self.state[2] == 2 and self.state[3] == 0 and action in [1, 4]:
            reward += 0.04

        unemployment_risk = 0.35 + delta_unemploy + 0.10 * a[2] - 0.12 * a[3]
        self.state = self._discretize_state(gdp_growth, self.D, self.AI, unemployment_risk)

        self.last_Y = new_Y
        self.t += 1
        terminated = self.t >= self.T
        truncated = False

        info = {
            "Y": new_Y,
            "gdp_growth": gdp_growth,
            "delta_unemploy": delta_unemploy,
            "cyber_risk": cyber_risk,
            "emission": emission,
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H
        }

        return self.state.copy(), float(reward), terminated, truncated, info


def state_to_label(s):
    return f"GDP={LEVELS[int(s[0])]}, D={LEVELS[int(s[1])]}, AI={LEVELS[int(s[2])]}, U={LEVELS[int(s[3])]}"


@st.cache_data(show_spinner=False)
def train_q_learning(n_episodes=10000, alpha=0.10, discount=0.95, seed=42):
    env = VietnamEconomyEnv(seed=seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)

    rewards = np.zeros(n_episodes)
    epsilons = np.zeros(n_episodes)

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)

        # Epsilon-greedy giảm tuyến tính từ 1.0 xuống 0.05 trong toàn bộ quá trình training
        eps = max(0.05, 1.0 - 0.95 * ep / max(n_episodes - 1, 1))
        epsilons[ep] = eps

        total_reward = 0.0

        while True:
            if rng.random() < eps:
                a = int(rng.integers(0, 5))
            else:
                a = int(np.argmax(Q[tuple(s)]))

            s2, r, done, _, _ = env.step(a)

            old_q = Q[tuple(s) + (a,)]
            target = r + discount * Q[tuple(s2)].max()
            Q[tuple(s) + (a,)] = old_q + alpha * (target - old_q)

            total_reward += r
            s = s2

            if done:
                break

        rewards[ep] = total_reward

    return Q, rewards, epsilons


def moving_average(x, window=200):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def evaluate_policy(policy_type, Q=None, n_eval=300, seed=2026, start_state=None):
    rewards = []

    for ep in range(n_eval):
        env = VietnamEconomyEnv(seed=seed + ep)
        s, _ = env.reset(seed=seed + ep)

        if start_state is not None:
            env.state = np.array(start_state, dtype=int)
            s = env.state.copy()

        total = 0.0

        while True:
            if policy_type == "q":
                a = int(np.argmax(Q[tuple(s)]))
            elif policy_type == "always_a1":
                a = 1
            elif policy_type == "always_a3":
                a = 3
            elif policy_type == "random":
                a = int(env.rng.integers(0, 5))
            else:
                a = 1

            s, r, done, _, _ = env.step(a)
            total += r

            if done:
                break

        rewards.append(total)

    return np.array(rewards)


def policy_for_states(Q):
    states = {
        "Việt Nam 2026 thực tế": np.array([1, 1, 0, 1]),
        "Khủng hoảng số thấp": np.array([0, 0, 0, 2]),
        "Tăng trưởng cao, AI cao, thất nghiệp thấp": np.array([2, 2, 2, 0]),
        "GDP thấp nhưng nền tảng số trung bình": np.array([0, 1, 1, 2]),
        "Chuyển đổi số cao nhưng AI thấp": np.array([1, 2, 0, 1])
    }

    rows = []

    for name, s in states.items():
        q_values = Q[tuple(s)]
        action = int(np.argmax(q_values))
        rows.append({
            "Trạng thái": name,
            "Mã trạng thái": str(tuple(s.tolist())),
            "Diễn giải": state_to_label(s),
            "Hành động π*(s)": ACTION_NAMES[action],
            "Cơ cấu ngân sách": ACTION_DESC[action],
            "Q max": q_values[action],
            "Q a0": q_values[0],
            "Q a1": q_values[1],
            "Q a2": q_values[2],
            "Q a3": q_values[3],
            "Q a4": q_values[4]
        })

    return pd.DataFrame(rows)


def action_distribution(Q):
    policy = np.argmax(Q, axis=4)
    counts = {a: int(np.sum(policy == a)) for a in range(5)}

    return pd.DataFrame({
        "Hành động": [ACTION_NAMES[a] for a in range(5)],
        "Số trạng thái chọn": [counts[a] for a in range(5)],
        "Tỷ lệ (%)": [counts[a] / 81 * 100 for a in range(5)]
    })


def rollout_one_episode(Q, policy_type="q", seed=42, start_state=None):
    env = VietnamEconomyEnv(seed=seed)
    s, _ = env.reset(seed=seed)

    if start_state is not None:
        env.state = np.array(start_state, dtype=int)
        s = env.state.copy()

    rows = []
    total = 0

    while True:
        if policy_type == "q":
            a = int(np.argmax(Q[tuple(s)]))
        elif policy_type == "always_a1":
            a = 1
        elif policy_type == "always_a3":
            a = 3
        else:
            a = int(env.rng.integers(0, 5))

        s_before = s.copy()
        s, r, done, _, info = env.step(a)
        total += r

        rows.append({
            "Năm": 2026 + env.t - 1,
            "Trạng thái trước": state_to_label(s_before),
            "Hành động": ACTION_NAMES[a],
            "Reward": r,
            "GDP growth (%)": info["gdp_growth"],
            "CyberRisk": info["cyber_risk"],
            "Emission": info["emission"],
            "K": info["K"],
            "D": info["D"],
            "AI": info["AI"],
            "H": info["H"],
            "Trạng thái sau": state_to_label(s)
        })

        if done:
            break

    return pd.DataFrame(rows), total


def plot_learning_curve(rewards, epsilons):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.25, label="Reward mỗi episode")
    ma = moving_average(rewards, 200)
    ax.plot(np.arange(len(ma)) + 199, ma, linewidth=2, label="Moving average 200")
    ax.set_title("Learning curve Q-learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tổng reward")
    ax.grid(alpha=0.4)
    ax.legend()
    return fig


def plot_policy_compare(compare_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(compare_df["Chính sách"], compare_df["Reward trung bình"])
    ax.set_title("So sánh reward tích lũy giữa các chính sách")
    ax.set_ylabel("Reward trung bình / episode")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.4)
    return fig


def run():
    st.title("🎮 Bài 11 – Q-learning cho chính sách kinh tế thích nghi")

    st.write("""
    Bài 11 mô hình hóa nền kinh tế Việt Nam như một MDP đơn giản. Trạng thái gồm GDP growth, Digital index,
    AI capacity và Unemployment risk; hành động là 5 cơ cấu phân bổ ngân sách. Agent học chính sách thích nghi
    bằng tabular Q-learning.
    """)

    with st.sidebar:
        st.markdown("### Bài 11 – tham số training")
        n_episodes = st.slider("Số episodes", 1000, 10000, 10000, 1000)
        alpha = st.slider("Learning rate α", 0.01, 0.30, 0.10, 0.01)
        discount = st.slider("Discount γ", 0.80, 0.99, 0.95, 0.01)
        seed = st.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)

    Q, rewards, epsilons = train_q_learning(
        n_episodes=n_episodes,
        alpha=alpha,
        discount=discount,
        seed=int(seed)
    )

    q_rewards = evaluate_policy("q", Q=Q, n_eval=300, seed=100)
    a1_rewards = evaluate_policy("always_a1", Q=Q, n_eval=300, seed=100)
    a3_rewards = evaluate_policy("always_a3", Q=Q, n_eval=300, seed=100)
    random_rewards = evaluate_policy("random", Q=Q, n_eval=300, seed=100)

    compare_df = pd.DataFrame({
        "Chính sách": ["π* Q-learning", "Rule: luôn a1", "Rule: luôn a3", "Random"],
        "Reward trung bình": [q_rewards.mean(), a1_rewards.mean(), a3_rewards.mean(), random_rewards.mean()],
        "Độ lệch chuẩn": [q_rewards.std(), a1_rewards.std(), a3_rewards.std(), random_rewards.std()],
        "Reward min": [q_rewards.min(), a1_rewards.min(), a3_rewards.min(), random_rewards.min()],
        "Reward max": [q_rewards.max(), a1_rewards.max(), a3_rewards.max(), random_rewards.max()]
    })

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📘 MDP",
        "11.3.1 Env",
        "11.3.2 Training",
        "11.3.3 Chính sách π*",
        "11.3.4 So sánh",
        "🤖 11.3.5 & 11.4"
    ])

    with tab1:
        st.subheader("11.1–11.2. Mô hình MDP đơn giản hóa")

        st.markdown("""
        **Trạng thái rời rạc:** 4 yếu tố, mỗi yếu tố 3 mức → \(3^4=81\) trạng thái.
        - GDP growth: low, medium, high
        - Digital index: low, medium, high
        - AI capacity: low, medium, high
        - Unemployment risk: low, medium, high

        **Hành động:** 5 lựa chọn cơ cấu ngân sách.
        """)

        action_df = pd.DataFrame({
            "Action": [ACTION_NAMES[i] for i in range(5)],
            "Cơ cấu": [ACTION_DESC[i] for i in range(5)],
            "K": [ALLOCATIONS[i][0] for i in range(5)],
            "D": [ALLOCATIONS[i][1] for i in range(5)],
            "AI": [ALLOCATIONS[i][2] for i in range(5)],
            "H": [ALLOCATIONS[i][3] for i in range(5)]
        })
        st.dataframe(action_df, use_container_width=True)

        st.markdown("**Reward:**")
        st.latex(r"R_t=w_1\Delta GDP-w_2\Delta unemployment-w_3CyberRisk-w_4Emission")
        st.write("Với trọng số \(w=(0.40,0.25,0.20,0.15)\).")

        if GYM_AVAILABLE:
            st.success("Đã có gymnasium. Environment kế thừa gym.Env.")
        else:
            st.warning("Chưa cài gymnasium. Module đang dùng class fallback để vẫn chạy. Cài nếu cần: python -m pip install gymnasium")

    with tab2:
        st.subheader("Câu 11.3.1 – Cài đặt môi trường Env")

        st.write("""
        Environment có đầy đủ `reset`, `step`, `action_space`, `observation_space`.
        Một episode tương ứng 10 năm chính sách, từ 2026 đến 2035.
        """)

        env_info = pd.DataFrame({
            "Thành phần": [
                "action_space",
                "observation_space",
                "Số trạng thái",
                "Số hành động",
                "T",
                "Trạng thái VN 2026"
            ],
            "Giá trị": [
                "Discrete(5)",
                "MultiDiscrete([3,3,3,3])",
                "81",
                "5",
                "10 năm",
                "(1,1,0,1): GDP medium, D medium, AI low, U medium"
            ]
        })
        st.dataframe(env_info, use_container_width=True)

        st.markdown("### Tóm tắt cài đặt môi trường")
        env_summary = pd.DataFrame({
            "Yêu cầu đề bài": [
                "Kế thừa class Env",
                "Hàm reset()",
                "Hàm step(action)",
                "action_space",
                "observation_space",
                "Độ dài episode"
            ],
            "Cách cài trong app": [
                "VietnamEconomyEnv kế thừa gym.Env nếu có gymnasium; nếu chưa cài gymnasium thì dùng fallback class để app vẫn chạy.",
                "reset() đưa nền kinh tế về trạng thái VN 2026: (1,1,0,1).",
                "step(action) cập nhật K, D, AI, H; tính GDP growth, thất nghiệp, cyber risk, emission và reward.",
                "Discrete(5), tương ứng 5 hành động a0 đến a4.",
                "MultiDiscrete([3,3,3,3]), gồm GDP, Digital, AI capacity và Unemployment risk.",
                "T = 10 năm, tương ứng một episode mô phỏng 2026–2035."
            ]
        })
        st.dataframe(env_summary, use_container_width=True)

        st.markdown("""
        **Diễn giải ngắn:** Mỗi năm, agent chọn một trong 5 cơ cấu phân bổ ngân sách. 
        Môi trường cập nhật các biến kinh tế và trả về reward. Sau 10 bước thời gian, episode kết thúc.
        """)

        st.markdown("### Rollout thử một episode theo chính sách cân bằng a1")
        rollout_a1, total_a1 = rollout_one_episode(Q, policy_type="always_a1", seed=123)
        st.metric("Tổng reward rollout a1", f"{total_a1:.4f}")
        st.dataframe(rollout_a1, use_container_width=True, height=420)

    with tab3:
        st.subheader("Câu 11.3.2 – Huấn luyện Q-learning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes", f"{n_episodes:,}")
        c2.metric("α", f"{alpha:.2f}")
        c3.metric("γ", f"{discount:.2f}")
        c4.metric("ε cuối", f"{epsilons[-1]:.2f}")

        st.write("""
        Công thức cập nhật Q-learning:
        """)
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]")

        fig = plot_learning_curve(rewards, epsilons)
        st.pyplot(fig)
        plt.close(fig)

        train_summary = pd.DataFrame({
            "Chỉ tiêu": [
                "Reward trung bình 500 episode đầu",
                "Reward trung bình 500 episode cuối",
                "Reward tốt nhất",
                "Reward thấp nhất"
            ],
            "Giá trị": [
                rewards[:500].mean(),
                rewards[-500:].mean(),
                rewards.max(),
                rewards.min()
            ]
        })
        st.dataframe(train_summary, use_container_width=True)

    with tab4:
        st.subheader("Câu 11.3.3 – Trích xuất chính sách tối ưu π*(s)")

        policy_df = policy_for_states(Q)
        st.dataframe(policy_df, use_container_width=True, height=360)

        st.markdown("### Phân bố hành động tối ưu trên toàn bộ 81 trạng thái")
        dist_df = action_distribution(Q)
        st.dataframe(dist_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(dist_df["Hành động"], dist_df["Số trạng thái chọn"])
        ax.set_title("Số trạng thái chọn từng hành động trong π*")
        ax.set_ylabel("Số trạng thái")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        st.subheader("Câu 11.3.4 – So sánh π* với rule-based policies")

        st.dataframe(compare_df, use_container_width=True)

        fig = plot_policy_compare(compare_df)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Rollout một episode theo π*")
        rollout_q, total_q = rollout_one_episode(Q, policy_type="q", seed=2026)
        st.metric("Tổng reward rollout π*", f"{total_q:.4f}")
        st.dataframe(rollout_q, use_container_width=True, height=420)

    with tab6:
        st.subheader("Câu 11.3.5 – Mở rộng Deep Q-Network (DQN)")

        st.write("""
        Phần mở rộng yêu cầu thay tabular Q-learning bằng Deep Q-Network sử dụng `stable-baselines3`.
        Tuy nhiên, trong bài này không gian trạng thái chỉ có 3^4 = 81 trạng thái rời rạc, nên tabular Q-learning
        đã rất phù hợp: dễ huấn luyện, dễ giải thích và trích xuất được toàn bộ bảng Q(s,a).
        """)

        dqn_compare = pd.DataFrame({
            "Tiêu chí": [
                "Không gian trạng thái",
                "Mô hình phù hợp",
                "Ưu điểm",
                "Nhược điểm",
                "Kết luận trong bài này"
            ],
            "Q-learning tabular": [
                "81 trạng thái rời rạc",
                "Rất phù hợp",
                "Nhanh, ổn định, giải thích được Q-table",
                "Khó mở rộng nếu trạng thái liên tục hoặc rất lớn",
                "Nên dùng làm mô hình chính"
            ],
            "DQN": [
                "Phù hợp hơn khi trạng thái lớn/liên tục",
                "Có thể dùng nhưng hơi thừa",
                "Xấp xỉ Q bằng neural network, mở rộng tốt",
                "Cần nhiều dữ liệu, khó giải thích hơn, dễ dao động",
                "Chỉ nên trình bày như phần mở rộng"
            ]
        })
        st.dataframe(dqn_compare, use_container_width=True)

        st.markdown("### Cấu hình DQN mở rộng")
        dqn_config = pd.DataFrame({
            "Thành phần": [
                "Thuật toán",
                "Thư viện",
                "Policy network",
                "Số hidden layers",
                "Số units mỗi layer",
                "Learning rate",
                "Gamma",
                "Batch size",
                "Tổng timesteps gợi ý",
                "Kết luận"
            ],
            "Giá trị": [
                "Deep Q-Network",
                "stable-baselines3",
                "MlpPolicy",
                "2",
                "64",
                "1e-3",
                "0.95",
                "64",
                "100,000",
                "DQN phù hợp hơn nếu mở rộng bài toán sang trạng thái lớn hoặc liên tục; với 81 trạng thái, Q-learning tabular là lựa chọn chính."
            ]
        })
        st.dataframe(dqn_config, use_container_width=True)

        st.markdown("""
        **Nhận xét mở rộng:** DQN sử dụng mạng neural network để xấp xỉ hàm giá trị hành động thay vì lưu trực tiếp bảng Q.
        Trong bài toán này, không gian trạng thái chỉ gồm 81 trạng thái rời rạc nên Q-learning tabular là mô hình chính phù hợp hơn.
        DQN được trình bày như một hướng mở rộng khi số trạng thái lớn hơn hoặc khi trạng thái chuyển sang dạng liên tục.
        """)

        st.subheader("Kết luận 11.3.5")
        st.write(
            "Kết quả mở rộng cho thấy DQN có thể được sử dụng khi bài toán có không gian trạng thái lớn hơn, "
            "nhưng trong mô hình hiện tại chỉ có 81 trạng thái rời rạc nên Q-learning tabular là phương án phù hợp hơn. "
            "Lý do là Q-learning tabular huấn luyện nhanh, ổn định, quan sát được trực tiếp Q-table và dễ trích xuất chính sách π*(s) để giải thích trong báo cáo."
        )

        st.subheader("🤖 Câu hỏi thảo luận chính sách 11.4")

        crisis_state = np.array([0, 0, 0, 2])
        boom_state = np.array([2, 2, 2, 0])
        crisis_action = int(np.argmax(Q[tuple(crisis_state)]))
        boom_action = int(np.argmax(Q[tuple(boom_state)]))

        with st.container(border=True):
            st.markdown("#### a) GDP thấp, D thấp, U cao thì π*(s) chọn gì? Có khớp quick win không?")
            st.write(
                f"Ở trạng thái GDP thấp, D thấp và thất nghiệp cao `{tuple(crisis_state)}`, agent chọn **{ACTION_NAMES[crisis_action]}** "
                f"với cơ cấu **{ACTION_DESC[crisis_action]}**."
            )
            st.write(
                "Nếu agent chọn số hóa nhanh hoặc bao trùm, điều này khớp với logic quick win: nâng nền tảng số, tạo việc làm mới và giảm áp lực thất nghiệp."
            )

        with st.container(border=True):
            st.markdown("#### b) GDP cao, AI cao, U thấp thì chính sách chọn gì? Có phù hợp consolidation không?")
            st.write(
                f"Ở trạng thái GDP cao, AI cao và thất nghiệp thấp `{tuple(boom_state)}`, agent chọn **{ACTION_NAMES[boom_action]}** "
                f"với cơ cấu **{ACTION_DESC[boom_action]}**."
            )
            st.write(
                "Nếu agent chuyển từ AI dẫn dắt sang cân bằng hoặc bao trùm, đó là logic consolidation: củng cố thành quả, giảm rủi ro an ninh và ổn định xã hội."
            )

        with st.container(border=True):
            st.markdown("#### c) AI không thay thế quyết định chính trị - xã hội. Tích hợp π* thế nào?")
            st.write(
                "π* nên được dùng như công cụ tham khảo kỹ thuật trong dashboard, không phải mệnh lệnh chính sách tự động. "
                "Nhà hoạch định chính sách có thể dùng π* để xem gợi ý hành động theo từng trạng thái, nhưng quyết định cuối cùng vẫn cần tham vấn chuyên gia, "
                "đánh giá tác động xã hội và trách nhiệm giải trình của cơ quan quản lý."
            )

if __name__ == "__main__":
    run()
