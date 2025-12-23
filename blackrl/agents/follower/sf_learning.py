import numpy as np
import torch


class SuccessorFeatureLearning:
    """Successor Feature (SF) Learning for Follower."""

    def __init__(
        self,
        env_spec,
        feature_dim: int,
        feature_fn,
        discount: float = 0.99,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        self.n_s = env_spec.observation_space.n if hasattr(env_spec.observation_space, "n") else 3
        self.n_la = env_spec.leader_action_space.n if hasattr(env_spec.leader_action_space, "n") else 2
        self.n_fa = env_spec.action_space.n if hasattr(env_spec.action_space, "n") else 3

        self.feature_dim = feature_dim
        self.feature_fn = feature_fn
        self.discount = discount
        self.learning_rate = learning_rate
        self.temperature = temperature

        # SFテーブル: ψ(s, leader_a, follower_b) -> vector
        self.sf_table = np.zeros((self.n_s, self.n_la, self.n_fa, feature_dim), dtype=np.float32)

    def get_q_value(self, s, la, fa, w):
        """SFと報酬パラメータwからQ値を計算: Q = w^T · ψ"""
        psi = self.sf_table[int(s), int(la), int(fa)]
        # w が Tensor なら numpy に変換
        w_np = w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else w
        return np.dot(w_np, psi)

    def sample_action(self, s, la, w):
        """現在のSFとwに基づいて行動をサンプリング (Softmax)"""
        q_values = np.array([self.get_q_value(s, la, b, w) for b in range(self.n_fa)])
        # 数値安定化のためのMax引き
        q_values -= np.max(q_values)
        probs = np.exp(q_values / self.temperature)
        probs /= np.sum(probs)
        return int(np.random.choice(self.n_fa, p=probs))

    def update(self, s, la, fa, next_s, done, leader_policy_probs, w):
        """SFの更新 (TD学習)"""
        s, la, fa, next_s = int(s), int(la), int(fa), int(next_s)

        # 1. 現在の特徴量 φ
        phi = self.feature_fn(s, la, fa)
        if hasattr(phi, "detach"):
            phi = phi.detach().cpu().numpy()

        # 2. 次の状態のSF期待値 E[ψ(s')]
        next_sf_expected = np.zeros(self.feature_dim)

        if not done:
            # リーダーの次の一手 P(a'|s')
            p_la_next = leader_policy_probs(next_s)

            for next_la, p_l in enumerate(p_la_next):
                if p_l <= 1e-8:
                    continue

                # フォロワーの次の一手 P(b'|s', a')
                # Q = w^T ψ から確率を計算
                qs = np.array([self.get_q_value(next_s, next_la, b, w) for b in range(self.n_fa)])
                qs -= np.max(qs)
                p_fa = np.exp(qs / self.temperature)
                p_fa /= np.sum(p_fa)

                # その (s', a') における SF の加重平均
                sf_next_la = self.sf_table[next_s, next_la]  # (n_fa, dim)
                expected_sf_for_l = np.dot(p_fa, sf_next_la)

                next_sf_expected += p_l * expected_sf_for_l

        # 3. 更新: ψ ← ψ + α(φ + γE[ψ] - ψ)
        target = phi + self.discount * next_sf_expected
        self.sf_table[s, la, fa] += self.learning_rate * (target - self.sf_table[s, la, fa])

    def get_initial_fev(self, initial_state_dist, leader_policy_probs, w):
        """初期状態分布における特徴量期待値 (Policy FEV) を計算"""
        fev = np.zeros(self.feature_dim)

        for s0, p_s0 in enumerate(initial_state_dist):
            if p_s0 <= 1e-8:
                continue

            p_la_s0 = leader_policy_probs(s0)
            for la, p_l in enumerate(p_la_s0):
                if p_l <= 1e-8:
                    continue

                # フォロワーの方策
                qs = np.array([self.get_q_value(s0, la, b, w) for b in range(self.n_fa)])
                qs -= np.max(qs)
                p_fa = np.exp(qs / self.temperature)
                p_fa /= np.sum(p_fa)

                # 期待SF
                sf_s0 = self.sf_table[s0, la]
                exp_sf = np.dot(p_fa, sf_s0)

                fev += p_s0 * p_l * exp_sf

        return fev

    # blackrl/agents/follower/sf_learning.py に追加推奨

    def get_psi(self, state, leader_action, follower_action):
        """Get the successor feature vector."""
        state = int(state) if not isinstance(state, int) else state
        # self.psi が torch.Tensor か numpy.ndarray か辞書かによる
        # テーブルの場合:
        return self.psi[state, leader_action, follower_action]

    def update_psi_direct(self, state, leader_action, follower_action, target_psi):
        """Update PSI using a pre-calculated target."""
        # TD update: psi <- psi + alpha * (target - psi)
        current = self.get_psi(state, leader_action, follower_action)
        if isinstance(target_psi, torch.Tensor):
            target_psi = target_psi.detach().numpy()  # 必要なら変換

        new_val = current + self.learning_rate * (target_psi - current)

        # 保存
        state = int(state)
        self.psi[state, leader_action, follower_action] = new_val
