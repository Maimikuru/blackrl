"""Training script for Bi-level RL with Baselines."""

import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_learning_curves


def create_simple_leader_policy(env_spec):
    """Create a simple leader policy (Uniform Random)."""

    def leader_policy(observation, deterministic=False):
        # [修正] 確率リストではなく、実際のアクション(int)を返す必要があります
        if deterministic:
            # 決定論的モードなら0を返す（あるいはargmax）
            return 0

        # 確率[0.5, 0.5]でサンプリング
        return int(np.random.choice([0, 1], p=[0.5, 0.5]))

    return leader_policy


def main():
    print("Initializing Bi-level RL comparison experiment (Feature: Cross-product 18-dim)...")

    # 1. 共通の設定を定義
    # -------------------------------------------------

    # === [修正] 特徴量関数 (直積・18次元) ===
    # ※前回の議論に基づき、状態依存の報酬を正しく表現できる「直積」を採用します
    def feature_fn(state, leader_action, follower_action):
        """Feature function: One-hot encoding of (s, a, b) pair.
        Dim = 3 * 2 * 3 = 18 (Cross-product)
        """
        num_states = 3
        num_leader_actions = 2
        num_follower_actions = 3

        total_dim = num_states * num_leader_actions * num_follower_actions

        # 一意なインデックスを計算
        # index = s * (A*B) + a * (B) + b
        s = int(state.item() if hasattr(state, "item") else state)
        a = int(leader_action.item() if hasattr(leader_action, "item") else leader_action)
        b = int(follower_action.item() if hasattr(follower_action, "item") else follower_action)

        index = s * (num_leader_actions * num_follower_actions) + a * num_follower_actions + b

        # One-hotベクトルを作成
        feature = np.zeros(total_dim, dtype=np.float32)
        feature[index] = 1.0

        return feature

    # ===========================================

    # 共通ハイパーパラメータ
    common_params = {
        "discount_leader": 0.99,
        "discount_follower": 0.8,
        "learning_rate_leader_actor": 1e-5,
        "learning_rate_leader_critic": 1e-4,
        "learning_rate_follower": 0.01,
        "mdce_irl_config": {
            "max_iterations": 1000,
            "tolerance": 0.01,
            "n_soft_q_iterations": 500,
            "n_monte_carlo_samples": 1000,
            "n_jobs": -1,
        },
        "soft_q_config": {
            "learning_rate": 0.1,
            "temperature": 1.0,
            "optimistic_init": 0,
        },
    }

    # 学習の長さ
    train_params = {
        "n_leader_iterations": 100,  # テスト用。本番は1000推奨
        "n_follower_iterations": 500,
        "n_episodes_per_iteration": 1000,
        "verbose": True,
    }
    # -------------------------------------------------

    # 結果を保存する辞書
    results = {}

    # === 実験 1: Proposed (IRL) ===
    print("\n[1/3] Running Proposed Method (IRL)...")
    env_irl = DiscreteToyEnvPaper()
    algo_irl = BilevelRL(
        env_spec=env_irl.spec,
        leader_policy=create_simple_leader_policy(env_irl.spec),
        feature_fn=feature_fn,  # [修正] reward_fn ではなく feature_fn
        **common_params,
    )
    # trainメソッドに oracle_mode を渡せるよう BilevelRL を修正している前提
    stats_irl = algo_irl.train(env=env_irl, oracle_mode="none", **train_params)
    results["Proposed (IRL)"] = stats_irl
    print("  -> Done. Final Return:", stats_irl["leader_return"][-1])

    # === 実験 2: Oracle (Soft Q-Learning) ===
    # 報酬は知っているが、遷移は知らない (Model-Free Baseline)
    print("\n[2/3] Running Oracle Baseline (SoftQL)...")
    env_sql = DiscreteToyEnvPaper()
    algo_sql = BilevelRL(
        env_spec=env_sql.spec,
        leader_policy=create_simple_leader_policy(env_sql.spec),
        feature_fn=feature_fn,  # [修正] 引数名を統一
        **common_params,
    )
    stats_sql = algo_sql.train(env=env_sql, oracle_mode="softql", **train_params)
    results["Oracle (SoftQL)"] = stats_sql
    print("  -> Done. Final Return:", stats_sql["leader_return"][-1])

    # === 実験 3: Oracle (SoftVI) ===
    # 報酬も遷移も知っている (Model-Based / Ideal Baseline)
    print("\n[3/3] Running Oracle Baseline (SoftVI)...")
    env_svi = DiscreteToyEnvPaper()
    algo_svi = BilevelRL(
        env_spec=env_svi.spec,
        leader_policy=create_simple_leader_policy(env_svi.spec),
        feature_fn=feature_fn,  # [修正] 引数名を統一
        **common_params,
    )
    stats_svi = algo_svi.train(env=env_svi, oracle_mode="softvi", **train_params)
    results["Oracle (SoftVI)"] = stats_svi
    print("  -> Done. Final Return:", stats_svi["leader_return"][-1])

    # === 保存とプロット ===
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 全結果を保存
    with open(output_dir / "all_stats.pkl", "wb") as f:
        pickle.dump(results, f)

    # プロット (plot_learning_curves の修正が必要)
    baselines = {
        "Oracle (SoftQL)": results["Oracle (SoftQL)"],
        "Oracle (SoftVI)": results["Oracle (SoftVI)"],
    }

    plot_learning_curves(
        stats=results["Proposed (IRL)"],
        save_path=output_dir / "comparison_curves.png",
        baselines=baselines,
    )

    print(f"\nComparison plot saved to: {output_dir / 'comparison_curves.png'}")


if __name__ == "__main__":
    main()
