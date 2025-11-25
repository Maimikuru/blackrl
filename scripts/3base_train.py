"""Training script for Bi-level RL with Baselines."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_learning_curves

def create_simple_leader_policy(env_spec):
    """Create a simple leader policy."""
    def leader_policy(observation, deterministic=False):
        return [0.5, 0.5]
    return leader_policy

def main():
    print("Initializing Bi-level RL comparison experiment...")

    # 1. 共通の設定を定義
    # -------------------------------------------------
    env = DiscreteToyEnvPaper()

    # 18次元の特徴量 (直積)
    def feature_fn(state, leader_action, follower_action):
        n_states = 3
        n_leader_actions = 2
        n_follower_actions = 3
        feature_dim = n_states * n_leader_actions * n_follower_actions

        s = int(state.item()) if hasattr(state, "item") else int(state)
        l = int(leader_action.item()) if hasattr(leader_action, "item") else int(leader_action)
        f = int(follower_action.item()) if hasattr(follower_action, "item") else int(follower_action)

        unique_index = (s * n_leader_actions * n_follower_actions) + \
                       (l * n_follower_actions) + \
                       f
        feature_vector = np.zeros(feature_dim)
        feature_vector[unique_index] = 1.0
        return feature_vector

    # 共通ハイパーパラメータ
    common_params = {
        "discount_leader": 0.99,
        "discount_follower": 0.8,
        "learning_rate_leader": 1e-6, # クリップありならこれくらい
        "learning_rate_follower": 0.01,
        "mdce_irl_config": {
            "max_iterations": 1000,
            "tolerance": 0.025,
            "n_soft_q_iterations": 1000,
            "n_monte_carlo_samples": 5000,
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
        "n_leader_iterations": 100, # テスト用。本番は1000推奨
        "n_follower_iterations": 500,
        "n_episodes_per_iteration": 50,
        "verbose": False, # ログを抑制して進捗だけ見る
    }
    # -------------------------------------------------

    # 結果を保存する辞書
    results = {}

    # === 実験 1: Proposed (IRL) ===
    print("\n[1/3] Running Proposed Method (IRL)...")
    env_irl = DiscreteToyEnvPaper() # 環境もリセット推奨
    algo_irl = BilevelRL(
        env_spec=env_irl.spec,
        leader_policy=create_simple_leader_policy(env_irl.spec),
        reward_fn=feature_fn,
        **common_params
    )
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
        reward_fn=feature_fn,
        **common_params
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
        reward_fn=feature_fn,
        **common_params
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

    # プロット
    # plot_learning_curves は stats, save_path, baselines を受け取るように修正されている前提
    # Proposedをメイン(stats)として渡し、残りを baselines 辞書として渡す

    baselines = {
        "Oracle (SoftQL)": results["Oracle (SoftQL)"],
        "Oracle (SoftVI)": results["Oracle (SoftVI)"]
    }

    plot_learning_curves(
        stats=results["Proposed (IRL)"],
        save_path=output_dir / "comparison_curves.png",
        baselines=baselines
    )

    print(f"\nComparison plot saved to: {output_dir / 'comparison_curves.png'}")

if __name__ == "__main__":
    main()