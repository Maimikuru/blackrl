"""Training script for Bi-level RL with Baselines (Parallel Ready)."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_learning_curves

COMMON_PARAMS = {
    "discount_leader": 0.99,
    "discount_follower": 0.80,
    "learning_rate_leader_actor": 1e-4,
    "learning_rate_leader_critic": 1e-3,
    "learning_rate_follower": 0.01,
    "mdce_irl_config": {
        "max_iterations": 20,
        "tolerance": 0.01,
        "n_soft_q_iterations": 1000,
    },
    "soft_q_config": {
        "learning_rate": 0.1,
        "temperature": 1.0,
    },
}

TRAIN_PARAMS = {
    "n_leader_iterations": 10,
    "n_episodes_per_iteration": 1000,
    "mdce_irl_frequency": 10,
    "verbose": True,
}


def create_simple_leader_policy():
    return np.random.choice([0, 1], p=[0.5, 0.5])


def feature_fn(state, leader_action, follower_action):
    num_states, num_leader_actions, num_follower_actions = 3, 2, 3
    total_dim = num_states * num_leader_actions * num_follower_actions
    s = int(state.item() if hasattr(state, "item") else state)
    a = int(leader_action.item() if hasattr(leader_action, "item") else leader_action)
    b = int(follower_action.item() if hasattr(follower_action, "item") else follower_action)
    index = s * (num_leader_actions * num_follower_actions) + a * num_follower_actions + b
    feature = np.zeros(total_dim, dtype=np.float32)
    feature[index] = 1.0
    return feature


def run_experiment(mode, output_dir):
    """指定されたモードの実験を1つだけ実行して保存する"""
    print(f"\n=== Starting Experiment: {mode} ===")

    env = DiscreteToyEnvPaper()

    # アルゴリズム初期化
    algo = BilevelRL(
        env_spec=env.spec,
        leader_policy=create_simple_leader_policy(),
        feature_fn=feature_fn,
        **COMMON_PARAMS,
    )

    # 学習実行
    oracle_arg = "none" if mode == "irl" or mode == "irl_no_second_term" else mode
    use_second_term = mode != "irl_no_second_term"

    stats = algo.train(env=env, oracle_mode=oracle_arg, use_second_term=use_second_term, **TRAIN_PARAMS)

    # 結果保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"stats_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"=== Finished {mode}. Saved to {save_path} ===")


def merge_and_plot(output_dir):
    """保存された結果を読み込んでプロットする"""
    output_dir = Path(output_dir)
    print(f"\n=== Merging Results and Plotting (from {output_dir}) ===")
    results = {}

    # 読み込みマッピング
    modes = {
        "irl": "Proposed (IRL)",
        "irl_no_second_term": "Proposed (IRL, no 2nd term)",
        "softql": "Oracle (SoftQL)",
        "softvi": "Oracle (SoftVI)",
    }

    for mode_key, mode_name in modes.items():
        path = output_dir / f"stats_{mode_key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                results[mode_name] = pickle.load(f)
            print(f"Loaded: {mode_name}")
        else:
            print(f"Warning: Result file for {mode_name} not found at {path}")

    if not results:
        print("No results found to plot.")
        return

    # プロット
    # Proposed (IRL) がある前提で、それをメインにし、他をベースラインにする
    main_stats = results.get("Proposed (IRL)", {})
    baselines = {k: v for k, v in results.items() if k != "Proposed (IRL)"}

    save_path = output_dir / "comparison_curves.pdf"
    plot_learning_curves(main_stats, save_path=save_path, baselines=baselines)
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["irl", "irl_no_second_term", "softql", "softvi", "plot"], required=True)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for stats and plots")
    args = parser.parse_args()

    if args.mode == "plot":
        merge_and_plot(args.output_dir)
    else:
        run_experiment(args.mode, args.output_dir)


if __name__ == "__main__":
    main()
