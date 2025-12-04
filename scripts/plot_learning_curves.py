import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(stats, save_path=None, baselines=None):
    """Plot comprehensive learning curves with comparison.

    Args:
        stats: Statistics dictionary for the main method (Proposed IRL)
        save_path: Path to save the plot
        baselines: Dictionary of baseline stats {name: stats_dict}

    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- Plot 1: Leader Performance Comparison [MAIN] ---
    ax = axes[0, 0]

    # 1. Main Method (Proposed IRL)
    # 私たちが一番見せたいのは「真のフォロワーに対する性能 (vs True)」です
    if "leader_return_true" in stats:
        means = np.array(stats["leader_return_true"])
        iterations = range(len(means))
        # 提案手法は目立つ色（赤やオレンジなど）で実線
        ax.plot(iterations, means, "r-", linewidth=2.5, label="Proposed (IRL)")

        if "leader_return_true_std" in stats:
            stds = np.array(stats["leader_return_true_std"])
            ax.fill_between(iterations, means - stds, means + stds, color="r", alpha=0.15)

    # 2. Baselines (Oracle SoftVI, SoftQL etc.)
    if baselines:
        colors = ["b", "g", "c", "m", "y"]  # ベースライン用の色
        for i, (name, base_stats) in enumerate(baselines.items()):
            color = colors[i % len(colors)]

            # ベースラインも「真の評価値」を使います
            # (キーが存在しない場合は leader_return をフォールバックとして使用)
            key = "leader_return_true" if "leader_return_true" in base_stats else "leader_return"

            if key in base_stats:
                means = np.array(base_stats[key])
                iterations = range(len(means))
                # ベースラインは少し細めの線や破線で
                ax.plot(iterations, means, color=color, linestyle="--", linewidth=2, label=name)

                # stdがあれば描画
                std_key = f"{key}_std"
                if std_key in base_stats:
                    stds = np.array(base_stats[std_key])
                    ax.fill_between(iterations, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel("Leader Iteration", fontsize=12)
    ax.set_ylabel("Total Reward (vs True Follower)", fontsize=12)
    ax.set_title("Leader Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    # --- Plot 2: IRL Delta FEM (Main Method only) ---
    ax = axes[0, 1]
    if "irl_delta_fem" in stats:
        deltas = []
        if len(stats["irl_delta_fem"]) > 0 and isinstance(stats["irl_delta_fem"][0], list):
            for sublist in stats["irl_delta_fem"]:
                deltas.extend(sublist)
        else:
            deltas = stats["irl_delta_fem"]

        if len(deltas) > 0:
            ax.plot(deltas, "g-", linewidth=1.5, label="Delta FEM")
            ax.axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="Threshold (0.01)")
            ax.set_yscale("log")
            ax.set_xlabel("IRL Steps")
            ax.set_title("IRL Feature Matching Error", fontsize=14, fontweight="bold")
            ax.legend()

    # --- Plot 3: IRL Likelihood (Main Method only) ---
    ax = axes[0, 2]
    if "irl_likelihood" in stats:
        vals = (
            [x[1] for x in stats["irl_likelihood"]]
            if isinstance(stats["irl_likelihood"][0], tuple)
            else stats["irl_likelihood"]
        )
        if len(vals) > 0:
            ax.plot(vals, "purple", linewidth=1.5)
            ax.set_title("IRL Log Likelihood", fontsize=14, fontweight="bold")

    # --- Plot 4: Leader Gradient Norm (Main Method) ---
    ax = axes[1, 0]
    if "leader_gradient_norm" in stats:
        ax.plot(stats["leader_gradient_norm"], "r-", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title("Leader Gradient Norm (Proposed)", fontsize=14, fontweight="bold")

    # --- Plot 5: Leader Q-Values (Main Method) ---
    ax = axes[1, 1]
    if "leader_mean_q_value" in stats:
        ax.plot(stats["leader_mean_q_value"], "c-", linewidth=2)
        ax.set_title("Leader Mean Q-Value (Proposed)", fontsize=14, fontweight="bold")

    # --- Plot 6: Summary Text ---
    ax = axes[1, 2]
    ax.axis("off")
    summary = "Experiment Summary\n------------------\n\n"

    # Proposed
    if "leader_return_true" in stats:
        final_val = stats["leader_return_true"][-1]
        summary += f"Proposed (IRL): {final_val:.4f}\n"

    # Baselines
    if baselines:
        for name, base_stats in baselines.items():
            key = "leader_return_true" if "leader_return_true" in base_stats else "leader_return"
            if key in base_stats:
                final_val = base_stats[key][-1]
                summary += f"{name}: {final_val:.4f}\n"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=13, verticalalignment="top", family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")
    return fig
