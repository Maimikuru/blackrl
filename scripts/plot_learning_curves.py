import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_learning_curves(stats, save_path=None):
    """Plot comprehensive learning curves with comparison."""

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- Plot 1: Leader Performance Gap (True vs Learned) [MAIN] ---
    ax = axes[0, 0]

    # 1. vs True Follower (Blue)
    if "leader_return_true" in stats:
        means = np.array(stats["leader_return_true"])
        iterations = range(len(means))
        ax.plot(iterations, means, "b-", linewidth=2, label="vs True Follower (SoftVI)")

        if "leader_return_true_std" in stats:
            stds = np.array(stats["leader_return_true_std"])
            ax.fill_between(iterations, means - stds, means + stds, color="b", alpha=0.15)

    # 2. vs Learned Follower (Orange)
    if "leader_return_learned" in stats:
        means = np.array(stats["leader_return_learned"])
        iterations = range(len(means))
        ax.plot(iterations, means, "orange", linewidth=2, linestyle="--", label="vs Learned Follower (IRL)")

        if "leader_return_learned_std" in stats:
            stds = np.array(stats["leader_return_learned_std"])
            ax.fill_between(iterations, means - stds, means + stds, color="orange", alpha=0.15)

    ax.set_xlabel("Leader Iteration", fontsize=12)
    ax.set_ylabel("Total Reward per Episode", fontsize=12)
    ax.set_title("Leader Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    # --- Plot 2: IRL Delta FEM ---
    ax = axes[0, 1]
    if "irl_delta_fem" in stats:
        deltas = []
        if len(stats["irl_delta_fem"]) > 0 and isinstance(stats["irl_delta_fem"][0], list):
             for sublist in stats["irl_delta_fem"]: deltas.extend(sublist)
        else:
             deltas = stats["irl_delta_fem"]

        ax.plot(deltas, "g-", linewidth=1.5, label="Delta FEM")
        ax.axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="Threshold (0.01)")
        ax.set_yscale("log")
        ax.set_xlabel("IRL Steps")
        ax.set_title("IRL Feature Matching Error", fontsize=14, fontweight="bold")
        ax.legend()

    # --- Plot 3: IRL Likelihood ---
    ax = axes[0, 2]
    if "irl_likelihood" in stats:
        vals = [x[1] for x in stats["irl_likelihood"]] if isinstance(stats["irl_likelihood"][0], tuple) else stats["irl_likelihood"]
        ax.plot(vals, "purple", linewidth=1.5)
        ax.set_title("IRL Log Likelihood", fontsize=14, fontweight="bold")

    # --- Plot 4: Leader Gradient Norm ---
    ax = axes[1, 0]
    if "leader_gradient_norm" in stats:
        ax.plot(stats["leader_gradient_norm"], "r-", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title("Leader Gradient Norm", fontsize=14, fontweight="bold")

    # --- Plot 5: Leader Q-Values ---
    ax = axes[1, 1]
    if "leader_mean_q_value" in stats:
        ax.plot(stats["leader_mean_q_value"], "c-", linewidth=2)
        ax.set_title("Leader Mean Q-Value", fontsize=14, fontweight="bold")

    # --- Plot 6: Summary ---
    ax = axes[1, 2]
    ax.axis("off")
    summary = "Training Summary\n----------------\n\n"

    if "leader_return_true" in stats:
        final_true = stats["leader_return_true"][-1]
        final_learned = stats["leader_return_learned"][-1]
        gap = final_true - final_learned

        summary += f"Final Performance:\n"
        summary += f"  vs True:    {final_true:.4f}\n"
        summary += f"  vs Learned: {final_learned:.4f}\n"
        summary += f"  Gap:        {gap:+.4f}\n\n"

        if abs(gap) < 1.0:
            summary += ">> SUCCESS: Gap is small.\n   IRL is working well."
        else:
            summary += ">> WARNING: Large Gap.\n   Leader is exploiting\n   model errors."

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=13, verticalalignment='top', family='monospace')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")
    return fig

if __name__ == "__main__":
    # Dummy test
    stats = {
        "leader_return_true": np.linspace(5, 18, 100) + np.random.randn(100),
        "leader_return_true_std": np.ones(100) * 2,
        "leader_return_learned": np.linspace(5, 20, 100) + np.random.randn(100), # 少し過学習気味を想定
        "leader_return_learned_std": np.ones(100) * 2,
        "leader_gradient_norm": np.random.rand(100),
        "irl_delta_fem": [np.linspace(0.5, 0.01, 100).tolist()],
        "irl_likelihood": np.linspace(-2, -1, 10),
        "leader_mean_q_value": np.linspace(0, 5, 100)
    }
    plot_learning_curves(stats, "test_curves.png")