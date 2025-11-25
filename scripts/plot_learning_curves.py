import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_learning_curves(stats, save_path=None):
    """Plot comprehensive learning curves.

    Plots 6 metrics:
    1. Leader Return (Undiscounted) - The raw score (performance)
    2. IRL Delta FEM - Convergence of feature matching
    3. IRL Likelihood - How well the reward explains behavior
    4. Leader Gradient Norm - Stability of leader's update
    5. Leader Objective (Discounted) - The optimization target
    6. Leader Q-Values - Value estimation statistics
    """

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Flatten axes for easier indexing
    # ax1, ax2, ax3
    # ax4, ax5, ax6

    # --- Plot 1: Leader Return (Undiscounted) [MAIN METRIC] ---
    ax = axes[0, 0]
    if "leader_return" in stats and len(stats["leader_return"]) > 0:
        iterations = range(len(stats["leader_return"]))
        ax.plot(iterations, stats["leader_return"], "b-", linewidth=2, label="Undiscounted Return")

        # Moving average (window=10)
        if len(stats["leader_return"]) > 20:
            window = 10
            ma = np.convolve(stats["leader_return"], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(stats["leader_return"])), ma, "r--", linewidth=1.5, alpha=0.8, label="Moving Avg (10)")

        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Total Reward per Episode", fontsize=12)
        ax.set_title("Leader Average Return (Raw Score)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No Return Data\n(Update bilevel_rl.py)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Leader Average Return", fontsize=14, fontweight="bold")

    # --- Plot 2: IRL Delta FEM ---
    ax = axes[0, 1]
    if "irl_delta_fem" in stats and len(stats["irl_delta_fem"]) > 0:
        # IRL runs less frequently, so create x-axis
        # Assuming logged every iteration or less
        # Flatten if nested list
        deltas = []
        if isinstance(stats["irl_delta_fem"][0], list):
            for sublist in stats["irl_delta_fem"]:
                deltas.extend(sublist)
        else:
            deltas = stats["irl_delta_fem"]

        ax.plot(deltas, "g-", linewidth=1.5, label="Delta FEM")
        ax.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="Threshold (0.1)")
        ax.axhline(y=0.025, color="m", linestyle=":", alpha=0.5, label="Target (0.025)")

        ax.set_yscale("log")
        ax.set_xlabel("IRL Steps (Cumulative)", fontsize=12)
        ax.set_ylabel("Max Relative Error (log scale)", fontsize=12)
        ax.set_title("IRL Feature Matching Error", fontsize=14, fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No IRL Data", ha="center", va="center", transform=ax.transAxes)

    # --- Plot 3: IRL Likelihood ---
    ax = axes[0, 2]
    if "irl_likelihood" in stats and len(stats["irl_likelihood"]) > 0:
        # Extract values from (iter, val) tuples or list
        if isinstance(stats["irl_likelihood"][0], tuple):
            values = [x[1] for x in stats["irl_likelihood"]]
        else:
            values = stats["irl_likelihood"]

        ax.plot(values, "purple", linewidth=1.5)
        ax.set_xlabel("IRL Evaluation Steps", fontsize=12)
        ax.set_ylabel("Log Likelihood", fontsize=12)
        ax.set_title("IRL Log Likelihood", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Likelihood Data", ha="center", va="center", transform=ax.transAxes)

    # --- Plot 4: Leader Gradient Norm ---
    ax = axes[1, 0]
    if "leader_gradient_norm" in stats and len(stats["leader_gradient_norm"]) > 0:
        ax.plot(stats["leader_gradient_norm"], "orange", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
        ax.set_title("Leader Gradient Stability", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Gradient Data", ha="center", va="center", transform=ax.transAxes)

    # --- Plot 5: Leader Objective (Discounted) ---
    ax = axes[1, 1]
    if "leader_objective" in stats and len(stats["leader_objective"]) > 0:
        ax.plot(stats["leader_objective"], "c-", linewidth=2, label="Discounted J")
        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Objective Value (J)", fontsize=12)
        ax.set_title("Leader Objective (Discounted)", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Objective Data", ha="center", va="center", transform=ax.transAxes)

    # --- Plot 6: Summary Text ---
    ax = axes[1, 2]
    ax.axis("off")

    summary = "Training Summary\n"
    summary += "----------------\n\n"

    if "leader_return" in stats and len(stats["leader_return"]) > 0:
        final_ret = stats["leader_return"][-1]
        max_ret = np.max(stats["leader_return"])
        summary += f"Leader Return (Raw):\n"
        summary += f"  Final: {final_ret:.4f}\n"
        summary += f"  Max:   {max_ret:.4f}\n\n"

    if "leader_objective" in stats and len(stats["leader_objective"]) > 0:
        final_obj = stats["leader_objective"][-1]
        summary += f"Leader Objective (J):\n"
        summary += f"  Final: {final_obj:.4f}\n\n"

    if "irl_delta_fem" in stats and len(stats["irl_delta_fem"]) > 0:
        deltas = []
        if isinstance(stats["irl_delta_fem"][0], list):
            for sublist in stats["irl_delta_fem"]:
                deltas.extend(sublist)
        else:
            deltas = stats["irl_delta_fem"]

        if len(deltas) > 0:
            final_delta = deltas[-1]
            min_delta = np.min(deltas)
            summary += f"IRL Delta FEM:\n"
            summary += f"  Final: {final_delta:.6f}\n"
            summary += f"  Best:  {min_delta:.6f}\n"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=14, verticalalignment='top', family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")

    return fig

if __name__ == "__main__":
    # Test with dummy data if run directly
    dummy_stats = {
        "leader_return": np.random.rand(100) * 10 + 5,
        "leader_objective": np.random.rand(100) * 5 + 8,
        "leader_gradient_norm": np.random.rand(100) * 100,
        "irl_delta_fem": [np.linspace(0.5, 0.01, 100).tolist()],
        "irl_likelihood": np.linspace(-2, -0.5, 10),
    }
    plot_learning_curves(dummy_stats, "dummy_plot.png")