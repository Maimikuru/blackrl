"""Plot learning curves for Bilevel RL training."""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(stats, save_path=None):
    """Plot comprehensive learning curves from training statistics.

    Args:
        stats: Dictionary of training statistics from algo.train()
        save_path: Optional path to save the figure

    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Bilevel RL Training Curves", fontsize=16, fontweight="bold")

    # Plot 1: Leader Objective
    ax = axes[0, 0]
    if "leader_objective" in stats and len(stats["leader_objective"]) > 0:
        iterations = range(len(stats["leader_objective"]))
        ax.plot(iterations, stats["leader_objective"], "b-", linewidth=2, label="Leader Objective")
        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.set_title("Leader Objective", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Leader Objective")

    # Plot 2: IRL Gradient Norm
    ax = axes[0, 1]
    if "irl_gradient_norm" in stats and len(stats["irl_gradient_norm"]) > 0:
        ax.plot(stats["irl_gradient_norm"], "r-", linewidth=2, alpha=0.7, label="Gradient Norm")
        ax.axhline(y=0.025, color="g", linestyle="--", linewidth=2, label="Tolerance (0.025)")
        ax.set_xlabel("IRL Iteration", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title("MDCE IRL Gradient Norm", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("MDCE IRL Gradient Norm")

    # Plot 3: IRL Likelihood
    ax = axes[0, 2]
    if "irl_likelihood" in stats and len(stats["irl_likelihood"]) > 0:
        # Extract iterations and likelihoods
        irl_iters, likelihoods = zip(*stats["irl_likelihood"], strict=False)
        ax.plot(irl_iters, likelihoods, "g-", linewidth=2, marker="o", markersize=6, label="Likelihood")
        ax.set_xlabel("IRL Iteration", fontsize=12)
        ax.set_ylabel("Log Likelihood", fontsize=12)
        ax.set_title("MDCE IRL Likelihood", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("MDCE IRL Likelihood")

    # Plot 4: Leader Gradient Norm
    ax = axes[1, 0]
    if "leader_gradient_norm" in stats and len(stats["leader_gradient_norm"]) > 0:
        iterations = range(len(stats["leader_gradient_norm"]))
        ax.plot(iterations, stats["leader_gradient_norm"], "m-", linewidth=2, label="Leader Gradient")
        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title("Leader Policy Gradient Norm", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Leader Policy Gradient Norm")

    # Plot 5: Leader Mean Q-value
    ax = axes[1, 1]
    if "leader_mean_q_value" in stats and len(stats["leader_mean_q_value"]) > 0:
        iterations = range(len(stats["leader_mean_q_value"]))
        ax.plot(iterations, stats["leader_mean_q_value"], "c-", linewidth=2, label="Mean Q-value")
        ax.set_xlabel("Leader Iteration", fontsize=12)
        ax.set_ylabel("Q-value", fontsize=12)
        ax.set_title("Leader Mean Q-value", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Leader Mean Q-value")

    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    ax.axis("off")

    # Collect summary statistics
    summary_text = "Training Summary\n" + "=" * 40 + "\n\n"

    if "leader_objective" in stats and len(stats["leader_objective"]) > 0:
        final_obj = stats["leader_objective"][-1]
        initial_obj = stats["leader_objective"][0]
        improvement = final_obj - initial_obj
        summary_text += "Leader Objective:\n"
        summary_text += f"  Initial: {initial_obj:.4f}\n"
        summary_text += f"  Final: {final_obj:.4f}\n"
        summary_text += f"  Improvement: {improvement:+.4f}\n\n"

    if "irl_gradient_norm" in stats and len(stats["irl_gradient_norm"]) > 0:
        final_grad = stats["irl_gradient_norm"][-1]
        converged = "Yes" if final_grad < 0.025 else "No"
        summary_text += "MDCE IRL:\n"
        summary_text += f"  Final Gradient: {final_grad:.6f}\n"
        summary_text += f"  Converged: {converged}\n\n"

    if "leader_gradient_norm" in stats and len(stats["leader_gradient_norm"]) > 0:
        avg_grad = np.mean(stats["leader_gradient_norm"])
        summary_text += "Leader Policy:\n"
        summary_text += f"  Avg Gradient: {avg_grad:.4f}\n\n"

    summary_text += f"Total Leader Iterations: {len(stats.get('leader_objective', []))}\n"

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    import pickle
    import sys

    if len(sys.argv) > 1:
        # Load stats from pickle file
        stats_file = sys.argv[1]
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)

        save_path = stats_file.replace(".pkl", "_curves.png")
        plot_learning_curves(stats, save_path=save_path)
        plt.show()
    else:
        print("Usage: python plot_learning_curves.py <stats_file.pkl>")
        print("Or import and use plot_learning_curves(stats) in your script")
