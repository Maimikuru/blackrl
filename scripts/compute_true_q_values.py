"""Compute true Q-values for the follower in the MDP.

This script computes the true optimal Q-values Q_F^*(s, a, b) for the follower
using Soft Value Iteration (SoftVI), assuming the follower knows its true reward function.

SoftVI uses the Soft Q-Learning Bellman equation with value iteration to find
the optimal Q-function for maximum entropy policies.
"""

import pickle

import numpy as np
import torch
from blackrl.envs import DiscreteToyEnvPaper


def compute_true_q_values(
    env,
    discount: float = 0.99,
    temperature: float = 1.0,
    max_iterations: int = 1000000,
    tolerance: float = 0,
    verbose: bool = True,
):
    """Compute true Q-values for the follower using Soft Value Iteration (SoftVI).

    This computes Q_F^*(s, a, b) assuming the follower knows its true reward function.
    Uses Soft Q-Learning Bellman equation (SoftVI):
        Q(s,a,b) = r(s,a,b) + γ * E_{a'}[V^soft(s',a')]
        V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b') / temperature)

    This is a Q-function based Soft Value Iteration algorithm.

    Args:
        env: Environment instance
        discount: Discount factor γ
        temperature: Temperature parameter for soft Q-learning
        max_iterations: Maximum iterations for value iteration
        tolerance: Convergence tolerance
        verbose: Whether to print progress

    Returns:
        Dictionary mapping (s, a, b) -> Q_F^*(s, a, b)

    """
    num_states = env.spec.observation_space.n
    num_leader_actions = env.spec.leader_action_space.n
    num_follower_actions = env.spec.action_space.n

    # Initialize Q-values
    Q_true = {}
    for s in range(num_states):
        for a in range(num_leader_actions):
            for b in range(num_follower_actions):
                Q_true[(s, a, b)] = 0.0

    # Value iteration
    for iteration in range(max_iterations):
        Q_old = Q_true.copy()
        max_delta = 0.0

        for s in range(num_states):
            for a in range(num_leader_actions):
                for b in range(num_follower_actions):
                    # Get reward and next state from environment
                    # Set initial state
                    env.reset(init_state=s)
                    env_step = env.step(a, b)

                    reward = env_step.reward
                    next_state = env_step.observation

                    if env_step.last:
                        # Terminal state
                        Q_true[(s, a, b)] = reward
                    else:
                        # SoftVI Bellman update for Soft Q-Learning
                        next_state_int = int(next_state.item() if isinstance(next_state, np.ndarray) else next_state)

                        # Assume uniform leader policy for next state value
                        v_next = 0.0
                        for a_next in range(num_leader_actions):
                            # Compute soft value: V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b')/temperature)
                            q_values = [
                                Q_old.get((next_state_int, a_next, b_next), 0.0) for b_next in range(num_follower_actions)
                            ]
                            # Convert to torch for logsumexp
                            q_tensor = torch.tensor(q_values)
                            soft_v = temperature * torch.logsumexp(q_tensor / temperature, dim=0).item()
                            v_next += soft_v / num_leader_actions

                        Q_true[(s, a, b)] = reward + discount * v_next

                    # Track convergence
                    delta = abs(Q_true[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                    max_delta = max(max_delta, delta)

        if verbose and (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration + 1}: max_delta = {max_delta:.6f}")

        if max_delta < tolerance:
            if verbose:
                print(f"\n✓ True Q-values converged in {iteration + 1} iterations (max_delta={max_delta:.6f})")
            break

    if iteration == max_iterations - 1:
        if verbose:
            if max_delta < 1e-4:
                print(
                    f"\n✓ True Q-values nearly converged after {max_iterations} iterations (max_delta={max_delta:.6f} < 1e-4)",
                )
            else:
                print(
                    f"\n⚠ WARNING: True Q-values did NOT converge after {max_iterations} iterations (max_delta={max_delta:.6f})",
                )

    return Q_true


def display_q_values(Q_true, env):
    """Display Q-values in a readable format.

    Args:
        Q_true: Dictionary mapping (s, a, b) -> Q-value
        env: Environment instance

    """
    num_states = env.spec.observation_space.n
    num_leader_actions = env.spec.leader_action_space.n
    num_follower_actions = env.spec.action_space.n

    print("\n" + "=" * 80)
    print("TRUE Q-VALUES: Q_F^*(s, a, b)")
    print("=" * 80)

    for s in range(num_states):
        print(f"\nState {s}:")
        for a in range(num_leader_actions):
            q_str = "  ".join([f"Q({s},{a},{b})={Q_true.get((s, a, b), 0.0):8.4f}" for b in range(num_follower_actions)])
            print(f"  Leader a={a}: {q_str}")

    # Statistics
    q_values_list = list(Q_true.values())
    print("\n" + "-" * 80)
    print("Q-value Statistics:")
    print(f"  Min:  {np.min(q_values_list):8.4f}")
    print(f"  Max:  {np.max(q_values_list):8.4f}")
    print(f"  Mean: {np.mean(q_values_list):8.4f}")
    print(f"  Std:  {np.std(q_values_list):8.4f}")
    print("=" * 80)


def main():
    """Main function."""
    print("Computing true Q-values for follower...")
    print("-" * 80)

    # Create environment
    env = DiscreteToyEnvPaper()
    print(f"Environment: {env.__class__.__name__}")
    print(f"  States: {env.spec.observation_space.n}")
    print(f"  Leader actions: {env.spec.leader_action_space.n}")
    print(f"  Follower actions: {env.spec.action_space.n}")
    print("  Discount: 0.8")
    print("  Temperature: 1.0")
    print("-" * 80)

    # Compute true Q-values
    Q_true = compute_true_q_values(
        env,
        discount=0.8,
        temperature=1.0,
        max_iterations=100000000000,
        tolerance=1e-15,
        verbose=True,
    )

    # Display results
    display_q_values(Q_true, env)

    # Save to file
    output_file = "true_q_values.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(Q_true, f)
    print(f"\n✓ Saved Q-values to {output_file}")


if __name__ == "__main__":
    main()
