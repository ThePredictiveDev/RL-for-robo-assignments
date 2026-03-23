"""
Visualization and experiment script for RL Drone Hover Assignment.

Trains Monte Carlo and Q-Learning agents, runs hyperparameter sweeps,
and generates publication-quality plots saved to the outputs/ directory.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from user_code import (
    run_monte_carlo,
    run_q_learning,
    discretize_state,
    extract_position,
    format_action,
    evaluate_policy,
    initialize_q_table,
    choose_action,
    NUM_BINS,
    MAX_STEPS,
)
from bonus_challenges import (
    run_sarsa,
    run_double_q_learning,
    run_td_with_replay,
    evaluate_policy as bonus_evaluate_policy,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


def smooth(values, window=25):
    """Compute a centred moving average with given window size."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def make_env():
    return HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)


# ------------------------------------------------------------------
# 1. Train baseline MC and Q-Learning
# ------------------------------------------------------------------

def train_baselines(num_episodes=500):
    np.random.seed(SEED)
    env = make_env()

    print("=" * 60)
    print("Training Monte Carlo (baseline) ...")
    print("=" * 60)
    t0 = time.time()
    q_mc, rewards_mc = run_monte_carlo(env, num_episodes=num_episodes)
    mc_time = time.time() - t0

    print()
    print("=" * 60)
    print("Training Q-Learning (baseline) ...")
    print("=" * 60)
    t0 = time.time()
    q_ql, rewards_ql = run_q_learning(env, num_episodes=num_episodes)
    ql_time = time.time() - t0

    mean_mc, std_mc = evaluate_policy(env, q_mc)
    mean_ql, std_ql = evaluate_policy(env, q_ql)
    env.close()

    return {
        "mc": {"q": q_mc, "rewards": rewards_mc, "mean": mean_mc, "std": std_mc, "time": mc_time},
        "ql": {"q": q_ql, "rewards": rewards_ql, "mean": mean_ql, "std": std_ql, "time": ql_time},
    }


# ------------------------------------------------------------------
# 2. Learning curves plot
# ------------------------------------------------------------------

def plot_learning_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (key, label, colour) in zip(
        axes, [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]
    ):
        raw = results[key]["rewards"]
        smoothed = smooth(raw, window=25)
        ax.plot(raw, alpha=0.25, color=colour, linewidth=0.6)
        ax.plot(
            np.arange(len(smoothed)) + 12,
            smoothed,
            color=colour,
            linewidth=2,
            label=f"{label} (smoothed)",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"{label} Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("[saved] learning_curves.png")


# ------------------------------------------------------------------
# 3. Comparison overlay plot
# ------------------------------------------------------------------

def plot_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, label, colour in [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]:
        raw = results[key]["rewards"]
        smoothed = smooth(raw, window=25)
        ax.plot(raw, alpha=0.15, color=colour, linewidth=0.5)
        ax.plot(
            np.arange(len(smoothed)) + 12,
            smoothed,
            color=colour,
            linewidth=2,
            label=f"{label} (smoothed)",
        )

    ax.axhline(y=220, color="red", linestyle="--", linewidth=1, label="Pass threshold (220)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Monte Carlo vs Q-Learning -- Training Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "mc_vs_ql_comparison.png"), dpi=150)
    plt.close(fig)
    print("[saved] mc_vs_ql_comparison.png")


# ------------------------------------------------------------------
# 4. Convergence analysis
# ------------------------------------------------------------------

def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, label, colour in [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]:
        raw = results[key]["rewards"]
        rolling = smooth(raw, window=50)
        episodes = np.arange(len(rolling)) + 25
        ax.plot(episodes, rolling, color=colour, linewidth=2, label=label)

    ax.axhline(y=220, color="red", linestyle="--", linewidth=1, label="Pass threshold (220)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Average Reward (window=50)")
    ax.set_title("Convergence Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence_analysis.png"), dpi=150)
    plt.close(fig)
    print("[saved] convergence_analysis.png")


# ------------------------------------------------------------------
# 5. Q-table heatmap (z-slice at target height bin)
# ------------------------------------------------------------------

def plot_q_heatmap(results):
    target_z_bin = NUM_BINS // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (key, label) in zip(axes, [("mc", "Monte Carlo"), ("ql", "Q-Learning")]):
        q_table = results[key]["q"]
        max_q = np.max(q_table[:, :, target_z_bin, :], axis=-1)
        im = ax.imshow(max_q, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xlabel("Y bin")
        ax.set_ylabel("X bin")
        ax.set_title(f"{label} -- max Q(x, y, z={target_z_bin})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "q_table_heatmap.png"), dpi=150)
    plt.close(fig)
    print("[saved] q_table_heatmap.png")


# ------------------------------------------------------------------
# 6. Hyperparameter sensitivity sweep
# ------------------------------------------------------------------

def hyperparam_sweep():
    """Run short training runs varying one hyperparameter at a time."""
    sweep_episodes = 300

    sweep_cfg = {
        "epsilon": {"values": [0.05, 0.1, 0.2, 0.3], "defaults": {"gamma": 0.99, "alpha": 0.1}},
        "gamma":   {"values": [0.95, 0.97, 0.99, 0.999], "defaults": {"epsilon": 0.1, "alpha": 0.1}},
        "alpha":   {"values": [0.05, 0.1, 0.15, 0.2], "defaults": {"epsilon": 0.1, "gamma": 0.99}},
    }

    all_results = {}

    for param_name, cfg in sweep_cfg.items():
        print(f"\n--- Sweep: {param_name} ---")
        all_results[param_name] = {}

        for val in cfg["values"]:
            kwargs = dict(cfg["defaults"])
            kwargs[param_name] = val
            kwargs["num_episodes"] = sweep_episodes
            label = f"{param_name}={val}"

            np.random.seed(SEED)
            env = make_env()
            _, rewards = run_q_learning(env, **kwargs)
            env.close()

            all_results[param_name][val] = rewards
            final_avg = np.mean(rewards[-50:])
            print(f"  {label} -> final avg(last 50): {final_avg:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (param_name, param_results) in zip(axes, all_results.items()):
        for val, rewards in param_results.items():
            smoothed = smooth(rewards, window=25)
            ax.plot(
                np.arange(len(smoothed)) + 12,
                smoothed,
                linewidth=1.8,
                label=f"{param_name}={val}",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (smoothed)")
        ax.set_title(f"Sensitivity: {param_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "hyperparam_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("[saved] hyperparam_sensitivity.png")

    return all_results


# ------------------------------------------------------------------
# 7. NUM_BINS sensitivity
# ------------------------------------------------------------------

def bins_sweep():
    """Sweep over different NUM_BINS values."""
    sweep_episodes = 300
    bins_values = [8, 10, 12, 15]
    results_bins = {}

    print("\n--- Sweep: NUM_BINS ---")

    for nb in bins_values:
        np.random.seed(SEED)
        env = make_env()

        q_table = np.zeros((nb,) * 3 + (3,))
        episode_rewards = []
        epsilon_min = 0.01

        for episode in range(sweep_episodes):
            current_epsilon = max(epsilon_min, 0.1 * (1.0 - episode / sweep_episodes))
            obs, _ = env.reset()
            pos = extract_position(obs)
            state = _disc(pos, nb)
            total_reward = 0.0

            for _ in range(MAX_STEPS):
                if np.random.random() < current_epsilon:
                    action = np.random.randint(3)
                else:
                    action = np.argmax(q_table[state])
                next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
                next_state = _disc(extract_position(next_obs), nb)

                td_target = reward + 0.99 * np.max(q_table[next_state])
                q_table[state][action] += 0.1 * (td_target - q_table[state][action])

                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)

        results_bins[nb] = episode_rewards
        env.close()
        print(f"  NUM_BINS={nb} -> final avg(last 50): {np.mean(episode_rewards[-50:]):.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for nb, rewards in results_bins.items():
        smoothed = smooth(rewards, window=25)
        ax.plot(np.arange(len(smoothed)) + 12, smoothed, linewidth=1.8, label=f"bins={nb}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Sensitivity: NUM_BINS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bins_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("[saved] bins_sensitivity.png")


def _disc(pos, num_bins):
    """Lightweight discretization for bins sweep."""
    bounds = np.array([[-1, 1], [-1, 1], [0, 2]])
    discrete = []
    for val, (low, high) in zip(pos, bounds):
        val = np.clip(val, low, high)
        normalized = (val - low) / (high - low)
        bin_idx = min(int(normalized * num_bins), num_bins - 1)
        discrete.append(bin_idx)
    return tuple(discrete)


# ------------------------------------------------------------------
# 8. Final performance bar chart
# ------------------------------------------------------------------

def plot_final_bar(results):
    methods = ["Monte Carlo", "Q-Learning"]
    means = [results["mc"]["mean"], results["ql"]["mean"]]
    stds = [results["mc"]["std"], results["ql"]["std"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=["tab:blue", "tab:orange"], edgecolor="black")
    ax.axhline(y=220, color="red", linestyle="--", linewidth=1, label="Pass threshold (220)")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Final Policy Performance (greedy, 10 episodes)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{m:.1f}",
                ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "final_performance_bar.png"), dpi=150)
    plt.close(fig)
    print("[saved] final_performance_bar.png")


# ------------------------------------------------------------------
# 9. Drone trajectory visualization (hover render)
# ------------------------------------------------------------------

def plot_drone_trajectory(results):
    """Run a greedy evaluation episode and plot the drone's 3D position over time."""
    env = make_env()

    for key, label, colour in [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]:
        q_table = results[key]["q"]
        obs, _ = env.reset()
        positions = [extract_position(obs).copy()]

        for _ in range(MAX_STEPS):
            state = discretize_state(extract_position(obs))
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, _ = env.step(format_action(action))
            positions.append(extract_position(obs).copy())
            if terminated or truncated:
                break

        results[key]["trajectory"] = np.array(positions)

    env.close()

    timesteps_sec = np.arange(results["ql"]["trajectory"].shape[0]) / 30.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axis_labels = ["X position (m)", "Y position (m)", "Z position (m)"]
    targets = [0.0, 0.0, 1.0]

    for row, (ax, ylabel, target_val) in enumerate(zip(axes, axis_labels, targets)):
        for key, label, colour in [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]:
            traj = results[key]["trajectory"]
            t = np.arange(traj.shape[0]) / 30.0
            ax.plot(t, traj[:, row], color=colour, linewidth=1.5, label=label)
        ax.axhline(y=target_val, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Target")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    axes[0].set_title("Drone Hover Trajectory -- Learned Policy Evaluation")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "drone_trajectory.png"), dpi=150)
    plt.close(fig)
    print("[saved] drone_trajectory.png")

    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    for key, label, colour in [("mc", "Monte Carlo", "tab:blue"), ("ql", "Q-Learning", "tab:orange")]:
        traj = results[key]["trajectory"]
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colour, linewidth=1.5, label=label)
        ax3d.scatter(*traj[0], color=colour, marker="o", s=60, zorder=5)
        ax3d.scatter(*traj[-1], color=colour, marker="x", s=80, zorder=5)
    ax3d.scatter(0, 0, 1, color="red", marker="*", s=200, zorder=10, label="Target [0,0,1]")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Drone Trajectory -- Policy Evaluation")
    ax3d.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "drone_trajectory_3d.png"), dpi=150)
    plt.close(fig)
    print("[saved] drone_trajectory_3d.png")


# ------------------------------------------------------------------
# 10. Bonus challenges training and bar chart
# ------------------------------------------------------------------

def run_bonus_and_plot():
    """Train bonus algorithms and produce a combined performance bar chart."""
    bonus_results = {}

    print("\n" + "=" * 60)
    print("Training Bonus Algorithms ...")
    print("=" * 60)

    np.random.seed(SEED)
    env = make_env()
    t0 = time.time()
    q_sarsa, rew_sarsa = run_sarsa(env, num_episodes=500)
    sarsa_time = time.time() - t0
    mean_sarsa, std_sarsa = evaluate_policy(env, q_sarsa)
    env.close()
    bonus_results["SARSA"] = {"mean": mean_sarsa, "std": std_sarsa, "rewards": rew_sarsa, "time": sarsa_time}
    print(f"SARSA Evaluation: {mean_sarsa:.2f} +/- {std_sarsa:.2f}")

    np.random.seed(SEED)
    env = make_env()
    t0 = time.time()
    q1, q2, rew_dql = run_double_q_learning(env, num_episodes=500)
    dql_time = time.time() - t0
    q_combined = (q1 + q2) / 2.0
    mean_dql, std_dql = evaluate_policy(env, q_combined)
    env.close()
    bonus_results["Double QL"] = {"mean": mean_dql, "std": std_dql, "rewards": rew_dql, "time": dql_time}
    print(f"Double Q-Learning Evaluation: {mean_dql:.2f} +/- {std_dql:.2f}")

    np.random.seed(SEED)
    env = make_env()
    t0 = time.time()
    q_replay, rew_replay = run_td_with_replay(env, num_episodes=500)
    replay_time = time.time() - t0
    mean_replay, std_replay = evaluate_policy(env, q_replay)
    env.close()
    bonus_results["Exp. Replay"] = {"mean": mean_replay, "std": std_replay, "rewards": rew_replay, "time": replay_time}
    print(f"Experience Replay Evaluation: {mean_replay:.2f} +/- {std_replay:.2f}")

    return bonus_results


def plot_all_methods_bar(baseline_results, bonus_results):
    """Bar chart comparing all 5 methods."""
    names = ["Monte Carlo", "Q-Learning", "SARSA", "Double QL", "Exp. Replay"]
    means = [
        baseline_results["mc"]["mean"],
        baseline_results["ql"]["mean"],
        bonus_results["SARSA"]["mean"],
        bonus_results["Double QL"]["mean"],
        bonus_results["Exp. Replay"]["mean"],
    ]
    stds = [
        baseline_results["mc"]["std"],
        baseline_results["ql"]["std"],
        bonus_results["SARSA"]["std"],
        bonus_results["Double QL"]["std"],
        bonus_results["Exp. Replay"]["std"],
    ]
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colours, edgecolor="black")
    ax.axhline(y=300, color="gray", linestyle="--", linewidth=1, label="Bonus threshold (300)")
    ax.axhline(y=220, color="red", linestyle="--", linewidth=1, label="Pass threshold (220)")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("All Algorithms -- Final Policy Performance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{m:.1f}",
                ha="center", va="bottom", fontweight="bold", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "all_methods_comparison.png"), dpi=150)
    plt.close(fig)
    print("[saved] all_methods_comparison.png")


def plot_bonus_learning_curves(bonus_results):
    """Learning curves for the three bonus algorithms."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, colour in [("SARSA", "tab:green"), ("Double QL", "tab:purple"), ("Exp. Replay", "tab:red")]:
        raw = bonus_results[label]["rewards"]
        smoothed = smooth(raw, window=25)
        ax.plot(raw, alpha=0.15, color=colour, linewidth=0.5)
        ax.plot(np.arange(len(smoothed)) + 12, smoothed, color=colour, linewidth=2, label=f"{label} (smoothed)")

    ax.axhline(y=300, color="gray", linestyle="--", linewidth=1, label="Bonus threshold (300)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Bonus Algorithms -- Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bonus_learning_curves.png"), dpi=150)
    plt.close(fig)
    print("[saved] bonus_learning_curves.png")


# ------------------------------------------------------------------
# 11. Write text summary
# ------------------------------------------------------------------

def write_summary(results, bonus_results=None):
    path = os.path.join(OUTPUT_DIR, "training_summary.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("RL DRONE HOVER -- TRAINING SUMMARY")
    lines.append("=" * 60)

    for key, label in [("mc", "Monte Carlo"), ("ql", "Q-Learning")]:
        r = results[key]
        lines.append(f"\n--- {label} ---")
        lines.append(f"  Evaluation reward : {r['mean']:.2f} +/- {r['std']:.2f}")
        lines.append(f"  Last-50 avg reward: {np.mean(r['rewards'][-50:]):.2f}")
        lines.append(f"  Training time     : {r['time']:.1f}s")
        lines.append(f"  Total episodes    : {len(r['rewards'])}")
        passed = "PASS" if r["mean"] >= 220 else "FAIL"
        lines.append(f"  Status            : {passed} (threshold 220)")

    if results["mc"]["mean"] > results["ql"]["mean"]:
        lines.append("\nBetter algorithm: Monte Carlo")
    elif results["ql"]["mean"] > results["mc"]["mean"]:
        lines.append("\nBetter algorithm: Q-Learning")
    else:
        lines.append("\nBoth algorithms performed equally.")

    if bonus_results:
        lines.append("\n" + "-" * 60)
        lines.append("BONUS CHALLENGES")
        lines.append("-" * 60)
        total_bonus = 0
        for label, pts in [("SARSA", 5), ("Double QL", 7), ("Exp. Replay", 8)]:
            r = bonus_results[label]
            earned = pts if r["mean"] >= 300 else 0
            total_bonus += earned
            status = "PASS" if r["mean"] >= 300 else "FAIL"
            lines.append(f"\n--- {label} ({pts} pts) ---")
            lines.append(f"  Evaluation reward : {r['mean']:.2f} +/- {r['std']:.2f}")
            lines.append(f"  Last-50 avg reward: {np.mean(r['rewards'][-50:]):.2f}")
            lines.append(f"  Training time     : {r['time']:.1f}s")
            lines.append(f"  Status            : {status} (threshold 300)")
            lines.append(f"  Points earned     : {earned}/{pts}")
        lines.append(f"\nTotal bonus points: {total_bonus}/20")

    lines.append("\n" + "-" * 60)
    lines.append("Hyperparameters used (baseline):")
    lines.append("  NUM_BINS  = 10")
    lines.append("  EPSILON   = 0.1 (exponential decay to 0.01)")
    lines.append("  GAMMA     = 0.99")
    lines.append("  ALPHA     = 0.1")
    lines.append("  EPISODES  = 500")
    lines.append("  MAX_STEPS = 240")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("[saved] training_summary.txt")
    print(text)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GENERATING ALL PLOTS AND ANALYSIS")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    results = train_baselines(num_episodes=500)

    plot_learning_curves(results)
    plot_comparison(results)
    plot_convergence(results)
    plot_q_heatmap(results)
    plot_final_bar(results)
    plot_drone_trajectory(results)

    hyperparam_sweep()
    bins_sweep()

    bonus_results = run_bonus_and_plot()
    plot_all_methods_bar(results, bonus_results)
    plot_bonus_learning_curves(bonus_results)

    write_summary(results, bonus_results)

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"Check: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
