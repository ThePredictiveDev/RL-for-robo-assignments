"""
main.py - Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage examples
--------------
# View the environment (biped + platform in GUI, no model needed):
    python main.py --mode view --task jump

# Train SAC for the timesteps set in utils.py:
    python main.py --mode train --algo sac --task jump

# Train SAC for a custom number of steps:
    python main.py --mode train --algo sac --task jump --timesteps 500000

# Evaluate the best saved checkpoint (10 episodes, headless):
    python main.py --mode test --algo sac --task jump

# Evaluate with GUI rendering:
    python main.py --mode test --algo sac --task jump --render --episodes 5

# Evaluate a specific model file:
    python main.py --mode test --algo sac --task jump \
        --model_path "models jump/sac_best/best_model"
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils import (
    BipedJumpEnv,
    RewardPlotCallback,
    TOTAL_TIMESTEPS,
    EVAL_FREQ,
    SAC_CONFIG,
    EVAL_EPISODES,
    ROBOT_MASS_KG,
)


# Task registry --------------------------------------------------------------
# Maps the --task CLI key to the gymnasium env class to instantiate.
TASK_ENV = {
    "jump": BipedJumpEnv,
}


# Algorithm registry ---------------------------------------------------------
# Maps the --algo CLI key to (algorithm_class, config_dict). SAC is the only
# algorithm required by the assignment; the registry pattern keeps DDPG/TD3
# trivially pluggable should they be needed.
ALGO_MAP = {
    "sac": (SAC, SAC_CONFIG),
}


# Default checkpoint location used by --mode test when --model_path omitted.
DEFAULT_BEST_MODEL = os.path.join("models jump", "sac_best", "best_model")


# Environment Preview --------------------------------------------------------
def view(task_key: str = "jump"):
    """Spawns the biped on the platform in GUI mode and steps the simulator
    with random actions so the user can confirm the env loads correctly.

    Press Ctrl+C in the terminal to exit cleanly.
    """
    env_cls = TASK_ENV[task_key]
    env = env_cls(render=True)
    obs, _ = env.reset()

    print(f"[view] Task '{task_key}' loaded. obs_dim={obs.shape[0]}, "
          f"act_dim={env.action_space.shape[0]}. Press Ctrl+C to quit.")

    try:
        ep_reward, ep_steps, ep_idx = 0.0, 0, 1
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            if terminated or truncated:
                print(f"[view] episode {ep_idx:02d}  steps={ep_steps:3d}  "
                      f"reward={ep_reward:8.2f}  landed={info.get('has_landed', False)}")
                ep_idx += 1
                ep_reward, ep_steps = 0.0, 0
                obs, _ = env.reset()
            time.sleep(env.timestep)
    except KeyboardInterrupt:
        print("\n[view] Interrupted by user, exiting.")
    finally:
        env.close()


# Training -------------------------------------------------------------------
def train(timesteps: int,
          render: bool = False,
          algo_key: str = "sac",
          task_key: str = "jump",
          *,
          model_dir: str = "models jump",
          monitor_csv: str = "logs/sac_monitor.csv",
          tb_log_dir: str = "logs/sac_goal/",
          best_model_subdir: str = "sac_best",
          eval_log_dir: str = "logs/sac_eval/",
          reward_plot_path: str = "reward_curve_sac.png",
          final_model_name: str = "sac_biped_goal",
          crash_model_name: str = "sac_biped_crashsave",
          algo_overrides: dict | None = None) -> str:
    """Train a SAC agent on the biped jump task and save the model.

    Returns the absolute path of the best-model checkpoint written by EvalCallback.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(monitor_csv), exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    best_model_path_dir = os.path.join(model_dir, best_model_subdir)
    os.makedirs(best_model_path_dir, exist_ok=True)

    env_cls = TASK_ENV[task_key]

    train_env = Monitor(env_cls(render=render), filename=monitor_csv)
    eval_env  = Monitor(env_cls(render=False))

    algo_cls, base_cfg = ALGO_MAP[algo_key]
    cfg = dict(base_cfg)
    if algo_overrides:
        cfg.update(algo_overrides)

    model = algo_cls(env=train_env, tensorboard_log=tb_log_dir, **cfg)

    reward_cb = RewardPlotCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path = best_model_path_dir,
        log_path             = eval_log_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
    )

    print(f"[train] algo={algo_key}  task={task_key}  timesteps={timesteps:,}")
    print(f"[train] best-model -> {best_model_path_dir}")

    try:
        model.learn(
            total_timesteps  = timesteps,
            callback         = [reward_cb, eval_cb],
            progress_bar     = False,
            log_interval     = 10,
        )
    except KeyboardInterrupt:
        crash_path = os.path.join(model_dir, crash_model_name)
        print(f"\n[train] KeyboardInterrupt - saving crash checkpoint to {crash_path}.zip")
        model.save(crash_path)
        reward_cb.plot_rewards(reward_plot_path)
        train_env.close()
        eval_env.close()
        raise

    final_path = os.path.join(model_dir, final_model_name)
    model.save(final_path)
    print(f"[train] final model saved to {final_path}.zip")

    reward_cb.plot_rewards(reward_plot_path)

    train_env.close()
    eval_env.close()

    return os.path.join(best_model_path_dir, "best_model")


# Evaluation -----------------------------------------------------------------
def test(model_path: str,
         episodes: int,
         render: bool,
         task_key: str = "jump",
         *,
         metrics_csv: str = "eval_metrics.csv") -> dict:
    """Load a trained SAC model and evaluate it for `episodes` rollouts.

    Computes per-episode and summary metrics required by Task 3:
      Average Reward, Fall Rate (%), Average Distance (m),
      Average Energy (J), Cost of Transport (CoT).

    Returns a dict of summary metrics for downstream aggregation.
    """
    DT = 1.0 / 50.0

    env_cls = TASK_ENV[task_key]
    env = env_cls(render=render)
    model = SAC.load(model_path, env=env)

    joint_idx = env.get_joint_indices()

    total_reward, total_distance, total_energy = 0.0, 0.0, 0.0
    fall_count, success_count = 0, 0
    per_episode = []

    print(f"[test] model={model_path}  episodes={episodes}  task={task_key}")
    print(f"[test] {'ep':>3} {'steps':>6} {'reward':>10} {'dist(m)':>9} "
          f"{'energy(J)':>11} {'landed':>7} {'fall':>5}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        start_pos = env.robot_initial_position()

        ep_reward, ep_energy, steps = 0.0, 0.0, 0
        terminated, truncated = False, False
        info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1

            joint_states = p.getJointState   # micro-opt local binding
            energy_step = 0.0
            for j in joint_idx:
                _pos, vel, _rxn, tau = p.getJointState(
                    env.robot_id, j, physicsClientId=env.physics_client)
                energy_step += abs(tau * vel) * DT
            ep_energy += energy_step

        # Distance per README: "Mean displacement from spawn to landing".
        # Use the recorded landing_pos when the agent actually landed; if it
        # never reached the landing criterion fall back to the final position
        # so the metric is still defined for failed rollouts.
        landing_pos = info.get("landing_pos", None)
        if landing_pos is not None:
            ref_pos = np.asarray(landing_pos, dtype=np.float32)
        else:
            ref_pos = env.robot_current_position()
        dx = float(ref_pos[0] - start_pos[0])
        dy = float(ref_pos[1] - start_pos[1])
        ep_distance = math.sqrt(dx * dx + dy * dy)

        landed  = bool(info.get("has_landed", False))
        success = bool(info.get("success", False))
        # README: Fall Rate = "Percentage of episodes that ended in a crash".
        # A crash is an actual physical failure (pelvis collapse OR post-land
        # topple), reported by the env via the "crash" info flag.
        crash = bool(info.get("crash", False))

        if crash:   fall_count    += 1
        if success: success_count += 1

        total_reward   += ep_reward
        total_distance += ep_distance
        total_energy   += ep_energy

        per_episode.append({
            "episode":  ep,
            "steps":    steps,
            "reward":   ep_reward,
            "distance": ep_distance,
            "energy":   ep_energy,
            "landed":   landed,
            "success":  success,
            "crash":    crash,
        })
        print(f"[test] {ep:>3d} {steps:>6d} {ep_reward:>10.2f} "
              f"{ep_distance:>9.3f} {ep_energy:>11.3f} "
              f"{str(landed):>7} {str(crash):>5}")

    n = float(episodes)
    avg_reward   = total_reward   / n
    fall_rate    = 100.0 * fall_count / n
    success_rate = 100.0 * success_count / n
    avg_distance = total_distance / n
    avg_energy   = total_energy   / n
    cot = total_energy / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print("-" * 72)
    print(f"[test] Average Reward      : {avg_reward:.3f}")
    print(f"[test] Fall Rate (%)       : {fall_rate:.1f}")
    print(f"[test] Success Rate (%)    : {success_rate:.1f}")
    print(f"[test] Average Distance(m) : {avg_distance:.3f}")
    print(f"[test] Average Energy (J)  : {avg_energy:.3f}")
    print(f"[test] Cost of Transport   : {cot:.4f}")

    # Persist per-episode + summary to CSV for the report writer.
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "steps", "reward", "distance_m",
                         "energy_J", "landed", "success", "crash"])
        for row in per_episode:
            writer.writerow([row["episode"], row["steps"], f"{row['reward']:.6f}",
                             f"{row['distance']:.6f}", f"{row['energy']:.6f}",
                             int(row["landed"]), int(row["success"]), int(row["crash"])])
        writer.writerow([])
        writer.writerow(["summary", "avg_reward", "fall_rate_pct", "success_rate_pct",
                         "avg_distance_m", "avg_energy_J", "CoT"])
        writer.writerow(["", f"{avg_reward:.6f}", f"{fall_rate:.4f}",
                         f"{success_rate:.4f}", f"{avg_distance:.6f}",
                         f"{avg_energy:.6f}", f"{cot:.6f}"])
    print(f"[test] metrics CSV saved to {metrics_csv}")

    env.close()

    return {
        "model_path":       model_path,
        "episodes":         episodes,
        "avg_reward":       avg_reward,
        "fall_rate_pct":    fall_rate,
        "success_rate_pct": success_rate,
        "avg_distance_m":   avg_distance,
        "avg_energy_J":     avg_energy,
        "CoT":              cot,
    }


# CLI entry-point ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 - Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode", choices=["view", "train", "test"], required=True,
                        help="view: preview env  |  train: train SAC  |  test: evaluate")
    parser.add_argument("--task", choices=list(TASK_ENV.keys()), default="jump",
                        help="Task key registered in TASK_ENV (default: jump)")
    parser.add_argument("--algo", choices=list(ALGO_MAP.keys()), default="sac",
                        help="Algorithm key registered in ALGO_MAP (default: sac)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override TOTAL_TIMESTEPS from utils.py")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model (.zip) for --mode test")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--render", action="store_true",
                        help="Enable PyBullet GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view(task_key=args.task)
        return

    if args.mode == "train":
        ts = args.timesteps if args.timesteps else TOTAL_TIMESTEPS
        train(timesteps=ts, render=args.render,
              algo_key=args.algo, task_key=args.task)
        return

    if args.mode == "test":
        model_path = args.model_path if args.model_path else DEFAULT_BEST_MODEL
        test(model_path=model_path, episodes=args.episodes,
             render=args.render, task_key=args.task)
        return


if __name__ == "__main__":
    main()
