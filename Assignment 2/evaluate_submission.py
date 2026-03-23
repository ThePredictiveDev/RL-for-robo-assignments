"""
Evaluation Framework for RL Assignment Submissions.

This script evaluates student implementations of:
1. Monte Carlo Learning
2. Temporal Difference Learning (Q-Learning)

Usage:
    python evaluate_submission.py --student_file user_code.py --method mc
    python evaluate_submission.py --student_file user_code.py --method td
    python evaluate_submission.py --student_file user_code.py --method all
"""

import numpy as np
import argparse
import sys
import os
import gymnasium as gym
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ========================================
# GRADING RUBRIC
# ========================================
RUBRIC = {
    "mc_implementation": {
        "weight": 0.30,
        "criteria": [
            "Episode generation with epsilon-greedy policy",
            "Return calculation (discounted cumulative reward)",
            "First-visit MC update (Q-value averaging)",
            "Policy improvement (greedy w.r.t. Q-values)"
        ]
    },
    "td_implementation": {
        "weight": 0.30,
        "criteria": [
            "TD update rule implementation",
            "Max Q-value calculation for next state",
            "Epsilon-greedy action selection",
            "Convergence over episodes"
        ]
    },
    "experiments": {
        "weight": 0.25,
        "criteria": [
            "Hyperparameter tuning",
            "Learning curve plotting",
            "Comparison of MC vs TD",
            "Ablation studies (if any)"
        ]
    },
    "code_quality": {
        "weight": 0.15,
        "criteria": [
            "Code readability and documentation",
            "Proper variable naming",
            "Error handling",
            "Comments explaining key steps"
        ]
    }
}

# ========================================
# EVALUATION SETTINGS
# ========================================
NUM_EVAL_EPISODES = 10  # Number of evaluation episodes
MIN_ACCEPTABLE_REWARD = 220  # Minimum reward for passing
DEFAULT_SEED = 42
NUM_EVAL_SEEDS = 3

def format_action(action):
    """Format discrete action index for ONE_D_RPM env.step()."""
    return np.array([[float(action - 1)]], dtype=np.float32)

def extract_position(obs):
    """Extract (x, y, z) from HoverAviary observation."""
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 2:
        return obs_arr[0, 0:3]
    return obs_arr[0:3]

# ========================================
# EVALUATION FUNCTIONS
# ========================================

def evaluate_policy(env, q_table, discretize_func, num_episodes=10, seed=DEFAULT_SEED):
    """
    Evaluate greedy policy from Q-table.
    
    Args:
        env: Gym environment
        q_table: Learned Q-table
        discretize_func: Function to discretize states
        num_episodes: Number of evaluation episodes
    
    Returns:
        tuple: (mean_reward, std_reward)
    """
    rewards = []
    max_steps = 240
    
    for episode_idx in range(num_episodes):
        state, _ = env.reset(seed=seed + episode_idx)
        state = discretize_func(extract_position(state))
        total_reward = 0
        
        for _ in range(max_steps):
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_func(extract_position(next_state))
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

def evaluate_mc_student(
    env,
    student_module,
    num_episodes=NUM_EVAL_EPISODES,
    seed=DEFAULT_SEED,
    min_acceptable_reward=MIN_ACCEPTABLE_REWARD,
    num_eval_seeds=NUM_EVAL_SEEDS,
):
    """
    Evaluate student Monte Carlo implementation.
    
    Args:
        env: Gym environment
        student_module: Student module with run_monte_carlo function
        num_episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation results
    """
    print("=" * 60)
    print("EVALUATING MONTE CARLO IMPLEMENTATION")
    print("=" * 60)
    
    try:
        np.random.seed(seed)
        # Run student MC implementation
        q_table, rewards = student_module.run_monte_carlo(
            env,
            num_episodes=500,
            epsilon=0.1,
            gamma=0.99,
            alpha=0.1
        )
        
        # Evaluate learned policy
        seed_means = []
        for seed_offset in range(num_eval_seeds):
            mean_reward_i, _ = evaluate_policy(
                env,
                q_table,
                student_module.discretize_state,
                num_episodes=num_episodes,
                seed=seed + 1000 * seed_offset,
            )
            seed_means.append(mean_reward_i)
        mean_reward = float(np.mean(seed_means))
        std_reward = float(np.std(seed_means))
        
        # Calculate additional metrics
        final_avg_50 = np.mean(rewards[-50:])
        convergence_speed = np.argmax([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
        
        results = {
            "passed": mean_reward >= min_acceptable_reward,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "final_avg_reward": final_avg_50,
            "convergence_episode": convergence_speed,
            "total_episodes": len(rewards),
            "q_table_shape": q_table.shape
        }
        
        print(f"Final Evaluation Reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
        print(f"Final 50-Episode Average: {final_avg_50:.2f}")
        print(f"Convergence Episode: {convergence_speed}")
        print(f"Q-Table Shape: {q_table.shape}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Student MC implementation failed: {e}")
        return {"passed": False, "error": str(e)}

def evaluate_td_student(
    env,
    student_module,
    num_episodes=NUM_EVAL_EPISODES,
    seed=DEFAULT_SEED,
    min_acceptable_reward=MIN_ACCEPTABLE_REWARD,
    num_eval_seeds=NUM_EVAL_SEEDS,
):
    """
    Evaluate student TD (Q-Learning) implementation.
    
    Args:
        env: Gym environment
        student_module: Student module with run_q_learning function
        num_episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation results
    """
    print("=" * 60)
    print("EVALUATING TD (Q-LEARNING) IMPLEMENTATION")
    print("=" * 60)
    
    try:
        np.random.seed(seed)
        # Run student Q-Learning implementation
        q_table, rewards = student_module.run_q_learning(
            env,
            num_episodes=500,
            epsilon=0.1,
            gamma=0.99,
            alpha=0.1
        )
        
        # Evaluate learned policy
        seed_means = []
        for seed_offset in range(num_eval_seeds):
            mean_reward_i, _ = evaluate_policy(
                env,
                q_table,
                student_module.discretize_state,
                num_episodes=num_episodes,
                seed=seed + 1000 * seed_offset,
            )
            seed_means.append(mean_reward_i)
        mean_reward = float(np.mean(seed_means))
        std_reward = float(np.std(seed_means))
        
        # Calculate additional metrics
        final_avg_td = np.mean(rewards[-50:])
        convergence_speed_td = np.argmax([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
        
        results = {
            "passed": mean_reward >= min_acceptable_reward,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "final_avg_reward": final_avg_td,
            "convergence_episode": convergence_speed_td,
            "total_episodes": len(rewards),
            "q_table_shape": q_table.shape
        }
        
        print(f"Final Evaluation Reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
        print(f"Final 50-Episode Average: {final_avg_td:.2f}")
        print(f"Convergence Episode: {convergence_speed_td}")
        print(f"Q-Table Shape: {q_table.shape}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Student TD implementation failed: {e}")
        return {"passed": False, "error": str(e)}

# ========================================
# GRADING FUNCTION
# ========================================

def calculate_grade(mc_results, td_results, min_acceptable_reward=MIN_ACCEPTABLE_REWARD):
    """
    Calculate final grade based on evaluation results.
    
    Args:
        mc_results: MC evaluation results
        td_results: TD evaluation results
    
    Returns:
        dict: Grade breakdown
    """
    grade = {
        "mc_score": 0.0,
        "td_score": 0.0,
        "experiment_score": 0.0,
        "code_quality_score": 0.0,
        "total_grade": 0.0,
        "passed": False,
        "feedback": []
    }
    
    # MC Score
    if mc_results.get("passed", False):
        grade["mc_score"] = RUBRIC["mc_implementation"]["weight"] * 100
        grade["feedback"].append("✓ Monte Carlo implementation PASSED")
    else:
        grade["feedback"].append("✗ Monte Carlo implementation FAILED")
        if "mean_reward" in mc_results:
            grade["feedback"].append(f"  - Reward: {mc_results['mean_reward']:.2f} (threshold: {min_acceptable_reward})")
    
    # TD Score
    if td_results.get("passed", False):
        grade["td_score"] = RUBRIC["td_implementation"]["weight"] * 100
        grade["feedback"].append("✓ TD (Q-Learning) implementation PASSED")
    else:
        grade["feedback"].append("✗ TD (Q-Learning) implementation FAILED")
        if "mean_reward" in td_results:
            grade["feedback"].append(f"  - Reward: {td_results['mean_reward']:.2f} (threshold: {min_acceptable_reward})")
    
    # Experiment Score (based on convergence and stability)
    mc_convergence = mc_results.get("convergence_episode", 500)
    td_convergence = td_results.get("convergence_episode", 500)
    
    if mc_convergence < 300 and td_convergence < 300:
        grade["experiment_score"] = RUBRIC["experiments"]["weight"] * 100
        grade["feedback"].append("✓ Both algorithms converged quickly")
    elif mc_convergence < 500 or td_convergence < 500:
        grade["experiment_score"] = RUBRIC["experiments"]["weight"] * 75
        grade["feedback"].append("~ One algorithm converged reasonably")
    else:
        grade["experiment_score"] = RUBRIC["experiments"]["weight"] * 50
        grade["feedback"].append("~ Convergence was slow")
    
    # Total Grade
    grade["total_grade"] = (
        grade["mc_score"] +
        grade["td_score"] +
        grade["experiment_score"]
    )
    
    # Passing grade (70% overall)
    grade["passed"] = grade["total_grade"] >= 70.0
    
    return grade

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Assignment Submissions")
    parser.add_argument("--student_file", type=str, default="user_code.py",
                        help="Path to student Python file")
    parser.add_argument("--method", type=str, default="all",
                        choices=["mc", "td", "all"],
                        help="Method to evaluate: 'mc', 'td', or 'all'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for deterministic evaluation")
    parser.add_argument("--min_reward", type=float, default=MIN_ACCEPTABLE_REWARD,
                        help="Minimum mean reward required to pass each method")
    parser.add_argument("--eval_seeds", type=int, default=NUM_EVAL_SEEDS,
                        help="Number of random seeds to average during policy evaluation")
    
    args = parser.parse_args()
    
    # Load student module
    print("=" * 60)
    print("RL ASSIGNMENT EVALUATION")
    print("=" * 60)
    print(f"Loading student file: {args.student_file}")
    
    # Import student module
    import importlib.util
    spec = importlib.util.spec_from_file_location("student_module", args.student_file)
    student_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(student_module)
    except Exception as e:
        print(f"ERROR: Failed to load student file: {e}")
        sys.exit(1)
    
    print("Student module loaded successfully!")
    print()
    
    # Initialize environment
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    
    mc_results = {}
    td_results = {}
    
    # Evaluate MC
    if args.method in ["mc", "all"]:
        env.close()
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
        mc_results = evaluate_mc_student(
            env,
            student_module,
            seed=args.seed,
            min_acceptable_reward=args.min_reward,
            num_eval_seeds=args.eval_seeds
        )
    
    # Evaluate TD
    if args.method in ["td", "all"]:
        env.close()
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
        td_results = evaluate_td_student(
            env,
            student_module,
            seed=args.seed,
            min_acceptable_reward=args.min_reward,
            num_eval_seeds=args.eval_seeds
        )
    
    # Calculate grade
    print()
    print("=" * 60)
    print("FINAL GRADE")
    print("=" * 60)
    
    if args.method == "all":
        grade = calculate_grade(mc_results, td_results, min_acceptable_reward=args.min_reward)
        
        print("\nFeedback:")
        for fb in grade["feedback"]:
            print(f"  {fb}")
        
        print(f"\nScore Breakdown:")
        print(f"  Monte Carlo: {grade['mc_score']:.1f}/30")
        print(f"  TD Learning: {grade['td_score']:.1f}/30")
        print(f"  Experiments: {grade['experiment_score']:.1f}/25")
        print(f"  -------------------")
        print(f"  TOTAL: {grade['total_grade']:.1f}/85")
        print()
        
        if grade["passed"]:
            print("✓✓✓ PASSED ASSIGNMENT ✓✓✓")
        else:
            print("✗✗✗ NEEDS IMPROVEMENT ✗✗✗")
    
    env.close()
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
