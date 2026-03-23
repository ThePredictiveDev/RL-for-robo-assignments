"""
Student Template for RL Assignment: Drone Hover Task.

This template provides stubs for:
1. Monte Carlo (MC) Learning
2. Temporal Difference (TD) Learning - Q-Learning

Students should implement the missing parts marked with TODO.

Environment: HoverAviary - Drone must hover at z=1.0
State: 3D position (x, y, z) relative to target [0, 0, 1]
Action: 3 discrete actions (thrust adjustment): -1, 0, +1
Reward: Based on proximity to target position
"""

import numpy as np
import gymnasium as gym
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ========================================
# CONFIGURATION (Students can modify)
# ========================================
NUM_BINS = 10
STATE_DIM = 3
NUM_EPISODES = 500
MAX_STEPS = 240

EPSILON = 0.1
GAMMA = 0.99
ALPHA = 0.1

# ========================================
# HELPER FUNCTIONS (Do not modify)
# ========================================

def discretize_state(state, num_bins=NUM_BINS):
    """Convert continuous state to discrete bins."""
    state = np.asarray(state)
    if state.ndim == 2:
        state = state[0, 0:3]
    else:
        state = state[0:3]

    bounds = np.array([[-1, 1], [-1, 1], [0, 2]])

    discrete = []
    for val, (low, high) in zip(state, bounds):
        val = np.clip(val, low, high)
        normalized = (val - low) / (high - low)
        bin_idx = int(normalized * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)
        discrete.append(bin_idx)

    return tuple(discrete)

def get_action_space_size():
    """Returns the size of the action space."""
    return 3

def action_index_to_value(action_idx):
    """Map action index {0,1,2} to thrust adjustment {-1,0,+1}."""
    return float(action_idx - 1)

def get_q_table_shape():
    """Returns the shape of the Q-table."""
    return (NUM_BINS,) * STATE_DIM + (get_action_space_size(),)

def initialize_q_table():
    """Initialize Q-table with zeros."""
    return np.zeros(get_q_table_shape())

def choose_action(q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(get_action_space_size())
    return np.argmax(q_table[state])

def extract_position(obs):
    """Extract (x, y, z) from HoverAviary observation."""
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 2:
        return obs_arr[0, 0:3]
    return obs_arr[0:3]

def format_action(action):
    """Format discrete action index for ONE_D_RPM env.step()."""
    return np.array([[action_index_to_value(action)]], dtype=np.float32)

def evaluate_policy(env, q_table, num_episodes=10):
    """Evaluate learned policy (greedy, no exploration)."""
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(extract_position(state))
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_state))

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)

# ========================================
# TODO: MONTE CARLO IMPLEMENTATION
# ========================================

def run_monte_carlo(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    First-visit Monte Carlo Control with epsilon-greedy exploration.

    Generates full episodes, computes discounted returns backwards,
    and updates Q-values only at the first visit of each (state, action) pair.

    Args:
        env: HoverAviary gymnasium environment.
        num_episodes: Number of training episodes.
        epsilon: Initial exploration rate for epsilon-greedy policy.
        gamma: Discount factor for future rewards.
        alpha: Learning rate for incremental Q-value updates.

    Returns:
        tuple: (q_table, episode_rewards) where q_table is the learned
               Q-value table and episode_rewards is a list of total rewards
               per episode.
    """
    q_table = initialize_q_table()
    returns_count = np.zeros(get_q_table_shape(), dtype=np.int64)
    episode_rewards = []
    epsilon_min = 0.01

    for episode in range(num_episodes):
        current_epsilon = max(epsilon_min, epsilon * (1.0 - episode / num_episodes))

        obs, _ = env.reset()
        state = discretize_state(extract_position(obs))

        trajectory = []
        total_reward = 0.0

        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, current_epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            trajectory.append((state, action, reward))
            total_reward += reward
            state = discretize_state(extract_position(next_obs))
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        # Compute returns at every timestep via backwards pass
        T = len(trajectory)
        returns = [0.0] * T
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = gamma * G + trajectory[t][2]
            returns[t] = G

        # First-visit update: forward pass to find first occurrence of each (s, a)
        visited = set()
        for t in range(T):
            state_t, action_t, _ = trajectory[t]
            sa_pair = (state_t, action_t)
            if sa_pair not in visited:
                visited.add(sa_pair)
                returns_count[state_t][action_t] += 1
                n = returns_count[state_t][action_t]
                lr = max(alpha, 1.0 / n)
                q_table[state_t][action_t] += lr * (returns[t] - q_table[state_t][action_t])

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"MC Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 50): {avg_reward:.2f}, "
                  f"Epsilon: {current_epsilon:.4f}")

    return q_table, episode_rewards

# ========================================
# TODO: TD LEARNING IMPLEMENTATION (Q-LEARNING)
# ========================================

def run_q_learning(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    Q-Learning (off-policy TD control) with epsilon-greedy exploration.

    At each step, updates Q-values using the max future Q-value (greedy
    w.r.t. next state) regardless of the action actually taken.

    Args:
        env: HoverAviary gymnasium environment.
        num_episodes: Number of training episodes.
        epsilon: Initial exploration rate for epsilon-greedy policy.
        gamma: Discount factor for future rewards.
        alpha: Learning rate for TD updates.

    Returns:
        tuple: (q_table, episode_rewards) where q_table is the learned
               Q-value table and episode_rewards is a list of total rewards
               per episode.
    """
    q_table = initialize_q_table()
    episode_rewards = []
    epsilon_min = 0.01

    for episode in range(num_episodes):
        current_epsilon = max(epsilon_min, epsilon * (1.0 - episode / num_episodes))

        obs, _ = env.reset()
        state = discretize_state(extract_position(obs))
        total_reward = 0.0

        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, current_epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs))
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(q_table[next_state])
            # Gentle step-size decay satisfies Robbins-Monro conditions
            # and reduces Q-value oscillation in later episodes.
            effective_alpha = alpha / (1.0 + 0.002 * episode)
            q_table[state][action] += effective_alpha * (td_target - q_table[state][action])

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Q-Learning Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 50): {avg_reward:.2f}, "
                  f"Epsilon: {current_epsilon:.4f}")

    return q_table, episode_rewards

# ========================================
# MAIN FUNCTION (Do not modify)
# ========================================

def main():
    """Main function to run MC and TD learning experiments."""

    print("=" * 60)
    print("RL Assignment: Monte Carlo vs TD Learning")
    print("Task: Drone Hover at z=1.0")
    print("=" * 60)

    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    print("Environment: HoverAviary")
    print(f"Target Position: {env.TARGET_POS}")
    print(f"Episode Length: {MAX_STEPS} steps ({MAX_STEPS/30:.1f} seconds)")
    print()

    print("-" * 40)
    print("Training Monte Carlo...")
    print("-" * 40)
    q_table_mc, rewards_mc = run_monte_carlo(env, num_episodes=NUM_EPISODES)
    mean_mc, std_mc = evaluate_policy(env, q_table_mc)
    print(f"MC Final Evaluation: {mean_mc:.2f} (+/- {std_mc:.2f})")

    print()
    print("-" * 40)
    print("Training Q-Learning...")
    print("-" * 40)
    q_table_td, rewards_td = run_q_learning(env, num_episodes=NUM_EPISODES)
    mean_td, std_td = evaluate_policy(env, q_table_td)
    print(f"TD Final Evaluation: {mean_td:.2f} (+/- {std_td:.2f})")

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Monte Carlo - Final Avg Reward (last 50): {np.mean(rewards_mc[-50:]):.2f}")
    print(f"Q-Learning  - Final Avg Reward (last 50): {np.mean(rewards_td[-50:]):.2f}")
    print()
    print(f"Monte Carlo - Evaluation: {mean_mc:.2f} (+/- {std_mc:.2f})")
    print(f"Q-Learning  - Evaluation: {mean_td:.2f} (+/- {std_td:.2f})")
    print()

    if mean_mc > mean_td:
        print("Monte Carlo performed better!")
    elif mean_td > mean_mc:
        print("Q-Learning performed better!")
    else:
        print("Both performed equally!")

    env.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
