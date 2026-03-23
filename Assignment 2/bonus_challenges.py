"""
BONUS CHALLENGES - Extra Credit Opportunities

Complete these challenges for up to 20 bonus points!

Each challenge requires implementing advanced RL concepts beyond the basic
Monte Carlo and TD assignments.

Challenge 1: SARSA (On-Policy TD) - 5 points
=============================================
Implement SARSA instead of Q-Learning. SARSA uses the actual next action
(from epsilon-greedy policy) rather than the greedy action.

SARSA Update: Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

Where a' is the action actually chosen in the next state.

Challenge 2: Double Q-Learning - 7 points
=========================================
Implement Double Q-Learning to reduce maximization bias.
Use two Q-tables (Q1 and Q2) and alternate between them.

Update rules:
- With 50% probability: update Q1 using max of Q2
- Otherwise: update Q2 using max of Q1

Selection uses Q1+Q2 average.

Challenge 3: Experience Replay - 8 points
=========================================
Implement Experience Replay buffer to store (s,a,r,s',done) tuples.
Sample random mini-batches for learning instead of online updates.

This should significantly improve learning stability and speed.
"""

import numpy as np
import sys
sys.path.append('..')

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ========================================
# CONFIGURATION
# ========================================
NUM_BINS = 10
STATE_DIM = 3
NUM_EPISODES = 500
MAX_STEPS = 240
EPSILON = 0.1
GAMMA = 0.99
ALPHA = 0.1

# ========================================
# HELPER FUNCTIONS
# ========================================

def extract_position(obs):
    """Extract (x, y, z) from HoverAviary observation."""
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 2:
        return obs_arr[0, 0:3]
    return obs_arr[0:3]

def action_index_to_value(action_idx):
    """Map action index {0,1,2} to thrust adjustment {-1,0,+1}."""
    return float(action_idx - 1)

def format_action(action):
    """Format discrete action index for ONE_D_RPM env.step()."""
    return np.array([[action_index_to_value(action)]], dtype=np.float32)

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
    return 3

def get_q_table_shape():
    return (NUM_BINS,) * STATE_DIM + (get_action_space_size(),)

def initialize_q_table():
    return np.zeros(get_q_table_shape())

def choose_action(q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(get_action_space_size())
    return np.argmax(q_table[state])

def evaluate_policy(env, q_table, num_episodes=10):
    """Evaluate greedy policy."""
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = discretize_state(extract_position(obs))
        total_reward = 0
        for _ in range(MAX_STEPS):
            action = np.argmax(q_table[state])
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            state = discretize_state(extract_position(next_obs))
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# ========================================
# BONUS CHALLENGES - IMPLEMENT THESE
# ========================================

def run_sarsa(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    Challenge 1: Implement SARSA (on-policy TD control).
    
    SARSA Update: Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    Unlike Q-Learning, SARSA chooses the next action using epsilon-greedy
    and uses that action's Q-value in the update.
    
    Returns:
        q_table, episode_rewards
    """
    q_table = initialize_q_table()
    episode_rewards = []
    epsilon_min = 0.01

    for episode in range(num_episodes):
        current_epsilon = max(epsilon_min, epsilon * (1.0 - episode / num_episodes))

        obs, _ = env.reset()
        state = discretize_state(extract_position(obs))
        action = choose_action(q_table, state, current_epsilon)
        total_reward = 0.0

        for step in range(MAX_STEPS):
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs))
            next_action = choose_action(q_table, next_state, current_epsilon)

            td_target = reward + gamma * q_table[next_state][next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            action = next_action
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"SARSA Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return q_table, episode_rewards

def run_double_q_learning(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    Challenge 2: Implement Double Q-Learning to reduce maximization bias.
    
    Use two Q-tables (Q1 and Q2) and alternate between them.
    Selection uses Q1+Q2 average.
    
    Returns:
        q1, q2, episode_rewards
    """
    q1 = initialize_q_table()
    q2 = initialize_q_table()
    episode_rewards = []
    epsilon_min = 0.01

    for episode in range(num_episodes):
        current_epsilon = max(epsilon_min, epsilon * (1.0 - episode / num_episodes))

        obs, _ = env.reset()
        state = discretize_state(extract_position(obs))
        total_reward = 0.0

        for step in range(MAX_STEPS):
            if np.random.random() < current_epsilon:
                action = np.random.randint(get_action_space_size())
            else:
                action = np.argmax(q1[state] + q2[state])

            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs))

            if np.random.random() < 0.5:
                best_next = np.argmax(q1[next_state])
                td_target = reward + gamma * q2[next_state][best_next]
                q1[state][action] += alpha * (td_target - q1[state][action])
            else:
                best_next = np.argmax(q2[next_state])
                td_target = reward + gamma * q1[next_state][best_next]
                q2[state][action] += alpha * (td_target - q2[state][action])

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Double Q-Learning Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return q1, q2, episode_rewards

class ReplayBuffer:
    """
    Challenge 3: Experience Replay buffer for off-policy learning.
    
    Stores (state, action, reward, next_state, done) tuples.
    """
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)  # Remove oldest
    
    def sample(self, batch_size=32):
        """Return random mini-batch of experiences."""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in batch]
    
    def __len__(self):
        return len(self.buffer)

def run_td_with_replay(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA, batch_size=32):
    """
    Challenge 3: Implement TD learning with Experience Replay.
    
    Store experiences, sample mini-batches, update Q-values.
    This should improve learning stability and speed.
    
    Returns:
        q_table, episode_rewards
    """
    q_table = initialize_q_table()
    replay_buffer = ReplayBuffer(capacity=10000)
    episode_rewards = []
    epsilon_min = 0.01
    replay_alpha = alpha * 0.5

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

            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                for s, a, r, ns, d in batch:
                    if d:
                        target = r
                    else:
                        target = r + gamma * np.max(q_table[ns])
                    q_table[s][a] += replay_alpha * (target - q_table[s][a])

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Experience Replay Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return q_table, episode_rewards

# ========================================
# EVALUATION
# ========================================

def evaluate_bonus_challenges():
    """Evaluate bonus challenge implementations."""
    print("=" * 60)
    print("EVALUATING BONUS CHALLENGES")
    print("=" * 60)
    
    # SARSA Evaluation
    print("\n--- Challenge 1: SARSA (5 points) ---")
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    try:
        q_table_sarsa, rewards_sarsa = run_sarsa(env)
        mean_sarsa, std_sarsa = evaluate_policy(env, q_table_sarsa)
        print(f"SARSA Evaluation: {mean_sarsa:.2f} (+/- {std_sarsa:.2f})")
        print(f"Bonus Points: {5 if mean_sarsa >= 300 else 0}/5")
    except Exception as e:
        print(f"SARSA error: {e}")
    env.close()
    
    # Double Q-Learning Evaluation
    print("\n--- Challenge 2: Double Q-Learning (7 points) ---")
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    try:
        q1, q2, rewards_dql = run_double_q_learning(env)
        # Combine Q1 and Q2 for evaluation
        q_combined = (q1 + q2) / 2
        mean_dql, std_dql = evaluate_policy(env, q_combined)
        print(f"Double Q-Learning Evaluation: {mean_dql:.2f} (+/- {std_dql:.2f})")
        print(f"Bonus Points: {7 if mean_dql >= 300 else 0}/7")
    except Exception as e:
        print(f"Double Q-Learning error: {e}")
    env.close()
    
    # Experience Replay Evaluation
    print("\n--- Challenge 3: Experience Replay (8 points) ---")
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    try:
        q_table_replay, rewards_replay = run_td_with_replay(env)
        mean_replay, std_replay = evaluate_policy(env, q_table_replay)
        print(f"Experience Replay Evaluation: {mean_replay:.2f} (+/- {std_replay:.2f})")
        print(f"Bonus Points: {8 if mean_replay >= 300 else 0}/8")
    except Exception as e:
        print(f"Experience Replay error: {e}")
    env.close()
    
    print("\n" + "=" * 60)
    print("BONUS CHALLENGES COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_bonus_challenges()
