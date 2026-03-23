"""
==========================================================================
                        UTILS.PY - STUDENT IMPLEMENTATION
==========================================================================
Students must implement the Dynamic Programming algorithms below.

Author: Assignment 1 - AR525
==========================================================================
"""

from collections import deque

import numpy as np

# Default number of obstacle cells (README: grid with randomly placed obstacles).
NUM_OBSTACLES_DEFAULT = 5


def _bfs_path_exists(rows, cols, start, goal, obstacle_states):
    """Return True iff a path exists from start to goal using 4-neighbors on free cells."""
    if start == goal:
        return True
    if start in obstacle_states or goal in obstacle_states:
        return False
    visited = set()
    q = deque([start])
    visited.add(start)

    def neighbors(state):
        r, c = state // cols, state % cols
        for dr, dc in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                ns = nr * cols + nc
                if ns not in obstacle_states:
                    yield ns

    while q:
        s = q.popleft()
        for ns in neighbors(s):
            if ns == goal:
                return True
            if ns not in visited:
                visited.add(ns)
                q.append(ns)
    return False


def sample_obstacle_states(rows, cols, start, goal, num_obstacles, rng, max_attempts=500):
    """
    Sample ``num_obstacles`` distinct obstacle cells excluding start and goal.
    Resample until BFS confirms start can still reach goal (solvable instance).

    Args:
        rows, cols: Grid shape.
        start, goal: State indices.
        num_obstacles: How many blocked cells.
        rng: numpy.random.Generator
        max_attempts: Cap on resampling to avoid infinite loop.

    Returns:
        frozenset of obstacle state indices.

    Raises:
        RuntimeError: If no valid placement found within max_attempts.
    """
    nS = rows * cols
    if num_obstacles <= 0:
        return frozenset()
    pool = [s for s in range(nS) if s != start and s != goal]
    if len(pool) < num_obstacles:
        raise ValueError(
            f"Cannot place {num_obstacles} obstacles: only {len(pool)} cells available "
            f"(excluding start={start} and goal={goal})."
        )
    for _ in range(max_attempts):
        choice = rng.choice(pool, size=num_obstacles, replace=False)
        obs = frozenset(int(x) for x in choice)
        if _bfs_path_exists(rows, cols, start, goal, obs):
            return obs
    raise RuntimeError(
        f"Could not sample {num_obstacles} obstacles with a valid path after {max_attempts} attempts."
    )


class GridEnv:
    
    def __init__(
        self,
        rows=5,
        cols=6,
        start=0,
        goal=29,
        num_obstacles=NUM_OBSTACLES_DEFAULT,
        obstacle_states=None,
        obstacle_seed=None,
        max_obstacle_resamples=500,
    ):
        """
        Grid world MDP with optional randomly placed obstacle cells.

        Obstacles block movement: attempting to step onto an obstacle leaves the agent
        in the current state (same as hitting an outer wall). States that are obstacles
        are absorbing under self-transitions if ever entered (start/goal are never obstacles).

        Args:
            rows, cols: Grid dimensions.
            start, goal: Start and goal state indices.
            num_obstacles: Number of random obstacles (ignored if obstacle_states is set).
            obstacle_states: If not None, use this exact frozenset/set of obstacle states.
            obstacle_seed: Seed for numpy.random.Generator when sampling obstacles.
            max_obstacle_resamples: Max samples when seeking a solvable obstacle layout.
        """
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}

        if obstacle_states is not None:
            obs = frozenset(int(s) for s in obstacle_states)
            if start in obs or goal in obs:
                raise ValueError("Obstacle states must not include start or goal.")
            if not _bfs_path_exists(rows, cols, start, goal, obs):
                raise ValueError("Provided obstacle_states block all paths from start to goal.")
            self.obstacle_states = obs
        elif num_obstacles and num_obstacles > 0:
            rng = np.random.default_rng(obstacle_seed)
            self.obstacle_states = sample_obstacle_states(
                rows, cols, start, goal, num_obstacles, rng, max_attempts=max_obstacle_resamples
            )
        else:
            self.obstacle_states = frozenset()

        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):

        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):

        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):

        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_next_state(self, state, action):
        # If somehow in an obstacle cell, cannot leave (closed dynamics on obstacle set).
        if state in self.obstacle_states:
            return state

        row, col = self._state_to_pos(state)

        if action == 0:    # LEFT
            col -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # UP
            row -= 1

        if not self._is_valid_pos(row, col):
            return state

        next_state = self._pos_to_state(row, col)
        if next_state in self.obstacle_states:
            return state

        return next_state
    
    def _build_dynamics(self):

        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                next_state = self._get_next_state(state, action)
                
                # Define reward structure:
                # - Large positive reward for reaching goal
                # - Small negative reward for each step (encourages shorter paths)
                # - Slightly higher penalty for hitting walls (staying in place)
                if next_state == self.goal:
                    reward = 100.0
                    done = True
                elif next_state == state:
                    # Hitting wall, boundary, or blocked obstacle cell
                    reward = -2.0
                    done = False
                else:
                    # Normal step cost
                    reward = -1.0
                    done = False
                
                P[state][action] = [(1.0, next_state, reward, done)]
        
        return P
    
    def get_optimal_path(self, policy):
        """
        Extract the optimal path from start to goal using the computed policy.
        
        Args:
            policy: numpy array of shape (nS,) containing action for each state
            
        Returns:
            list: Sequence of states from start to goal
        """
        path = []
        state = self.start
        max_steps = self.nS * 2  # Prevent infinite loops
        visited = set()
        
        while state != self.goal and len(path) < max_steps:
            if state in visited:
                # Detected loop - break to avoid infinite loop
                break
            visited.add(state)
            path.append(state)
            
            # Get action from policy
            action = policy[state]
            
            # Get next state from dynamics
            transitions = self.P[state][action]
            # transitions is a list of (probability, next_state, reward, done)
            # Since our environment is deterministic, there's only one transition
            prob, next_state, reward, done = transitions[0]
            state = next_state
        
        # Add goal state if we reached it
        if state == self.goal:
            path.append(self.goal)
        
        return path


# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS
# ==========================================================================

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Evaluate a policy by computing the state-value function V.
    
    Uses the Bellman expectation equation:
    V(s) = sum_a pi(a|s) * sum_{s',r} P(s',r|s,a) * [r + gamma * V(s')]
    
    Args:
        env: GridEnv instance
        policy: numpy array of shape (nS, nA) representing stochastic policy,
                or shape (nS,) representing deterministic policy
        gamma: discount factor (default 0.99)
        theta: convergence threshold (default 1e-8)
        
    Returns:
        numpy array: State-value function V of shape (nS,)
    """
    # Initialize value function
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        
        # Iterate over all states
        for state in range(env.nS):
            v = 0
            
            # Handle both deterministic and stochastic policies
            if policy.ndim == 1:
                # Deterministic policy: policy[state] gives the action
                action = policy[state]
                action_probs = np.zeros(env.nA)
                action_probs[action] = 1.0
            else:
                # Stochastic policy: policy[state, action] gives probability
                action_probs = policy[state]
            
            # Sum over all actions
            for action in range(env.nA):
                action_prob = action_probs[action]
                
                # Sum over all possible transitions
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            
            # Update delta for convergence check
            delta = max(delta, abs(V[state] - v))
            V[state] = v
        
        # Check for convergence
        if delta < theta:
            break
    
    return V


def q_from_v(env, V, state, gamma=0.99):
    """
    Compute the action-value function Q(s,a) from the state-value function V(s).
    
    Uses the Bellman equation:
    Q(s,a) = sum_{s',r} P(s',r|s,a) * [r + gamma * V(s')]
    
    Args:
        env: GridEnv instance
        V: State-value function, numpy array of shape (nS,)
        state: Current state (int)
        gamma: discount factor (default 0.99)
        
    Returns:
        numpy array: Q-values for all actions at state, shape (nA,)
    """
    Q = np.zeros(env.nA)
    
    for action in range(env.nA):
        q_value = 0
        
        # Sum over all possible transitions
        for prob, next_state, reward, done in env.P[state][action]:
            q_value += prob * (reward + gamma * V[next_state])
        
        Q[action] = q_value
    
    return Q


def policy_improvement(env, V, gamma=0.99):
    """
    Improve a policy by making it greedy with respect to the value function.
    
    Computes new policy: pi'(s) = argmax_a Q(s,a)
    
    Args:
        env: GridEnv instance
        V: State-value function, numpy array of shape (nS,)
        gamma: discount factor (default 0.99)
        
    Returns:
        numpy array: New deterministic policy, shape (nS,)
    """
    policy = np.zeros(env.nS, dtype=int)
    
    for state in range(env.nS):
        # Compute Q-values for all actions at this state
        Q = q_from_v(env, V, state, gamma)
        
        # Select the action with highest Q-value (greedy)
        policy[state] = np.argmax(Q)
    
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Run Policy Iteration algorithm to find optimal policy.
    
    Algorithm:
    1. Initialize random policy
    2. Loop:
       - Policy Evaluation: Compute V for current policy
       - Policy Improvement: Get new greedy policy
       - Stop if policy doesn't change
    
    Args:
        env: GridEnv instance
        gamma: discount factor (default 0.99)
        theta: convergence threshold for policy evaluation (default 1e-8)
        
    Returns:
        tuple: (optimal_policy, optimal_V)
            - optimal_policy: numpy array of shape (nS,)
            - optimal_V: numpy array of shape (nS,)
    """
    # Initialize with a random policy
    policy = np.random.randint(0, env.nA, size=env.nS)
    
    iteration = 0
    while True:
        iteration += 1
        
        # Policy Evaluation: Compute V for current policy
        V = policy_evaluation(env, policy, gamma, theta)
        
        # Policy Improvement: Get new greedy policy
        new_policy = policy_improvement(env, V, gamma)
        
        # Check if policy has converged
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Run Value Iteration algorithm to find optimal policy.
    
    Uses the Bellman optimality equation:
    V(s) = max_a sum_{s',r} P(s',r|s,a) * [r + gamma * V(s')]
    
    Args:
        env: GridEnv instance
        gamma: discount factor (default 0.99)
        theta: convergence threshold (default 1e-8)
        
    Returns:
        tuple: (optimal_policy, optimal_V)
            - optimal_policy: numpy array of shape (nS,)
            - optimal_V: numpy array of shape (nS,)
    """
    # Initialize value function
    V = np.zeros(env.nS)
    
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        
        # Iterate over all states
        for state in range(env.nS):
            # Compute Q-values for all actions
            Q = q_from_v(env, V, state, gamma)
            
            # Get the maximum Q-value
            max_q = np.max(Q)
            
            # Update delta for convergence check
            delta = max(delta, abs(V[state] - max_q))
            V[state] = max_q
        
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy from optimal value function
    policy = policy_improvement(env, V, gamma)
    
    return policy, V
