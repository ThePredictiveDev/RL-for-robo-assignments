"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.

Contains
--------
  - SAC_CONFIG          Hyperparameters for Soft Actor-Critic (YOU will tune these)
  - Training constants  TOTAL_TIMESTEPS, EVAL_FREQ, EVAL_EPISODES, ROBOT_MASS_KG
  - RewardPlotCallback  Records episode rewards and saves a plot after training
  - BipedJumpEnv        Gymnasium environment — provided, do not modify
"""

# ===========================================================================
# Hyperparameters  (edit these for Task 3)
# ===========================================================================

# Total training timesteps for the default SAC run (Task 2 requires >= 500_000).
TOTAL_TIMESTEPS = 500_000

# How often (env steps) EvalCallback rolls out the deterministic policy.
EVAL_FREQ = 10_000

# Max steps per episode — must match BipedJumpEnv.max_steps below.
MAX_EPISODE_STEPS = 500

# ---------------------------------------------------------------------------
# SAC  (Soft Actor-Critic) — the only algorithm used in this assignment.
# Baseline ("Config A") that matches Haarnoja et al. (2018) defaults adapted
# for a small biped (6 actuators, 25-D obs, 50 Hz control).
# ---------------------------------------------------------------------------
SAC_CONFIG = dict(
    policy           = "MlpPolicy",
    learning_rate    = 3e-4,
    buffer_size      = 1_000_000,
    batch_size       = 256,
    tau              = 0.005,
    gamma            = 0.99,
    ent_coef         = "auto",
    learning_starts  = 10_000,
    train_freq       = 1,
    gradient_steps   = 1,
    policy_kwargs    = dict(net_arch=[256, 256]),
    verbose          = 1,
)

# ---------------------------------------------------------------------------
# Evaluation / metric settings  (do not change)
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05   # used to compute Cost of Transport (CoT)


# ===========================================================================
# RewardPlotCallback
# ===========================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]

        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True   # returning False would stop training

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")

        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv  — provided environment, do not modify
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSEST_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: the biped robot spawns on top of a 1 m tall platform and must
    jump off, then land upright on the ground below.

    Phases
    ------
    1. On platform  
    2. In flight    
    3. Landing      

   
    """

    # Geometry constants (matched to README so the README hints hold verbatim):
    #   - Platform: 1.0 m tall stage with a 2 x 2 m footprint, top at z=1.0.
    #   - The URDF biped is loaded with `globalScaling = ROBOT_SCALE` so the
    #     foot-to-pelvis standing height becomes 0.31 * 2.6 = 0.806 m, matching
    #     the README hint "spawn z = 1.81 m  (platform top + standing height)".
    #   - PyBullet's globalScaling also scales link masses by scale^3, which
    #     would swamp the URDF's 15 N.m joint limits, so we explicitly RESET
    #     each link mass back to its URDF value via changeDynamics during
    #     environment construction (see _restore_link_masses below).
    #   - README landing criterion: pelvis z < 1.15 m AND both feet on ground.
    #     With a 0.806 m standing height the robot satisfies z<1.15 m the
    #     instant it lands on the plane, so this is exactly the criterion
    #     used for `landed_now` in step().
    #   - FALLEN_Z must sit well BELOW the standing pelvis height (0.806 m)
    #     so that successful landings are not immediately killed.
    ROBOT_SCALE       = 2.6
    PLATFORM_H        = 1.0
    PLATFORM_HALF_XY  = 0.75                         # 1.5 m x 1.5 m stage
    STAND_H           = 0.31 * ROBOT_SCALE           # ~0.806 m
    SPAWN_Z           = PLATFORM_H + STAND_H + 0.005 # ~1.81 m
    SPAWN_X           = -0.30                        # 0.30 m back from centre
    GROUND_Z          = STAND_H                      # standing pelvis on ground
    LAND_Z_THRESHOLD  = 1.15                         # README: z < 1.15 m
    FALLEN_Z          = 0.30                         # ~37 % of standing height

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = 500
        self.step_counter      = 0
        self.land_stable_steps = 0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        # 1 m tall platform with a 1.5 m x 1.5 m footprint - wide enough for
        # a 2-3 step walk-up but small enough that the +x edge is reachable
        # in well under a 500-step episode at the agent's natural gait speed.
        half_x = self.PLATFORM_HALF_XY
        half_y = self.PLATFORM_HALF_XY
        half_z = self.PLATFORM_H * 0.5
        plat_col = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[half_x, half_y, half_z],
                                          physicsClientId=cid)
        plat_vis = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[half_x, half_y, half_z],
                                       rgbaColor=[0.78, 0.78, 0.80, 1.0],
                                       physicsClientId=cid)
        self.platform_id = p.createMultiBody(0, plat_col, plat_vis,
                                              [0, 0, half_z],
                                              physicsClientId=cid)
        p.changeDynamics(self.platform_id, -1,
                         lateralFriction=1.2, restitution=0.0,
                         physicsClientId=cid)

        # Robot - URDF is geometrically scaled up so the foot-to-pelvis height
        # is ~0.81 m, matching the README's "spawn = 1.81 m" hint. PyBullet's
        # globalScaling also multiplies link masses by ROBOT_SCALE^3, which
        # would put the robot well outside the URDF's 15 N.m joint envelope,
        # so we restore each link mass to its original URDF value below.
        urdf_path = os.path.join(_ASSEST_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, self.SPAWN_Z],
                                    useFixedBase    = False,
                                    globalScaling   = self.ROBOT_SCALE,
                                    physicsClientId = cid)
        self._restore_link_masses()
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z     = self.SPAWN_Z
        self.has_landed = False
        self.reset()

    # ------------------------------------------------------------------
    # Original URDF mass per link (kg). Loaded once and re-applied via
    # changeDynamics so that PyBullet's globalScaling does not turn the
    # 2.05 kg robot into a 36 kg one that the URDF's 15 N.m joints could
    # never lift. Inertias are rescaled to mass * (link_extent / 12).
    _URDF_LINK_MASSES = {
        "pelvis":      0.600,
        "left_thigh":  0.300,
        "left_shin":   0.300,
        "left_foot":   0.125,
        "right_thigh": 0.300,
        "right_shin":  0.300,
        "right_foot":  0.125,
    }

    def _restore_link_masses(self):
        """Resets every link mass back to the value declared in the URDF.

        PyBullet's `globalScaling` multiplies link masses by `scale^3`, which
        for `ROBOT_SCALE=2.6` would inflate the 2.05 kg robot to ~36 kg and
        instantly exceed the URDF's 15 N.m joint torque envelope. We therefore
        explicitly restore each link's URDF mass and let PyBullet recompute
        the inertia tensor from the (now consistent) collision geometry.
        """
        cid = self.physics_client
        # Base link is index -1, named via the URDF root link name "pelvis".
        p.changeDynamics(self.robot_id, -1,
                         mass=self._URDF_LINK_MASSES["pelvis"],
                         physicsClientId=cid)
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=cid)
            link_name = info[12].decode("utf-8")
            if link_name in self._URDF_LINK_MASSES:
                p.changeDynamics(self.robot_id, j,
                                 mass=self._URDF_LINK_MASSES[link_name],
                                 physicsClientId=cid)

    # ------------------------------------------------------------------
    def _cache_joint_metadata(self):
        """Lazily caches per-joint max-force and position limits from the URDF."""
        if getattr(self, "_joint_meta_cached", False):
            return
        self.joint_max_force = []
        self.joint_lower = []
        self.joint_upper = []
        for j in self.joint_indices:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.physics_client)
            self.joint_lower.append(float(info[8]))
            self.joint_upper.append(float(info[9]))
            # info[10] = jointMaxForce (URDF "effort"). Fall back to 15 N.m if zero.
            self.joint_max_force.append(float(info[10]) if info[10] > 0 else 15.0)
        self.joint_lower = np.asarray(self.joint_lower, dtype=np.float32)
        self.joint_upper = np.asarray(self.joint_upper, dtype=np.float32)
        self.joint_max_force = np.asarray(self.joint_max_force, dtype=np.float32)
        # URDF zero pose corresponds to the robot standing fully extended,
        # which is the natural neutral / equilibrium configuration.
        self._standing_pose = np.clip(np.zeros_like(self.joint_lower),
                                      self.joint_lower, self.joint_upper)
        self._joint_meta_cached = True

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Re-positions the robot at spawn, zeros joints/velocities, resets counters."""
        super().reset(seed=seed)

        self._cache_joint_metadata()

        cid = self.physics_client
        # Re-place the robot at the platform-top spawn pose with identity orientation.
        # Spawn ~0.30 m behind the centre so the robot has a clear forward walk
        # toward the +x edge (~1.05 m away) instead of starting at the rim.
        spawn_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.SPAWN_X, 0.0, self.SPAWN_Z],
            spawn_quat, physicsClientId=cid
        )
        p.resetBaseVelocity(
            self.robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid
        )

        # Zero every actuated joint and clear residual motor torques.
        for j in self.joint_indices:
            p.resetJointState(self.robot_id, j, targetValue=0.0,
                              targetVelocity=0.0, physicsClientId=cid)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL,
                                    force=0.0, physicsClientId=cid)

        self.step_counter         = 0
        self.land_stable_steps    = 0
        self.has_landed           = False
        self.land_step            = -1
        self.landing_pos          = None
        self._left_platform_once  = False
        self.prev_z               = float(self.SPAWN_Z)
        self.prev_x               = float(self.SPAWN_X)
        self.initial_pos          = np.array([self.SPAWN_X, 0.0, self.SPAWN_Z],
                                              dtype=np.float32)
        self._last_action         = np.zeros(self.n_actuated, dtype=np.float32)
        self._landed_this_step    = False
        self._prev_l_contact      = 0.0
        self._prev_r_contact      = 0.0
        self._step_count_with_swap = 0

        # Settle the robot for one frame so contact buffers are valid.
        p.stepSimulation(physicsClientId=cid)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Returns the 25-D observation vector: joint pos+vel, base pose, velocity,
        foot contacts, COM height, and a binary 'has_landed' flag."""
        cid = self.physics_client

        joint_states = p.getJointStates(self.robot_id, self.joint_indices,
                                        physicsClientId=cid)
        joint_pos = np.asarray([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.asarray([s[1] for s in joint_states], dtype=np.float32)

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id,
                                                             physicsClientId=cid)
        lin_vel, _ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=cid)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        l_contact = 1.0 if p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.plane_id,
            linkIndexA=self.left_foot_link, physicsClientId=cid) else 0.0
        r_contact = 1.0 if p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.plane_id,
            linkIndexA=self.right_foot_link, physicsClientId=cid) else 0.0

        # Any-link contact between the robot and the platform body. Used to
        # differentiate "actually airborne" from "crumpled on the platform top".
        platform_touch = 1.0 if p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.platform_id, physicsClientId=cid) else 0.0

        obs = np.concatenate([
            joint_pos,                                         # 6
            joint_vel,                                         # 6
            np.asarray(base_pos, dtype=np.float32),            # 3
            np.asarray([roll, pitch, yaw], dtype=np.float32),  # 3
            np.asarray(lin_vel, dtype=np.float32),             # 3
            np.asarray([l_contact, r_contact], dtype=np.float32),  # 2
            np.asarray([base_pos[2]], dtype=np.float32),       # 1
            np.asarray([1.0 if self.has_landed else 0.0], dtype=np.float32),  # 1
        ]).astype(np.float32)

        # Cache for step() and reward function so we don't query twice.
        self._cache_pos       = base_pos
        self._cache_orn       = (roll, pitch, yaw)
        self._cache_lin_vel   = lin_vel
        self._cache_contacts  = (l_contact, r_contact)
        self._cache_plat_touch = platform_touch
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, landed_now, success):
        """Dense reward designed to produce a natural sequence:
        (1) WALK forward across the platform toward the +x edge,
        (2) LEAVE the platform from the forward edge (not by tumbling sideways),
        (3) MAINTAIN upright posture in flight,
        (4) LAND stably on the ground (z < 1.15 m AND both feet on the plane),
        (5) STAY UP for the README's 20-step stability window.

        The shaping is rooted in the bipedal-locomotion literature: a constant
        small `alive_bonus` so existing-while-upright is encouraged, a strong
        `forward_velocity` term so the agent is rewarded for walking in +x,
        a quadratic `upright_penalty` that grows fast as the torso tips, and
        a `gait_bonus` that fires when the two feet are in DIFFERENT contact
        states (the canonical single-support phase of a walking cycle, used
        by Schulman et al. 2015 / DeepMimic / Walker2D-style envs).

        Calibration (approx total rewards over a 500-step episode):
        - "stand still upright on platform"      ~  -100
        - "wriggle on platform and tumble off"   ~  -200
        - "walk to edge, fall flat and crash"    ~   +50
        - "walk, jump, land but topple"          ~  +700
        - "walk, jump, land + 20 stable steps"   ~ +1300
        """
        roll, pitch, _yaw = orn
        x, _y, z = float(pos[0]), float(pos[1]), float(pos[2])
        vx, _vy, _vz = lin_vel
        forward_v = float(vx)

        l_contact, r_contact = self._cache_contacts
        feet_on_ground = (l_contact == 1.0 and r_contact == 1.0)
        on_platform    = self._cache_plat_touch == 1.0
        upright        = (abs(roll) < 0.4 and abs(pitch) < 0.4)

        # ---- Per-step survival -------------------------------------------
        # Small alive bonus only while upright. Quadratic upright penalty is
        # mild on the platform (lets the robot lean to initiate a step) and
        # strict after landing (so the robot has to recover balance to score).
        alive_bonus     = 0.5 if upright else 0.0
        if self.has_landed:
            upright_penalty = -3.0 * (roll * roll + pitch * pitch)
        else:
            upright_penalty = -1.5 * (roll * roll + pitch * pitch)

        # ---- Walking incentive (active any time the robot is on its feet) -
        # Reward forward velocity directly. Make standing still on the
        # platform STRONGLY negative so the agent cannot earn reward by
        # freezing in place (the local optimum we observed in the smoke test).
        forward_reward  = 2.0 * float(np.clip(forward_v, -0.5, 2.0))
        moving_forward  = forward_v > 0.10
        stuck_penalty   = (-2.5 if (on_platform and not moving_forward
                                    and not self.has_landed) else 0.0)

        # ---- Bipedal gait shaping (single-support reward) ----------------
        # +0.5 only when EXACTLY one foot is in contact AND the robot is
        # actually moving forward - the canonical single-support phase of a
        # walking cycle. Awarding it without forward motion would encourage
        # marching in place, so we gate on `moving_forward`.
        single_support  = ((l_contact == 1.0) ^ (r_contact == 1.0))
        gait_bonus      = 0.5 if (single_support and moving_forward
                                  and not self.has_landed) else 0.0

        # ---- Position-toward-edge progress (dense walking signal) --------
        # Reward closing the distance to the +x platform edge while still on
        # the platform. This gives a continuous gradient from the spawn pose
        # all the way to the edge, regardless of momentary velocity dips.
        forward_edge = self.PLATFORM_HALF_XY - 0.05
        if on_platform and not self.has_landed:
            edge_progress = 5.0 * float(x - self.prev_x)   # delta-x to edge
        else:
            edge_progress = 0.0

        # ---- Edge-leaving one-shot ---------------------------------------
        # Big bonus when the robot leaves the platform from the forward edge.
        # Sideways / backward falls get a NEGATIVE one-shot so they are
        # actively discouraged in favour of a clean walk-off.
        edge_bonus = 0.0
        if (not self._left_platform_once) and (not on_platform):
            self._left_platform_once = True
            if x > forward_edge and upright:
                edge_bonus = 200.0
            elif x > forward_edge:
                edge_bonus = 100.0
            else:
                edge_bonus = -50.0

        # ---- Flight phase shaping ----------------------------------------
        # Strong per-step reward for staying upright while airborne.
        in_flight       = (not on_platform) and (not self.has_landed) and (z > 0.4)
        flight_upright  = 5.0 if (in_flight and upright) else 0.0
        descent_prog    = (3.0 * max(0.0, self.prev_z - z)
                           if (in_flight or not self.has_landed) else 0.0)

        # ---- Landing + stabilisation -------------------------------------
        landed_one_shot = 500.0 if landed_now else 0.0
        if self.has_landed and feet_on_ground and upright:
            post_land_stable = 10.0
        else:
            post_land_stable = 0.0
        success_bonus   = 500.0 if success else 0.0

        # ---- Action regulariser (CoT proxy) ------------------------------
        action_penalty  = -0.001 * float(np.sum(self._last_action ** 2))

        return (alive_bonus + upright_penalty + forward_reward + stuck_penalty
                + gait_bonus + edge_progress + edge_bonus + flight_upright
                + descent_prog + landed_one_shot + post_land_stable
                + success_bonus + action_penalty)

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        """Returns the list of revolute (actuated) joint indices."""
        return list(self.joint_indices)

    def robot_initial_position(self):
        """Returns the robot base position at the start of the current episode."""
        return np.asarray(self.initial_pos, dtype=np.float32)

    def robot_current_position(self):
        """Returns the current robot base position as a numpy array."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id,
                                                 physicsClientId=self.physics_client)
        return np.asarray(pos, dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        """Position control biased so action=0 corresponds to the URDF zero
        pose (legs fully extended, robot standing). action in [-1, 1] is then
        mapped to the joint's [lower, upper] range with the standing pose at
        zero, which keeps the unforced policy near the upright equilibrium and
        lets exploration populate both 'maintain-balance' and 'leave-edge'
        transitions early in training.
        """
        cid = self.physics_client

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action

        # action=0  -> 0 (standing pose, robot's URDF zero configuration)
        # action=+1 -> upper limit ; action=-1 -> lower limit
        positive = action >= 0
        targets  = np.where(positive,
                            self._standing_pose + action * (self.joint_upper - self._standing_pose),
                            self._standing_pose + action * (self._standing_pose - self.joint_lower))
        targets  = np.clip(targets, self.joint_lower, self.joint_upper).astype(np.float32)

        for idx, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId    = self.robot_id,
                jointIndex      = j,
                controlMode     = p.POSITION_CONTROL,
                targetPosition  = float(targets[idx]),
                force           = float(2.0 * self.joint_max_force[idx]),  # 2x URDF effort
                positionGain    = 0.5,
                velocityGain    = 1.0,
                physicsClientId = cid,
            )

        p.stepSimulation(physicsClientId=cid)
        if self.render_mode:
            time.sleep(self.timestep)

        self.step_counter += 1
        obs = self._get_obs()
        pos, orn, lin_vel = self._cache_pos, self._cache_orn, self._cache_lin_vel
        l_contact, r_contact = self._cache_contacts

        # ---- Landing detection (README: z < 1.15 m AND both feet down) -------
        feet_on_ground = (l_contact == 1.0 and r_contact == 1.0)
        landed_now = ((not self.has_landed)
                      and feet_on_ground
                      and (pos[2] < self.LAND_Z_THRESHOLD))
        if landed_now:
            self.has_landed     = True
            self.land_step      = self.step_counter
            self.landing_pos    = np.asarray(pos, dtype=np.float32)
            self.land_stable_steps = 0
        elif self.has_landed and feet_on_ground \
                and abs(orn[0]) < 0.4 and abs(orn[1]) < 0.4:
            self.land_stable_steps += 1

        # ---- Termination ------------------------------------------------------
        # `fallen` = pelvis collapsed below floor level (true physical crash).
        # `post_land_crash` = robot landed then tipped past ~115 deg in any
        # axis. Both qualify as the README's "crash" for Fall-Rate accounting.
        # `success` closes the 20-step stability window.
        # We do NOT terminate on a momentary post-landing tilt unless it has
        # become so severe that recovery is physically impossible -- this gives
        # SAC enough horizon to actually learn to balance after impact.
        fallen          = pos[2] < self.FALLEN_Z
        severe_topple   = self.has_landed and (abs(orn[0]) > 2.0
                                               or abs(orn[1]) > 2.0)
        post_land_crash = severe_topple
        success         = self.has_landed and self.land_stable_steps >= 20

        reward = self._compute_reward(pos, orn, lin_vel, landed_now, success)
        self.prev_z = float(pos[2])
        self.prev_x = float(pos[0])

        if fallen or post_land_crash:
            reward -= 100.0

        terminated = bool(fallen or post_land_crash or success)
        truncated  = bool(self.step_counter >= self.max_steps)

        info = {
            "has_landed":        self.has_landed,
            "land_step":         self.land_step,
            "landing_pos":       self.landing_pos.tolist()
                                 if self.landing_pos is not None else None,
            "land_stable_steps": self.land_stable_steps,
            "success":           success,
            "fallen":            fallen,
            "post_land_crash":   post_land_crash,
            "crash":             bool(fallen or post_land_crash),
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        try:
            p.disconnect(self.physics_client)
        except Exception:
            pass
