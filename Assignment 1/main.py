"""
==========================================================================
                    MAIN.PY - UR5 GRID NAVIGATION
==========================================================================
Students implement DP algorithms in utils.py and run this to see results.

Dependencies:
    - pybullet
    - numpy
    - utils.py

Usage:
    python main.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import time
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns


from utils import (
    GridEnv,
    policy_iteration,
    value_iteration,
    policy_evaluation,
    q_from_v,
    policy_improvement
)

# ==========================================================================
# VIDEO/GIF RECORDING SETUP
# ==========================================================================
frames = []  # Global list to store frames for GIF
RECORD_FPS = 20  # Frames per second for GIF
DELIVERABLES_PATH = os.path.join(os.path.dirname(__file__), 'deliverables')

# Ensure deliverables folder exists
if not os.path.exists(DELIVERABLES_PATH):
    os.makedirs(DELIVERABLES_PATH)
    print(f"Created deliverables folder: {DELIVERABLES_PATH}")


def save_frame():
    """Capture current frame from PyBullet camera for GIF creation."""
    try:
        width, height = 1280, 720
        
        # Get camera view matrix matching the visualizer
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=1.5,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100
        )
        
        img = p.getCameraImage(width, height, view_matrix, proj_matrix)
        if img is None or img[2] is None:
            return None
        rgb_array = np.array(img[2]).reshape(height, width, 4)[:, :, :3]  # RGB only
        
        image = Image.fromarray(rgb_array.astype(np.uint8))
        return image
    except:
        return None


def save_gif(filename, duration=50):
    """Save captured frames as GIF.
    
    Args:
        filename: Output GIF filename (full path)
        duration: Duration of each frame in milliseconds (50ms = 20fps)
    """
    global frames
    if len(frames) == 0:
        print("No frames to save!")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Save GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"\n[GIF SAVED] {filename}")
    print(f"   Frames: {len(frames)}")
    print(f"   Duration: {len(frames) * duration / 1000:.1f} seconds")


def generate_value_heatmap(V, rows, cols, title, filename, policy=None):
    """Generate a proper seaborn heatmap of the value function.
    
    Args:
        V: Value function array
        rows: Number of rows in grid
        cols: Number of columns in grid
        title: Title for the heatmap
        filename: Output filename (full path)
        policy: Optional policy array to show arrows
    """
    # Reshape V to grid
    V_grid = V.reshape(rows, cols)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(V_grid, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Value V(s)'}, ax=ax,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10, 'weight': 'bold'})
    
    # Mark start and goal
    ax.text(0.5, 0.5, 'S', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='lime',
            bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    ax.text(cols-0.5, rows-0.5, 'G', ha='center', va='center',
            fontsize=16, fontweight='bold', color='red',
            bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    # Add policy arrows if provided
    if policy is not None:
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                if state != len(V) - 1:  # Not goal
                    action = policy[state]
                    ax.text(j+0.5, i+0.75, action_symbols[action], 
                           ha='center', va='center', fontsize=14, color='white')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[HEATMAP SAVED] {filename}")
    plt.close()


def generate_comparison_heatmap(V1, V2, rows, cols, title1, title2, filename):
    """Generate side-by-side heatmaps comparing two value functions.
    
    Args:
        V1, V2: Two value function arrays to compare
        rows, cols: Grid dimensions
        title1, title2: Titles for each heatmap
        filename: Output filename
    """
    V1_grid = V1.reshape(rows, cols)
    V2_grid = V2.reshape(rows, cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First heatmap
    sns.heatmap(V1_grid, annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'Value V(s)'}, ax=ax1,
                linewidths=0.5, linecolor='white')
    ax1.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=14, color='lime', fontweight='bold')
    ax1.text(cols-0.5, rows-0.5, 'G', ha='center', va='center', fontsize=14, color='red', fontweight='bold')
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Second heatmap
    sns.heatmap(V2_grid, annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'Value V(s)'}, ax=ax2,
                linewidths=0.5, linecolor='white')
    ax2.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=14, color='lime', fontweight='bold')
    ax2.text(cols-0.5, rows-0.5, 'G', ha='center', va='center', fontsize=14, color='red', fontweight='bold')
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[COMPARISON SAVED] {filename}")
    plt.close()


def clear_frames():
    """Clear frame buffer."""
    global frames
    frames = []


def state_to_position(state, rows, cols, grid_size=0.10, 
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols
    
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    
    return [x, y, z]



def get_m_shape_cells():

    m_cells = set()

    # Left vertical leg (cols 0-1, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 0))
        m_cells.add((row, 1))

    # Right vertical leg (cols 7-8, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 7))
        m_cells.add((row, 8))

    # Left diagonal (from top-left going down to center)
    # Row 5: extend to col 2
    m_cells.add((5, 2))
    # Row 4: cols 2, 3
    m_cells.add((4, 2))
    m_cells.add((4, 3))
    # Row 3: cols 2, 3, 4 (connecting to middle)
    m_cells.add((3, 2))
    m_cells.add((3, 3))
    m_cells.add((3, 4))
    # Row 2: col 4 (bottom of V)
    m_cells.add((2, 4))

    # Right diagonal (from top-right going down to center)
    # Row 5: extend to col 6
    m_cells.add((5, 6))
    # Row 4: cols 5, 6
    m_cells.add((4, 5))
    m_cells.add((4, 6))
    # Row 3: cols 4, 5, 6 (connecting to middle - 4 already added)
    m_cells.add((3, 5))
    m_cells.add((3, 6))

    # Middle vertical extension (above the V bottom)
    m_cells.add((4, 4))  # extends middle upward
    m_cells.add((5, 4))  # continues up
    m_cells.add((6, 4))  # top of middle extension

    return m_cells


def cell_to_position(row, col, rows, cols, grid_size=0.10,
                     table_center=[0, -0.3, 0.65], z_offset=0.10):
    """Convert (row, col) to world position."""
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    return [x, y, z]


def draw_m_grid(grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw the M-shaped grid with individual blocks.
    Returns the valid cells, start cell, and end cell.
    """
    m_cells = get_m_shape_cells()
    rows, cols = 7, 9  # Grid dimensions for the M shape

    line_color = [0, 0, 0]  # Black borders
    line_width = 2
    z = table_center[2] + 0.001
    half = grid_size / 2

    # Draw each M-shaped cell as a block
    for (row, col) in m_cells:
        pos = cell_to_position(row, col, rows, cols, grid_size, table_center, z_offset=0.001)
        x, y = pos[0], pos[1]

        # Draw filled square (4 border lines)
        p.addUserDebugLine([x-half, y-half, z], [x+half, y-half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y-half, z], [x+half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y+half, z], [x-half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x-half, y+half, z], [x-half, y-half, z], line_color, line_width)

    # Define start and end cells
    start_cell = (0, 0)   # Bottom-left of M
    end_cell = (0, 8)     # Bottom-right of M

    # Draw start marker (green)
    start_pos = cell_to_position(start_cell[0], start_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    marker_half = half * 0.6
    green = [0, 1, 0]
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugText("START", [start_pos[0], start_pos[1], start_pos[2] + 0.05], green, 1.0)

    # Draw end marker (red)
    end_pos = cell_to_position(end_cell[0], end_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugText("END", [end_pos[0], end_pos[1], end_pos[2] + 0.05], red, 1.0)



    return m_cells, start_cell, end_cell


def draw_grid(rows, cols, grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw a regular grid for the DP algorithm visualization.
    Returns grid boundaries.
    """
    line_color = [0.3, 0.3, 0.3]  # Dark gray
    line_width = 2
    z = table_center[2] + 0.001
    
    # Calculate grid boundaries
    x_start = table_center[0] - (cols * grid_size) / 2
    y_start = table_center[1] - (rows * grid_size) / 2
    
    # Draw horizontal lines
    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_start + cols * grid_size, y, z], line_color, line_width)
    
    # Draw vertical lines
    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_start + rows * grid_size, z], line_color, line_width)
    
    # Draw start marker (green) - top-left corner (state 0)
    start_row, start_col = 0, 0
    start_x = x_start + (start_col + 0.5) * grid_size
    start_y = y_start + (start_row + 0.5) * grid_size
    marker_half = grid_size * 0.3
    green = [0, 1, 0]
    p.addUserDebugLine([start_x-marker_half, start_y-marker_half, z+0.01],
                       [start_x+marker_half, start_y-marker_half, z+0.01], green, 4, 0)
    p.addUserDebugLine([start_x+marker_half, start_y-marker_half, z+0.01],
                       [start_x+marker_half, start_y+marker_half, z+0.01], green, 4, 0)
    p.addUserDebugLine([start_x+marker_half, start_y+marker_half, z+0.01],
                       [start_x-marker_half, start_y+marker_half, z+0.01], green, 4, 0)
    p.addUserDebugLine([start_x-marker_half, start_y+marker_half, z+0.01],
                       [start_x-marker_half, start_y-marker_half, z+0.01], green, 4, 0)
    p.addUserDebugText("START", [start_x, start_y, z + 0.05], green, 1.0)
    
    # Draw goal marker (red) - bottom-right corner (state rows*cols-1)
    goal_row, goal_col = rows - 1, cols - 1
    goal_x = x_start + (goal_col + 0.5) * grid_size
    goal_y = y_start + (goal_row + 0.5) * grid_size
    red = [1, 0, 0]
    p.addUserDebugLine([goal_x-marker_half, goal_y-marker_half, z+0.01],
                       [goal_x+marker_half, goal_y-marker_half, z+0.01], red, 4, 0)
    p.addUserDebugLine([goal_x+marker_half, goal_y-marker_half, z+0.01],
                       [goal_x+marker_half, goal_y+marker_half, z+0.01], red, 4, 0)
    p.addUserDebugLine([goal_x+marker_half, goal_y+marker_half, z+0.01],
                       [goal_x-marker_half, goal_y+marker_half, z+0.01], red, 4, 0)
    p.addUserDebugLine([goal_x-marker_half, goal_y+marker_half, z+0.01],
                       [goal_x-marker_half, goal_y-marker_half, z+0.01], red, 4, 0)
    p.addUserDebugText("GOAL", [goal_x, goal_y, z + 0.05], red, 1.0)


def display_value_function(V, rows, cols, grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Display the value function as text on the grid.
    """
    z = table_center[2] + 0.05
    x_start = table_center[0] - (cols * grid_size) / 2
    y_start = table_center[1] - (rows * grid_size) / 2
    
    text_color = [0, 0, 0]  # Black
    
    for state in range(len(V)):
        row = state // cols
        col = state % cols
        
        x = x_start + (col + 0.5) * grid_size
        y = y_start + (row + 0.5) * grid_size
        
        # Display rounded value
        value_text = f"{V[state]:.1f}"
        p.addUserDebugText(value_text, [x-0.02, y, z], text_color, 0.8)


def move_robot_to_position(ur5_id, target_pos, target_orn=None, duration=2.0, record=False):
    """
    Move robot end-effector to target position using Inverse Kinematics.
    
    Args:
        ur5_id: PyBullet body ID of the robot
        target_pos: Target position [x, y, z]
        target_orn: Target orientation as quaternion (optional)
        duration: Time to take for movement (in seconds)
        record: Whether to capture frames for GIF (default: False)
    """
    global frames
    
    if target_orn is None:
        # Default downward facing orientation
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    # Get number of joints
    num_joints = p.getNumJoints(ur5_id)
    
    # Find end effector link index (usually the last link for UR5)
    end_effector_index = num_joints - 1
    
    # Calculate IK
    joint_poses = p.calculateInverseKinematics(
        ur5_id,
        end_effector_index,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    # Set joint positions with position control
    joint_index = 0
    for i in range(num_joints):
        joint_info = p.getJointInfo(ur5_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                ur5_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[joint_index],
                force=500,
                maxVelocity=1.0
            )
            joint_index += 1
    
    # Step simulation to execute movement
    steps = int(duration * 240)  # 240 Hz simulation
    frame_interval = max(1, steps // (duration * RECORD_FPS)) if record else steps + 1
    
    for step in range(steps):
        p.stepSimulation()
        
        # Capture frame if recording
        if record and step % frame_interval == 0:
            frame = save_frame()
            if frame is not None:
                frames.append(frame)
        
        time.sleep(1./240.)


if __name__ == "__main__":
    

    ROWS = 5
    COLS = 6
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    THETA = 1e-8
    
    # Create environment
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL)

    # Run Policy Iteration
    print("\n" + "="*60)
    print("Running Policy Iteration...")
    print("="*60)
    start_time = time.time()
    pi_policy, pi_V = policy_iteration(env, gamma=GAMMA, theta=THETA)
    pi_time = time.time() - start_time
    
    # Get optimal path from Policy Iteration
    pi_path = env.get_optimal_path(pi_policy)
    
    print(f"Policy Iteration completed in {pi_time:.4f} seconds")
    print(f"Optimal path length: {len(pi_path)} states")
    print(f"Path: {pi_path}")
    print(f"V(start) = {pi_V[START]:.2f}")
    print(f"V(goal) = {pi_V[GOAL]:.2f}")
    
    # Run Value Iteration
    print("\n" + "="*60)
    print("Running Value Iteration...")
    print("="*60)
    start_time = time.time()
    vi_policy, vi_V = value_iteration(env, gamma=GAMMA, theta=THETA)
    vi_time = time.time() - start_time
    
    # Get optimal path from Value Iteration
    vi_path = env.get_optimal_path(vi_policy)
    
    print(f"Value Iteration completed in {vi_time:.4f} seconds")
    print(f"Optimal path length: {len(vi_path)} states")
    print(f"Path: {vi_path}")
    print(f"V(start) = {vi_V[START]:.2f}")
    print(f"V(goal) = {vi_V[GOAL]:.2f}")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"Policy Iteration time: {pi_time:.4f}s")
    print(f"Value Iteration time: {vi_time:.4f}s")
    print(f"Both paths have same length: {len(pi_path) == len(vi_path)}")
    print(f"Policies are identical: {np.array_equal(pi_policy, vi_policy)}")
    print("="*60)

    # ========================================================================
    # GENERATE SEABORN HEATMAPS
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING SEABORN HEATMAPS")
    print("="*60)
    
    # Generate heatmaps for both algorithms
    generate_value_heatmap(
        pi_V, ROWS, COLS, 
        'Policy Iteration: State-Value Function V(s)', 
        os.path.join(DELIVERABLES_PATH, 'heatmap_policy_iteration.png'),
        policy=pi_policy
    )
    
    generate_value_heatmap(
        vi_V, ROWS, COLS,
        'Value Iteration: State-Value Function V(s)',
        os.path.join(DELIVERABLES_PATH, 'heatmap_value_iteration.png'),
        policy=vi_policy
    )
    
    # Generate comparison heatmap
    generate_comparison_heatmap(
        pi_V, vi_V, ROWS, COLS,
        'Policy Iteration', 'Value Iteration',
        os.path.join(DELIVERABLES_PATH, 'heatmap_comparison.png')
    )
    
    print("="*60)

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    p.loadURDF("plane.urdf")
    
    table_path = os.path.join("assest", "table", "table.urdf")
    p.loadURDF(table_path, [0, -0.3, 0], globalScaling=2.0)
    
    stand_path = os.path.join("assest", "robot_stand.urdf")
    p.loadURDF(stand_path, [0, -0.8, 0], useFixedBase=True)
    
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    sys.stderr = old_stderr

    # Draw the grid for visualization
    draw_grid(ROWS, COLS, grid_size=0.10, table_center=[0, -0.3, 0.65])
    
    # Display value function as heatmap text
    display_value_function(vi_V, ROWS, COLS, grid_size=0.10, table_center=[0, -0.3, 0.65])
    
    print("\n" + "="*60)
    print("SIMULATION WITH VIDEO RECORDING")
    print("="*60)
    print(f"GIF will be saved to: {DELIVERABLES_PATH}")
    print("="*60)
    
    print("\nInitializing robot position...")
    
    # Capture initial frame
    frame = save_frame()
    if frame is not None:
        frames.append(frame)
    
    # Move robot to start position first (with recording)
    start_pos = state_to_position(START, ROWS, COLS, grid_size=0.10, 
                                   table_center=[0, -0.3, 0.65], z_offset=0.20)
    move_robot_to_position(ur5_id, start_pos, duration=3.0, record=True)
    
    print(f"Robot at start position: {start_pos}")
    print("Starting path execution in 2 seconds...")
    time.sleep(2)
    
    # Execute the optimal path using Value Iteration results
    optimal_path = vi_path
    optimal_policy = vi_policy
    
    print(f"\nExecuting optimal path with {len(optimal_path)} waypoints...")
    
    # Store previous position for drawing trail
    prev_pos = None
    trail_color = [0, 1, 0]  # Green
    trail_width = 3
    
    # Move through each state in the path (with recording)
    for i, state in enumerate(optimal_path):
        # Convert state to world position
        target_pos = state_to_position(state, ROWS, COLS, grid_size=0.10,
                                        table_center=[0, -0.3, 0.65], z_offset=0.20)
        
        print(f"Step {i+1}/{len(optimal_path)}: State {state}")
        
        # Move robot to position (with recording)
        move_robot_to_position(ur5_id, target_pos, duration=1.0, record=True)
        
        # Draw trail (green line from previous to current position)
        if prev_pos is not None:
            # Draw at end-effector height
            p.addUserDebugLine(prev_pos, target_pos, trail_color, trail_width, lifeTime=0)
        
        prev_pos = target_pos.copy()
        
        # Small pause between steps
        time.sleep(0.2)
    
    print("\n" + "="*60)
    print("Path execution complete!")
    print(f"Final position: State {GOAL} (Goal)")
    print("="*60)
    
    # Display policy arrows on grid (BEFORE final frames to ensure they're visible)
    z_display = 0.65 + 0.002
    x_start = - (COLS * 0.10) / 2
    y_start = -0.3 - (ROWS * 0.10) / 2
    arrow_color = [0, 0, 1]  # Blue
    
    for state in range(env.nS):
        if state == GOAL:
            continue
            
        row = state // COLS
        col = state % COLS
        
        x = x_start + (col + 0.5) * 0.10
        y = y_start + (row + 0.5) * 0.10
        
        action = optimal_policy[state]
        
        # Draw arrow based on action
        arrow_length = 0.03
        if action == 0:  # LEFT
            p.addUserDebugLine([x, y, z_display], [x - arrow_length, y, z_display], arrow_color, 2)
        elif action == 1:  # DOWN
            p.addUserDebugLine([x, y, z_display], [x, y + arrow_length, z_display], arrow_color, 2)
        elif action == 2:  # RIGHT
            p.addUserDebugLine([x, y, z_display], [x + arrow_length, y, z_display], arrow_color, 2)
        elif action == 3:  # UP
            p.addUserDebugLine([x, y, z_display], [x, y - arrow_length, z_display], arrow_color, 2)
    
    print("Policy arrows displayed on grid (Blue = optimal action)")
    
    # Capture final frames with arrows visible
    print("\n  Capturing final frames...")
    for _ in range(15):
        frame = save_frame()
        if frame is not None:
            frames.append(frame)
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Save GIF to deliverables folder
    gif_path = os.path.join(DELIVERABLES_PATH, 'simulation_demo.gif')
    save_gif(gif_path, duration=50)  # 50ms per frame = 20 FPS
    
    print("\n" + "="*60)
    print("DELIVERABLES SAVED - PARTS 1-6 COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {DELIVERABLES_PATH}")
    print("\nGenerated files:")
    print("  1. simulation_demo.gif - Video recording of robot navigation")
    print("  2. heatmap_policy_iteration.png - Seaborn heatmap (PI)")
    print("  3. heatmap_value_iteration.png - Seaborn heatmap (VI)")
    print("  4. heatmap_comparison.png - Side-by-side comparison")
    print("  5. analysis_report.txt - Detailed analysis")
    print("  6. comparison_table.txt - Performance comparison")
    print("\nAssignment Parts Demonstrated:")
    print("  - Part 1: Policy Evaluation (in analysis.py)")
    print("  - Part 2: Q-value Computation (in analysis.py)")
    print("  - Part 3: Policy Improvement (in analysis.py)")
    print("  - Part 4: Policy Iteration (this simulation)")
    print("  - Part 5: Value Iteration (this simulation)")
    print("  - Part 6: Unseen environments (in analysis.py)")
    print("="*60)
    
    print("\nSimulation running. Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nSimulation ended.")
    except Exception as e:
        print(f"\nError: {e}")
