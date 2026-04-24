"""
render_gifs.py - Produce one GIF per trained SAC configuration.

Loads each best-checkpoint, runs a deterministic rollout in headless DIRECT
mode, captures frames with PyBullet's offscreen camera (tracking the robot
base) and writes an animated GIF suitable for showing during evaluation.

Usage
-----
python render_gifs.py                       # default: render all 3 configs
python render_gifs.py --episodes 1          # one rollout per config
python render_gifs.py --width 640 --height 480 --fps 30
python render_gifs.py --only baseline,long_horizon_fast_updates
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pybullet as p
import imageio.v2 as imageio
from stable_baselines3 import SAC

from utils import BipedJumpEnv


# --------------------------------------------------------------------------- #
# Configuration registry  (must match run_experiments.py)
# --------------------------------------------------------------------------- #

CONFIGS = [
    {
        "name":        "baseline",
        "model_path":  os.path.join("models jump", "sac_baseline",
                                    "best", "best_model.zip"),
        "gif_path":    "demo_baseline.gif",
    },
    {
        "name":        "high_explore",
        "model_path":  os.path.join("models jump", "sac_high_explore",
                                    "best", "best_model.zip"),
        "gif_path":    "demo_high_explore.gif",
    },
    {
        "name":        "long_horizon_fast_updates",
        "model_path":  os.path.join("models jump", "sac_long_horizon_fast_updates",
                                    "best", "best_model.zip"),
        "gif_path":    "demo_long_horizon_fast_updates.gif",
    },
]


# --------------------------------------------------------------------------- #
# Frame capture helpers
# --------------------------------------------------------------------------- #

def _build_projection_matrix(width: int, height: int) -> List[float]:
    """Pinhole projection matched to camera-image aspect ratio."""
    aspect = float(width) / float(height)
    return p.computeProjectionMatrixFOV(
        fov=55.0, aspect=aspect, nearVal=0.05, farVal=20.0
    )


def _build_view_matrix(target_xyz: np.ndarray) -> List[float]:
    """Side-front tracking shot framed so both the 1 m platform and the
    full jump arc (spawn ~1.8 m, ground ~0.8 m) are visible.

    The camera target tracks the robot in x but is biased forward by 0.4 m
    so the action is centred in the frame, vertically anchored at 0.9 m
    (just above the platform top) and held 4 m away to keep the entire
    crouch-leap-descend-stabilise trajectory in shot.
    """
    cam_target = [float(target_xyz[0]) + 0.4,
                  float(target_xyz[1]),
                  0.9]
    cam_distance = 4.0
    cam_yaw      = 55.0
    cam_pitch    = -18.0
    cam_roll     = 0.0
    up_axis_index = 2
    return p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition = cam_target,
        distance             = cam_distance,
        yaw                  = cam_yaw,
        pitch                = cam_pitch,
        roll                 = cam_roll,
        upAxisIndex          = up_axis_index,
    )


def _capture_frame(env: BipedJumpEnv, width: int, height: int,
                   target_xyz: np.ndarray) -> np.ndarray:
    """Single RGB frame from PyBullet's offscreen renderer."""
    view_mat = _build_view_matrix(target_xyz)
    proj_mat = _build_projection_matrix(width, height)
    _, _, rgb, _, _ = p.getCameraImage(
        width            = width,
        height           = height,
        viewMatrix       = view_mat,
        projectionMatrix = proj_mat,
        renderer         = p.ER_BULLET_HARDWARE_OPENGL,
        flags            = p.ER_NO_SEGMENTATION_MASK,
        physicsClientId  = env.physics_client,
    )
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 4)
    return rgb[:, :, :3]                         # drop alpha for GIF


# --------------------------------------------------------------------------- #
# Rollout + GIF writer
# --------------------------------------------------------------------------- #

def render_one(model_path: str, gif_path: str, *,
               episodes: int = 1,
               width: int = 480,
               height: int = 360,
               fps: int = 25,
               sample_every: int = 2,
               max_steps: int = 500,
               seed: Optional[int] = 0) -> dict:
    """Roll the model out and write an animated GIF.

    Parameters
    ----------
    model_path   : path to the SB3 SAC zip checkpoint.
    gif_path     : output GIF file.
    episodes     : number of consecutive rollouts to concatenate into the GIF.
    width,height : frame resolution.
    fps          : playback frame-rate written into the GIF.
    sample_every : capture 1 frame every `sample_every` env steps (env runs
                   at 50 Hz, default 2 -> 25 fps real-time playback).
    max_steps    : hard cap per episode (env already enforces 500).
    seed         : reset seed for reproducibility (None -> random).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")

    env   = BipedJumpEnv(render=False)            # DIRECT mode + offscreen GL
    model = SAC.load(model_path, env=env)

    frames: List[np.ndarray] = []
    ep_summaries = []

    for ep in range(1, episodes + 1):
        reset_kwargs = {"seed": seed + ep - 1} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)

        # Always grab the first frame so the GIF starts on the spawn pose.
        base_pos = env.robot_current_position()
        frames.append(_capture_frame(env, width, height, base_pos))

        ep_reward, ep_steps = 0.0, 0
        terminated, truncated = False, False
        info = {}
        t0 = time.time()

        while not (terminated or truncated) and ep_steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps  += 1

            if (ep_steps % sample_every) == 0:
                base_pos = env.robot_current_position()
                frames.append(_capture_frame(env, width, height, base_pos))

        # Hold the last frame for ~0.4 s so the viewer can read the outcome.
        hold_frames = max(1, int(0.4 * fps))
        last = frames[-1]
        frames.extend([last] * hold_frames)

        landed = bool(info.get("has_landed", False))
        success = bool(info.get("success", False))
        ep_summaries.append({
            "episode": ep,
            "steps":   ep_steps,
            "reward":  ep_reward,
            "landed":  landed,
            "success": success,
            "wall_s":  time.time() - t0,
        })
        print(f"  ep {ep}: steps={ep_steps:3d}  reward={ep_reward:+8.2f}  "
              f"landed={landed}  success={success}  "
              f"wall={time.time() - t0:5.2f}s")

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)) or ".",
                exist_ok=True)
    duration_s = 1.0 / float(fps)
    imageio.mimsave(gif_path, frames, format="GIF",
                    duration=duration_s, loop=0)

    size_kb = os.path.getsize(gif_path) / 1024.0
    print(f"  GIF saved: {gif_path}  "
          f"({len(frames)} frames, {size_kb:.1f} KB, {fps} fps)")

    return {
        "gif_path":    gif_path,
        "n_frames":    len(frames),
        "size_kb":     size_kb,
        "episodes":    ep_summaries,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(
        description="Render demo GIFs for each trained SAC configuration.")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Episodes per GIF (concatenated, default 1).")
    ap.add_argument("--width",  type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps",    type=int, default=25,
                    help="Playback fps (env is 50 Hz).")
    ap.add_argument("--sample_every", type=int, default=2,
                    help="Capture 1 frame every N env steps "
                         "(default 2 -> 25 fps real-time).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Reset seed for reproducibility.")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated config names to render "
                         "(default: all). E.g. baseline,long_horizon_fast_updates")
    return ap.parse_args()


def main():
    args = parse_args()

    selected = CONFIGS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [c for c in CONFIGS if c["name"] in wanted]
        if not selected:
            print(f"[render_gifs] no configs matched --only={args.only!r}; "
                  f"available: {[c['name'] for c in CONFIGS]}",
                  file=sys.stderr)
            sys.exit(2)

    print(f"[render_gifs] rendering {len(selected)} config(s) "
          f"@ {args.width}x{args.height}, {args.fps} fps, "
          f"sample_every={args.sample_every}, episodes={args.episodes}")

    results = []
    for cfg in selected:
        print(f"\n[render_gifs] config = {cfg['name']}")
        print(f"  model    : {cfg['model_path']}")
        print(f"  gif_out  : {cfg['gif_path']}")
        if not os.path.isfile(cfg["model_path"]):
            print(f"  SKIP - checkpoint missing: {cfg['model_path']}")
            continue
        try:
            res = render_one(
                model_path   = cfg["model_path"],
                gif_path     = cfg["gif_path"],
                episodes     = args.episodes,
                width        = args.width,
                height       = args.height,
                fps          = args.fps,
                sample_every = args.sample_every,
                seed         = args.seed,
            )
            res["name"] = cfg["name"]
            results.append(res)
        except Exception as exc:
            print(f"  ERROR rendering {cfg['name']}: {exc}", file=sys.stderr)

    print("\n[render_gifs] DONE")
    for r in results:
        print(f"  {r['name']:32s} -> {r['gif_path']}  "
              f"({r['n_frames']} frames, {r['size_kb']:.1f} KB)")


if __name__ == "__main__":
    main()
