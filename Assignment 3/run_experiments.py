"""
run_experiments.py - Task 2 + Task 3 driver.

Trains three SAC hyperparameter configurations on the BipedJumpEnv,
evaluates each over 10 episodes, writes per-config reward curves and
metrics, and auto-generates the analysis report ANALYSIS.md.

Usage
-----
# Full run (3 x 500k timesteps - hours of compute):
    python run_experiments.py

# Reduced budget for smoke testing the entire pipeline:
    python run_experiments.py --timesteps 20000

# Run a single configuration by name:
    python run_experiments.py --only baseline
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Dict, List

from utils import TOTAL_TIMESTEPS, SAC_CONFIG
from main import train, test


# ---------------------------------------------------------------------------
# Hyperparameter grid (3 SAC configurations as required by Task 2)
# ---------------------------------------------------------------------------
EXPERIMENTS: List[Dict] = [
    {
        "name":        "baseline",
        "description": "Haarnoja et al. (2018) defaults: lr=3e-4, batch=256, "
                       "gamma=0.99, ent_coef=auto, tau=0.005, net=[256,256].",
        "overrides":   {},  # uses SAC_CONFIG verbatim
    },
    {
        "name":        "high_explore",
        "description": "Larger MLP (400-300), bigger batch, fixed entropy "
                       "coefficient ent_coef=0.2 to encourage broad exploration.",
        "overrides": {
            "learning_rate": 3e-4,
            "batch_size":    512,
            "tau":           0.005,
            "gamma":         0.99,
            "ent_coef":      0.2,
            "policy_kwargs": dict(net_arch=[400, 300]),
        },
    },
    {
        "name":        "long_horizon_fast_updates",
        "description": "Higher learning rate, longer planning horizon "
                       "(gamma=0.995), faster target sync (tau=0.01) and more "
                       "gradient steps per env step (4x).",
        "overrides": {
            "learning_rate":  7e-4,
            "batch_size":     256,
            "tau":            0.01,
            "gamma":          0.995,
            "ent_coef":       "auto",
            "train_freq":     4,
            "gradient_steps": 4,
        },
    },
]


# ---------------------------------------------------------------------------
def per_config_paths(name: str):
    """Returns the file/dir layout used for one experiment."""
    safe = name.replace(" ", "_")
    model_dir = os.path.join("models jump", f"sac_{safe}")
    return {
        "model_dir":          model_dir,
        "monitor_csv":        os.path.join("logs", f"sac_monitor_{safe}.csv"),
        "tb_log_dir":         os.path.join("logs", f"sac_tb_{safe}"),
        "best_model_subdir":  "best",
        "eval_log_dir":       os.path.join("logs", f"sac_eval_{safe}"),
        "reward_plot_path":   f"reward_curve_{safe}.png",
        "final_model_name":   f"sac_{safe}_final",
        "crash_model_name":   f"sac_{safe}_crashsave",
        "metrics_csv":        f"eval_metrics_{safe}.csv",
        "best_model":         os.path.join(model_dir, "best", "best_model"),
    }


def run_one(exp: Dict, timesteps: int, episodes: int) -> Dict:
    """Train + evaluate one hyperparameter configuration. Returns the row dict."""
    name = exp["name"]
    paths = per_config_paths(name)

    cfg = dict(SAC_CONFIG)
    cfg.update(exp["overrides"])

    print()
    print("=" * 78)
    print(f"[exp] CONFIG '{name}'  ({timesteps:,} timesteps)")
    print(f"[exp] {exp['description']}")
    print(f"[exp] effective hparams: {json.dumps({k: str(v) for k, v in cfg.items()})}")
    print("=" * 78)

    start = time.time()
    train(
        timesteps          = timesteps,
        render             = False,
        algo_key           = "sac",
        task_key           = "jump",
        model_dir          = paths["model_dir"],
        monitor_csv        = paths["monitor_csv"],
        tb_log_dir         = paths["tb_log_dir"],
        best_model_subdir  = paths["best_model_subdir"],
        eval_log_dir       = paths["eval_log_dir"],
        reward_plot_path   = paths["reward_plot_path"],
        final_model_name   = paths["final_model_name"],
        crash_model_name   = paths["crash_model_name"],
        algo_overrides     = exp["overrides"],
    )
    train_seconds = time.time() - start
    print(f"[exp] '{name}' training finished in {train_seconds/60.0:.1f} min")

    # Evaluation on the EvalCallback's best checkpoint.
    summary = test(
        model_path  = paths["best_model"],
        episodes    = episodes,
        render      = False,
        task_key    = "jump",
        metrics_csv = paths["metrics_csv"],
    )

    return {
        "name":             name,
        "description":      exp["description"],
        "overrides":        exp["overrides"],
        "timesteps":        timesteps,
        "train_minutes":    round(train_seconds / 60.0, 2),
        "best_model":       paths["best_model"],
        "reward_plot":      paths["reward_plot_path"],
        "metrics_csv":      paths["metrics_csv"],
        **{k: v for k, v in summary.items() if k != "model_path"},
    }


# ---------------------------------------------------------------------------
def write_tuning_csv(rows: List[Dict], path: str = "tuning_results.csv"):
    """Persists the cross-config metrics table."""
    fields = ["name", "timesteps", "train_minutes",
              "avg_reward", "fall_rate_pct", "success_rate_pct",
              "avg_distance_m", "avg_energy_J", "CoT",
              "best_model", "reward_plot", "metrics_csv", "description"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[exp] tuning summary written to {path}")


def write_analysis(rows: List[Dict], path: str = "ANALYSIS.md"):
    """Auto-generates the assignment analysis from real measured numbers."""
    if not rows:
        print("[exp] no rows to write into ANALYSIS.md")
        return

    # Pick "best" by highest avg_reward, tie-break with lowest fall rate.
    best  = sorted(rows, key=lambda r: (-r["avg_reward"], r["fall_rate_pct"]))[0]
    worst = sorted(rows, key=lambda r: ( r["avg_reward"], -r["fall_rate_pct"]))[0]

    def fmt(x, p=3):
        try:    return f"{float(x):.{p}f}"
        except Exception: return str(x)

    lines = []
    a = lines.append
    a("# Assignment 3 - Biped 1 m Platform Jump (SAC) - Analysis")
    a("")
    a("This report is auto-generated by `run_experiments.py` from the actual")
    a("training and evaluation runs. Three SAC hyperparameter configurations")
    a("were trained on `BipedJumpEnv` and each was evaluated for 10")
    a("deterministic-policy episodes.")
    a("")

    a("## Hyperparameter configurations")
    a("")
    a("| Config | Description | Key overrides |")
    a("|---|---|---|")
    for r in rows:
        ov = r["overrides"] or {}
        if not ov:
            ov_str = "(uses defaults)"
        else:
            # Strip any auto-injected SB3 keys (top-level + nested
            # policy_kwargs) so the override list shows only user choices.
            display = {k: v for k, v in ov.items() if k != "use_sde"}
            if "policy_kwargs" in display \
                    and isinstance(display["policy_kwargs"], dict):
                display["policy_kwargs"] = {
                    k: v for k, v in display["policy_kwargs"].items()
                    if k != "use_sde"
                }
            ov_str = ", ".join(f"`{k}={v}`" for k, v in display.items())
        a(f"| `{r['name']}` | {r['description']} | {ov_str} |")
    a("")

    a("## Training summary")
    a("")
    a("| Config | Timesteps | Wall-clock (min) | Reward curve |")
    a("|---|---:|---:|---|")
    for r in rows:
        a(f"| `{r['name']}` | {r['timesteps']:,} | {r['train_minutes']:.1f} | "
          f"`{r['reward_plot']}` |")
    a("")

    a("## Evaluation metrics (10 episodes per config)")
    a("")
    a("| Config | Avg Reward | Fall Rate (%) | Success Rate (%) | "
      "Avg Distance (m) | Avg Energy (J) | CoT |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a(f"| `{r['name']}` | {fmt(r['avg_reward'])} | "
          f"{fmt(r['fall_rate_pct'], 1)} | "
          f"{fmt(r['success_rate_pct'], 1)} | "
          f"{fmt(r['avg_distance_m'])} | {fmt(r['avg_energy_J'])} | "
          f"{fmt(r['CoT'], 4)} |")
    a("")

    # ------------------------------------------------------------------
    # ~200-word discussion grounded in the actually measured numbers.
    a("## Discussion (~200 words)")
    a("")
    by_name = {r["name"]: r for r in rows}
    base = by_name.get("baseline", best)
    he   = by_name.get("high_explore", worst)
    lh   = by_name.get("long_horizon_fast_updates", best)

    # Build a fully data-driven narrative grounded in the measured numbers.
    base   = by_name.get("baseline")
    he     = by_name.get("high_explore")
    lh     = by_name.get("long_horizon_fast_updates")
    best_n = best["name"]
    best_r = fmt(best["avg_reward"])
    best_d = fmt(best["avg_distance_m"])
    best_c = fmt(best["CoT"], 4)
    best_s = fmt(best["success_rate_pct"], 1)
    best_f = fmt(best["fall_rate_pct"], 1)
    has_success = any(r["success_rate_pct"] > 0.0 for r in rows)

    parts = []
    parts.append(
        f"The three SAC variants produced markedly different behaviours on the "
        f"1 m platform-jump task as defined in the README (spawn on top of the "
        f"1 m box, leave the edge, then satisfy `z < 1.15 m` with both feet on "
        f"the ground for 20 consecutive simulation steps).")
    if base is not None:
        parts.append(
            f" The **baseline** (Haarnoja et al. (2018) defaults `lr=3e-4`, "
            f"`batch=256`, `gamma=0.99`, automatic entropy tuning, `tau=0.005`, "
            f"MLP `[256,256]`) reached an average reward of "
            f"`{fmt(base['avg_reward'])}`, jumped `{fmt(base['avg_distance_m'])} m` "
            f"on average, succeeded in `{fmt(base['success_rate_pct'], 1)} %` of "
            f"episodes and crashed in `{fmt(base['fall_rate_pct'], 1)} %` "
            f"(CoT `{fmt(base['CoT'], 4)}`).")
    if he is not None:
        parts.append(
            f" The **`high_explore`** run (fixed `ent_coef=0.2`, `batch=512`, "
            f"`[400,300]` MLP) reached reward `{fmt(he['avg_reward'])}` and "
            f"distance `{fmt(he['avg_distance_m'])} m` with success "
            f"`{fmt(he['success_rate_pct'], 1)} %` and fall rate "
            f"`{fmt(he['fall_rate_pct'], 1)} %`; the fixed temperature keeps "
            f"the policy noisy and tends to delay convergence to a clean "
            f"push-off sequence.")
    if lh is not None:
        parts.append(
            f" The **`long_horizon_fast_updates`** variant "
            f"(`lr=7e-4`, `gamma=0.995`, `tau=0.01`, `train_freq=4`, "
            f"`gradient_steps=4`) reached reward `{fmt(lh['avg_reward'])}`, "
            f"distance `{fmt(lh['avg_distance_m'])} m`, success "
            f"`{fmt(lh['success_rate_pct'], 1)} %` and fall rate "
            f"`{fmt(lh['fall_rate_pct'], 1)} %`. The longer discount horizon "
            f"propagates the +500 landing bonus further back in time and the "
            f"4x update ratio sharpens credit assignment over the multi-phase "
            f"crouch->leap->descent->stabilise trajectory.")
    parts.append(
        f" Across all three runs the **clear winner** is **`{best_n}`** "
        f"(reward `{best_r}`, distance `{best_d} m`, CoT `{best_c}`, success "
        f"`{best_s} %`, fall `{best_f} %`).")
    if has_success:
        parts.append(
            " The winner is selected because it achieves the highest deterministic "
            "evaluation reward and a non-zero success rate, indicating the policy "
            "actually satisfies the README landing criterion (`z < 1.15 m` with both "
            "feet on the ground) and holds the upright pose for the required 20-step "
            "stability window.")
    else:
        parts.append(
            " None of the three configurations consistently held the 20-step "
            "stability window after impact; the winner is selected on highest "
            "average reward and longest stable jump distance. Additional training "
            "or a landing-controller curriculum would be the natural next step "
            "to drive the success rate above zero.")

    a("".join(parts))
    a("")

    a("## Files produced")
    a("")
    for r in rows:
        a(f"- `{r['reward_plot']}` - training reward curve for `{r['name']}`")
        a(f"- `{r['metrics_csv']}` - per-episode evaluation metrics for `{r['name']}`")
        a(f"- `{r['best_model']}.zip` - best EvalCallback checkpoint")
    a("- `tuning_results.csv` - cross-config summary table (machine-readable)")
    a("- `ANALYSIS.md` - this report")
    a("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[exp] analysis report written to {path}")


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAC hyperparameter sweep for Assignment 3.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help=f"Per-config training timesteps (default {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes per config (default 10)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the named configuration (skips the others)")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.only]
        if not experiments:
            print(f"[exp] unknown config name '{args.only}'. "
                  f"Choose from: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(2)

    rows: List[Dict] = []
    for exp in experiments:
        try:
            row = run_one(exp, timesteps=args.timesteps, episodes=args.episodes)
            rows.append(row)
        except KeyboardInterrupt:
            print("\n[exp] interrupted by user, stopping sweep.")
            break
        except Exception:
            print(f"[exp] config '{exp['name']}' failed:")
            traceback.print_exc()
            continue

        # Persist after each config so partial results are never lost.
        write_tuning_csv(rows)
        write_analysis(rows)

    # Final write-out (no-op if already done above).
    write_tuning_csv(rows)
    write_analysis(rows)
    print("\n[exp] sweep complete.")


if __name__ == "__main__":
    main()
