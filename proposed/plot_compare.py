"""
Plot proposed (Local Spatial Self-Attention) vs MAPPO baseline reward curves.

- Proposed curve: parsed from proposed/train_full.log
  (lines: "eval average episode rewards of environment: <value>")
- MAPPO baseline curve: parsed from a tensorboard-style summary.json
  (key contains "eval_episode_rewards", value = list of [timestamp, step, value])

Output: proposed/compare_reward.png
"""
import os
import re
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_STEPS = 100000


def ema(values, alpha=0.1):
    """Exponential moving average for smoothing noisy reward curves."""
    if len(values) == 0:
        return values
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def load_proposed(log_path):
    """Return (steps, rewards) for the proposed run from train_full.log."""
    pat = re.compile(r"eval average episode rewards of environment:\s*([-+0-9.eE]+)")
    ys = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ys.append(float(m.group(1)))
    ys = np.asarray(ys, dtype=float)
    xs = np.linspace(0, TOTAL_STEPS, len(ys)) if len(ys) else ys
    return xs, ys


def find_eval_series(summary):
    """Find the eval_episode_rewards key in a summary.json dict."""
    for k, v in summary.items():
        if "eval_episode_rewards" in k:
            return v
    return None


def load_mappo(candidate_paths):
    """Return (steps, rewards) for the MAPPO baseline from the first
    summary.json that contains an eval_episode_rewards series."""
    for p in candidate_paths:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                summary = json.load(f)
        except Exception:
            continue
        series = find_eval_series(summary)
        if not series:
            continue
        arr = np.asarray(series, dtype=float)  # columns: ts, step, value
        steps = arr[:, 1]
        vals = arr[:, 2]
        # keep only the first pass through steps (avoid accumulated re-runs)
        cut = len(steps)
        for i in range(1, len(steps)):
            if steps[i] < steps[i - 1]:
                cut = i
                break
        return steps[:cut], vals[:cut], p
    return None, None, None


def main():
    prop_log = os.path.join(HERE, "train_full.log")
    px, py = load_proposed(prop_log)

    # The genuine Bahdanau baseline series is labeled "MUAV_MAPPO" and lives in
    # this summary.json (eval ends ~700-730). Prefer it over any stray file.
    mappo_candidates = [
        os.path.join(HERE, "results_proposed", "results_a85_rnn", "logs", "summary.json"),
        os.path.join(HERE, "..", "MAPPO", "results", "logs", "summary.json"),
    ]
    mx, my, mappo_src = load_mappo(mappo_candidates)

    print("proposed points:", len(py), "-> mean last 10:",
          float(np.mean(py[-10:])) if len(py) else "n/a")
    if my is not None:
        print("mappo points:", len(my), "from", mappo_src,
              "-> mean last 10:", float(np.mean(my[-10:])))
    else:
        print("MAPPO baseline not found in candidates.")

    plt.figure(figsize=(9, 5.5))

    # MAPPO baseline
    if my is not None:
        plt.plot(mx, my, color="#c0c0c0", linewidth=0.8, alpha=0.6)
        plt.plot(mx, ema(my, 0.05), color="#1f77b4", linewidth=2.2,
                 label="MAPPO (Bahdanau attention)")

    # Proposed
    if len(py):
        plt.plot(px, py, color="#f0c0c0", linewidth=0.8, alpha=0.6)
        plt.plot(px, ema(py, 0.1), color="#d62728", linewidth=2.2,
                 label="Proposed (Local Spatial Self-Attention)")

    plt.xlabel("Training steps")
    plt.ylabel("Average episode reward (eval)")
    plt.title("Reward curve: Proposed vs MAPPO baseline")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(HERE, "compare_reward.png")
    plt.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
