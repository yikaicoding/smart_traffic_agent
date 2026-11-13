"""
train.py - Train the Q-learning policy and record learning curves.

Outputs
  outputs/q_table.npz            learned policy (long-term memory)
  outputs/training_log.csv       per-episode reward, epsilon, TD error
  outputs/validation_log.csv     greedy-policy validation every 20 episodes
  outputs/hyperparameter_sweep.csv  alpha / gamma comparison
  figures/learning_curve.png, figures/hyperparameter_sweep.png
Run:  python train.py
"""
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import TrafficAgent
from env import Junction
from perception import SensorArray
from q_agent import QAgent

VALIDATION_HOURS = [8, 12, 18]          # morning peak, off-peak, evening peak
VALIDATION_SEED = 7
VALIDATE_EVERY = 20


def load_cfg():
    with open("config.json") as f:
        return json.load(f)


def validate(cfg, agent):
    """Greedy policy (no exploration, no learning) on fixed validation hours."""
    waits = []
    for h in VALIDATION_HOURS:
        rng = np.random.default_rng(VALIDATION_SEED + h)
        ctrl = TrafficAgent(cfg, "ql", agent, train=False)
        m = ctrl.run(Junction(cfg, rng), SensorArray(cfg, rng), h * 3600, (h + 1) * 3600)
        waits.append(m["avg_wait_s"])
    return float(np.mean(waits))


def train(cfg, alpha=None, gamma=None, episodes=None, verbose=True):
    q = cfg["q_learning"]
    alpha = q["learning_rate_alpha"] if alpha is None else alpha
    gamma = q["discount_gamma"] if gamma is None else gamma
    episodes = q["episodes"] if episodes is None else episodes
    cfg = json.loads(json.dumps(cfg))
    cfg["q_learning"]["learning_rate_alpha"], cfg["q_learning"]["discount_gamma"] = alpha, gamma

    rng = np.random.default_rng(q["seed"])
    agent = QAgent(cfg, rng)
    sim = cfg["simulation"]
    ep_len = sim["training_episode_length_s"]
    train_log, val_log = [], []
    best = (float("inf"), None, 0)          # early stopping: keep the best validated policy
    for ep in range(1, episodes + 1):
        start_hour = int(rng.integers(sim["day_start_hour"], sim["day_end_hour"]))
        ctrl = TrafficAgent(cfg, "ql", agent, train=True)
        m = ctrl.run(Junction(cfg, rng), SensorArray(cfg, rng), start_hour * 3600, start_hour * 3600 + ep_len)
        train_log.append({"episode": ep, "start_hour": start_hour, "epsilon": round(agent.epsilon, 4),
                          "avg_wait_s": round(m["avg_wait_s"], 2), "avg_queue_veh": round(m["avg_queue_veh"], 2),
                          "mean_abs_td_error": round(float(np.mean(ctrl.td_errors)), 4)})
        agent.decay()
        if ep % VALIDATE_EVERY == 0 or ep == 1:
            v = validate(cfg, agent)
            val_log.append({"episode": ep, "val_avg_wait_s": round(v, 2)})
            if v < best[0]:
                best = (v, (agent.Q.copy(), agent.N.copy()), ep)
            if verbose:
                print("ep %4d | eps %.3f | train wait %6.2fs | |TD| %.3f | VALIDATION wait %6.2fs"
                      % (ep, agent.epsilon, m["avg_wait_s"], train_log[-1]["mean_abs_td_error"], v))
    agent.best_episode, agent.best_val = best[2], round(best[0], 2)
    agent.final_Q, agent.final_N = agent.Q.copy(), agent.N.copy()
    agent.Q, agent.N = best[1]
    return agent, train_log, val_log


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_learning_curve(train_log, val_log, path):
    ep = [r["episode"] for r in train_log]
    td = [r["mean_abs_td_error"] for r in train_log]
    td_smooth = np.convolve(td, np.ones(20) / 20, mode="valid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["episode"] for r in val_log], [r["val_avg_wait_s"] for r in val_log],
                 color="black", marker="o", markersize=3, linewidth=1.5)
    axes[0].set_title("Validation: average wait per vehicle (greedy policy)")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Average wait (s)")
    axes[1].plot(ep[19:], td_smooth, color="black", linewidth=1.5)
    axes[1].set_title("Mean |TD error| per episode (20-ep moving avg)")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("|TD error|")
    for ax in axes:
        ax.grid(True, color="0.85", linewidth=0.6)
        for s in ax.spines.values():
            s.set_color("black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sweep(cfg):
    sw = cfg["hyperparameter_sweep"]
    rows = []
    for a in sw["alphas"]:
        t0 = time.time()
        _, _, val = train(cfg, alpha=a, episodes=sw["episodes"], verbose=False)
        rows.append({"parameter": "alpha", "value": a, "gamma": cfg["q_learning"]["discount_gamma"],
                     "final_val_wait_s": val[-1]["val_avg_wait_s"],
                     "best_val_wait_s": min(v["val_avg_wait_s"] for v in val),
                     "curve": [v["val_avg_wait_s"] for v in val], "episodes_axis": [v["episode"] for v in val]})
        print("sweep alpha=%s -> final %.2fs (%.0fs)" % (a, rows[-1]["final_val_wait_s"], time.time() - t0))
    for g in sw["gammas"]:
        _, _, val = train(cfg, gamma=g, episodes=sw["episodes"], verbose=False)
        rows.append({"parameter": "gamma", "value": g, "alpha": cfg["q_learning"]["learning_rate_alpha"],
                     "final_val_wait_s": val[-1]["val_avg_wait_s"],
                     "best_val_wait_s": min(v["val_avg_wait_s"] for v in val),
                     "curve": [v["val_avg_wait_s"] for v in val], "episodes_axis": [v["episode"] for v in val]})
        print("sweep gamma=%s -> final %.2fs" % (g, rows[-1]["final_val_wait_s"]))
    return rows


def plot_sweep(rows, path):
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, param in zip(axes, ["alpha", "gamma"]):
        for st, r in zip(styles, [r for r in rows if r["parameter"] == param]):
            ax.plot(r["episodes_axis"], r["curve"], color="black", linestyle=st, linewidth=1.5,
                    label="%s = %s" % ("α" if param == "alpha" else "γ", r["value"]))
        ax.set_title("Learning-rate (α) comparison" if param == "alpha" else "Discount-factor (γ) comparison")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Validation average wait (s)")
        ax.legend(frameon=False)
        ax.grid(True, color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_cfg()
    t0 = time.time()
    print("=== Training main policy ===")
    agent, train_log, val_log = train(cfg)
    agent.save("outputs/q_table.npz")
    print("Best validated policy: episode %d (val wait %.2fs) -> saved as outputs/q_table.npz"
          % (agent.best_episode, agent.best_val))
    write_csv("outputs/training_log.csv", train_log)
    write_csv("outputs/validation_log.csv", val_log)
    plot_learning_curve(train_log, val_log, "figures/learning_curve.png")
    print("Main training done in %.0fs" % (time.time() - t0))

    print("=== Hyperparameter sweep ===")
    rows = sweep(cfg)
    write_csv("outputs/hyperparameter_sweep.csv",
              [{k: v for k, v in r.items() if k not in ("curve", "episodes_axis")} for r in rows])
    plot_sweep(rows, "figures/hyperparameter_sweep.png")
    print("All done in %.0fs" % (time.time() - t0))
