"""
evaluate.py - Compare the trained agent against a fixed-time baseline.

Controllers : fixed (30 s / 30 s), ql (policy only), agent (policy + fuzzy safety + fallback)
Scenarios   : normal day  (06:00-22:00, demand seen in training)
              stress day  (same + UNSEEN E-W surge at 11:00 + E detector outage 14:00-14:10)
Seeds       : 5 held-out seeds (never used in training or validation)
Outputs     : outputs/evaluation_results.csv, outputs/evaluation_summary.csv,
              figures/evaluation_bars.png, figures/hourly_queue.png
Run:  python evaluate.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import TrafficAgent
from env import Junction
from perception import SensorArray
from q_agent import QAgent

CONTROLLERS = [("fixed", "Fixed-time 30/30"), ("ql", "Q-learning only"), ("agent", "Full agent")]
METRICS = ["avg_wait_s", "p95_wait_s", "avg_queue_veh", "max_queue_veh", "throughput_veh",
           "wasted_green_s", "switches"]


def run_day(cfg, mode, agent, seed, scenario, log=False):
    ev, sim = cfg["evaluation"], cfg["simulation"]
    rng = np.random.default_rng(seed)
    surge = outage = None
    if scenario == "stress":
        surge = (ev["unseen_surge_hour"], ev["unseen_surge_multiplier_ew"])
        h = ev["sensor_fault_window_hour"]
        outage = ("E", h * 3600, h * 3600 + 600)
    env = Junction(cfg, rng, surge=surge)
    ctrl = TrafficAgent(cfg, mode, agent, train=False, log=log)
    m = ctrl.run(env, SensorArray(cfg, rng, outage=outage), sim["day_start_hour"] * 3600, sim["day_end_hour"] * 3600)
    hourly = [float(np.mean(env.queue_samples[i * 3600:(i + 1) * 3600]))
              for i in range(sim["day_end_hour"] - sim["day_start_hour"])]
    return m, ctrl, hourly


def main():
    cfg = json.load(open("config.json"))
    agent = QAgent(cfg, np.random.default_rng(0))
    agent.load("outputs/q_table.npz")
    rows, hourly_store = [], {}
    for scenario in ["normal", "stress"]:
        for mode, label in CONTROLLERS:
            for seed in cfg["evaluation"]["test_seeds"]:
                m, ctrl, hourly = run_day(cfg, mode, agent, seed, scenario)
                rows.append({"scenario": scenario, "controller": label, "seed": seed,
                             **{k: round(v, 2) for k, v in m.items()},
                             "safety_overrides": ctrl.counts.get("SAFETY_OVERRIDE", 0),
                             "fallback_decisions": ctrl.counts.get("FALLBACK", 0),
                             "low_confidence_decisions": ctrl.counts.get("LOW_CONFIDENCE", 0)})
                hourly_store.setdefault((scenario, label), []).append(hourly)
            print("done", scenario, label)

    with open("outputs/evaluation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for scenario in ["normal", "stress"]:
        base = None
        for _, label in CONTROLLERS:
            sel = [r for r in rows if r["scenario"] == scenario and r["controller"] == label]
            s = {"scenario": scenario, "controller": label}
            for k in METRICS + ["safety_overrides", "fallback_decisions", "low_confidence_decisions"]:
                vals = [r[k] for r in sel]
                s[k + "_mean"] = round(float(np.mean(vals)), 2)
                s[k + "_std"] = round(float(np.std(vals)), 2)
            if base is None:
                base = s
            s["wait_change_vs_fixed_pct"] = round(100 * (s["avg_wait_s_mean"] - base["avg_wait_s_mean"]) / base["avg_wait_s_mean"], 1)
            s["queue_change_vs_fixed_pct"] = round(100 * (s["avg_queue_veh_mean"] - base["avg_queue_veh_mean"]) / base["avg_queue_veh_mean"], 1)
            summary.append(s)
    with open("outputs/evaluation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n%-8s %-18s %10s %10s %10s %10s %12s %9s" % ("scenario", "controller", "avg wait", "p95 wait",
                                                         "avg queue", "wasted g", "wait vs fix", "override"))
    for s in summary:
        print("%-8s %-18s %9.2fs %9.1fs %10.2f %9.0fs %11.1f%% %9.1f" % (
            s["scenario"], s["controller"], s["avg_wait_s_mean"], s["p95_wait_s_mean"], s["avg_queue_veh_mean"],
            s["wasted_green_s_mean"], s["wait_change_vs_fixed_pct"], s["safety_overrides_mean"]))

    # ---- Figure: bar chart (black/white) ---------------------------------
    hatches = ["", "///", "..."]
    ax_pad = {"avg_wait_s": 1.0, "avg_queue_veh": 0.4}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric, title) in zip(axes, [("avg_wait_s", "Average wait per vehicle (s)"),
                                          ("avg_queue_veh", "Average total queue (vehicles)")]):
        x = np.arange(2)
        for i, (_, label) in enumerate(CONTROLLERS):
            means = [next(s for s in summary if s["scenario"] == sc and s["controller"] == label)[metric + "_mean"]
                     for sc in ["normal", "stress"]]
            stds = [next(s for s in summary if s["scenario"] == sc and s["controller"] == label)[metric + "_std"]
                    for sc in ["normal", "stress"]]
            bars = ax.bar(x + (i - 1) * 0.26, means, 0.24, yerr=stds, capsize=3, color="white",
                          edgecolor="black", hatch=hatches[i], label=label, error_kw={"ecolor": "black"})
            for b, v, sd in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, v + sd + ax_pad[metric], "%.1f" % v, ha="center",
                        va="bottom", fontsize=8)
        top = max(s[metric + "_mean"] + s[metric + "_std"] for s in summary)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels(["Normal day", "Stress day (unseen)"])
        ax.set_title(title)
        ax.grid(True, axis="y", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/evaluation_bars.png", dpi=200)
    plt.close(fig)

    # ---- Figure: hourly queue, normal day ---------------------------------
    hours = list(range(cfg["simulation"]["day_start_hour"], cfg["simulation"]["day_end_hour"]))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # Q-learning only and Full agent are identical on the normal day (no fault, no surge),
    # so the full agent is plotted once and labelled accordingly.
    for (label, name), st, mk in zip([("Fixed-time 30/30", "Fixed-time 30/30"),
                                      ("Full agent", "Full agent (= Q-learning only on normal day)")],
                                     ["-", "--"], ["o", "s"]):
        h = np.mean(hourly_store[("normal", label)], axis=0)
        ax.plot(hours, h, color="black", linestyle=st, linewidth=1.6, marker=mk, markersize=4,
                markerfacecolor="white" if mk == "s" else "black", label=name)
    ax.set_xticks(hours)
    ax.set_xticklabels(["%02d:00" % h for h in hours], rotation=45)
    ax.set_ylabel("Average total queue (veh)")
    ax.set_title("Average queue by hour - normal day (mean of 5 test seeds)")
    ax.legend(frameon=False)
    ax.grid(True, color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig("figures/hourly_queue.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
