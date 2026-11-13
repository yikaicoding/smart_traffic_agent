"""plot_fuzzy.py - draws the fuzzy membership functions used in fuzzy_safety.py (black only)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fuzzy_safety import queue_sets, wait_sets

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
styles = ["-", "--", ":"]
xq, xw = np.linspace(0, 20, 400), np.linspace(0, 100, 400)
for (name, st) in zip(["LOW", "MEDIUM", "HIGH"], styles):
    axes[0].plot(xq, [queue_sets(x)[name] for x in xq], color="black", linestyle=st, linewidth=1.6, label=name)
for (name, st) in zip(["SHORT", "MEDIUM", "LONG"], styles):
    axes[1].plot(xw, [wait_sets(x)[name] for x in xw], color="black", linestyle=st, linewidth=1.6, label=name)
axes[0].axvline(12, color="black", linewidth=0.8)
axes[1].axvline(55, color="black", linewidth=0.8)
axes[0].set_title("Input 1: red-side queue (vehicles)")
axes[1].set_title("Input 2: red-side waiting time (s)")
for ax in axes:
    ax.set_ylabel("Membership degree")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="center right")
    ax.grid(True, color="0.85", linewidth=0.6)
fig.tight_layout()
fig.savefig("figures/fuzzy_membership.png", dpi=200)
