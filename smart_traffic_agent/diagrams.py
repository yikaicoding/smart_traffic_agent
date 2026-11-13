"""
diagrams.py - Draws the report diagrams in black only, on an exact grid so every
box is aligned and evenly spaced.
  figures/architecture.png     (Task 1.2 - agent architecture)
  figures/sequence.png         (Task 1.3 - UML sequence diagram, KB update loop)
  figures/decision_flow.png    (Task 2.1 - how a decision is picked)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

LW = 1.2
FS = 8.5


def box(ax, cx, cy, w, h, text, bold=False, fs=FS, rounded=False):
    style = "round,pad=0,rounding_size=0.12" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style,
                                fc="white", ec="black", lw=LW, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def diamond(ax, cx, cy, w, h, text, fs=FS):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc="white", ec="black", lw=LW, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, dashed=False, text=None, tx=None, ty=None, ha="left", fs=FS - 0.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW, mutation_scale=11,
                                linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))
    if text:
        ax.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2,
                text, fontsize=fs, ha=ha, va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.0))


def line(ax, xs, ys, dashed=False):
    ax.plot(xs, ys, color="black", lw=LW, zorder=2, linestyle=(0, (4, 3)) if dashed else "-")


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


# ======================================================================
# 1. ARCHITECTURE
# ======================================================================
def architecture():
    fig, ax = canvas(9, 9.6, 12, 13.4)
    ax.set_ylim(1.2, 13.4)
    layers = [  # (title, y_centre, boxes)
        ("1  ENVIRONMENT", 12.2, ["Signalised 4-way junction\n(vehicles on N, S, E, W)",
                                  "Disturbances\n(surges, incidents, overrides)"]),
        ("2  PERCEPTION", 9.9, ["Vehicle detectors\n(noisy, may drop out)",
                                 "Sensor gateway\n(MQTT, every 5 s)"]),
        ("3  MEMORY", 7.6, ["Short-term memory\nlast 6 readings per approach\n(median estimate)",
                            "Long-term knowledge base\nQ-table + visit counts,\ndemand profile, event log"]),
        ("4  REASONING & DECISION", 5.3, ["Fault check\n(3 missing\nreadings)", "Q-learning\npolicy\n(ML core)",
                                           "Fuzzy safety\nsupervisor", "Hard limits\nmin 10 s\nmax 90 s"]),
        ("5  ACTION & HUMAN OVERSIGHT", 3.0, ["Signal controller\n(KEEP / SWITCH phase)",
                                              "Operator dashboard\n(explanations, alerts,\nmanual override)"]),
    ]
    band_l, band_r, band_h = 1.1, 10.9, 1.85
    for title, yc, items in layers:
        ax.add_patch(Rectangle((band_l, yc - band_h / 2), band_r - band_l, band_h, fc="white",
                               ec="black", lw=0.8, linestyle=(0, (2, 2)), zorder=1))
        ax.text(band_l + 0.12, yc + band_h / 2 - 0.2, title, fontsize=FS, fontweight="bold", va="center")
        n = len(items)
        inner_l, inner_r = band_l + 0.35, band_r - 0.35
        gap = 0.35
        w = (inner_r - inner_l - gap * (n - 1)) / n
        for i, t in enumerate(items):
            cx = inner_l + w / 2 + i * (w + gap)
            box(ax, cx, yc - 0.12, w, 1.2, t)
            if n == 4 and i < 3:                     # arrows inside reasoning layer
                arrow(ax, cx + w / 2, yc - 0.12, cx + w / 2 + gap, yc - 0.12)
    # downward flow between layers (centre)
    ycs = [12.2, 9.9, 7.6, 5.3, 3.0]
    for y1, y2 in zip(ycs, ycs[1:]):
        arrow(ax, 6, y1 - band_h / 2, 6, y2 + band_h / 2)
    # action -> environment (left loop)
    line(ax, [band_l, 0.55, 0.55], [2.88, 2.88, 12.08])
    arrow(ax, 0.55, 12.08, band_l, 12.08)
    ax.text(0.38, 7.4, "actuate signal phase", rotation=90, fontsize=FS - 0.5, ha="center", va="center")
    # environment -> KB (right loop: reward feedback)
    line(ax, [band_r, 11.45, 11.45], [12.08, 12.08, 7.48])
    arrow(ax, 11.45, 7.48, band_r, 7.48)
    ax.text(11.62, 9.8, "reward feedback (queue change)", rotation=270, fontsize=FS - 0.5,
            ha="center", va="center")
    # fallback path text
    ax.text(6, 1.55, "If the fault check fires, the policy is bypassed and a fixed-time plan runs (fallback); "
            "the operator is alerted.", ha="center", fontsize=FS - 0.5, style="italic")
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/architecture.png", dpi=220)
    plt.close(fig)


# ======================================================================
# 2. UML SEQUENCE DIAGRAM
# ======================================================================
def sequence():
    names = ["Junction\n(environment)", "Sensors", "Short-term\nmemory", "Agent\ncontroller",
             "Q-learning\npolicy", "Fuzzy\nsupervisor", "Knowledge\nbase", "Signal\ncontroller", "Operator"]
    n = len(names)
    xmax, ymax = 18.0, 15.0
    fig, ax = canvas(10.5, 9.9, xmax, ymax)
    ax.set_ylim(-0.9, ymax)
    xs = [1.0 + i * (xmax - 2.0) / (n - 1) for i in range(n)]
    top, bottom = 14.2, -0.75
    for x, nm in zip(xs, names):
        box(ax, x, top, 1.75, 0.95, nm, fs=8)
        line(ax, [x, x], [top - 0.475, bottom], dashed=True)
    X = dict(zip(["env", "sen", "stm", "agt", "pol", "fuz", "kb", "sig", "op"], xs))

    def msg(a, b, y, text, ret=False):
        arrow(ax, X[a], y, X[b], y, dashed=ret)
        ax.text((X[a] + X[b]) / 2, y + 0.12, text, fontsize=7.5, ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)

    y = 13.0
    step = 0.78
    msg("env", "sen", y, "1: traffic state (every 5 s)"); y -= step
    msg("sen", "stm", y, "2: readings (noisy / missing)"); y -= step
    msg("stm", "agt", y, "3: median queue estimates + fault flags", ret=False); y -= step
    msg("agt", "kb", y, "4: update demand profile"); y -= step + 0.25

    # alt frame
    f_top, f_l, f_r = y + 0.45, X["agt"] - 0.95, X["op"] + 0.75
    y_split = y - 0.75 - 2 * step - 0.35
    f_bot = y_split - 0.75 - 3 * step - 0.4
    ax.add_patch(Rectangle((f_l, f_bot), f_r - f_l, f_top - f_bot, fc="none", ec="black", lw=LW, zorder=1))
    ax.add_patch(Polygon([(f_l, f_top), (f_l + 0.9, f_top), (f_l + 0.9, f_top - 0.3), (f_l + 0.75, f_top - 0.45),
                          (f_l, f_top - 0.45)], closed=True, fc="white", ec="black", lw=LW, zorder=2))
    ax.text(f_l + 0.42, f_top - 0.22, "alt", fontsize=8, fontweight="bold", ha="center", va="center", zorder=3)
    ax.text(f_l + 1.05, f_top - 0.22, "[sensor fault: 3 missing readings]", fontsize=7.5, va="center")
    y -= 0.75
    msg("agt", "sig", y, "5a: fixed-time plan (fallback)"); y -= step
    msg("agt", "kb", y, "6a: log SENSOR_FAULT event"); y -= step
    msg("kb", "op", y, "7a: alert operator"); y -= step
    line(ax, [f_l, f_r], [y_split, y_split], dashed=True)
    ax.text(f_l + 1.05, y_split - 0.22, "[sensors healthy]", fontsize=7.5, va="center")
    y = y_split - 0.85
    msg("agt", "pol", y, "5b: state s = (green q, red q, elapsed)"); y -= step
    msg("pol", "agt", y, "6b: Q(s,KEEP), Q(s,SWITCH), visits", ret=True); y -= step
    msg("agt", "fuz", y, "7b: red queue, red wait"); y -= step
    msg("fuz", "agt", y, "8b: urgency (0-1)", ret=True); y -= step
    y = f_bot - 0.6
    msg("agt", "sig", y, "9: KEEP / SWITCH (after min / max green check)"); y -= step
    msg("sig", "env", y, "10: set signal phase"); y -= step
    msg("env", "agt", y, "11: new queues -> reward r", ret=True); y -= step
    msg("agt", "kb", y, "12: Q-update + visit count, append decision log"); y -= step
    msg("kb", "op", y, "13: explanation on dashboard", ret=True)
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/sequence.png", dpi=220)
    plt.close(fig)


# ======================================================================
# 3. DECISION FLOWCHART (how the ML picks an action)
# ======================================================================
def decision_flow():
    fig, ax = canvas(8.4, 10.4, 12, 15.3)
    ax.set_ylim(2.3, 15.3)
    cx, rx = 4.1, 8.6            # main spine and right-hand branch boxes
    w, h = 4.2, 0.78
    bw = 3.2
    lane_lc, lane_fb = 10.55, 11.25   # separate return lanes: low-confidence, fallback
    ys = {"start": 14.9, "read": 13.95, "stm": 13.0, "fault": 11.85, "state": 10.45, "conf": 9.25,
          "argmax": 7.85, "over": 6.6, "hard": 5.15, "act": 4.1, "learn": 3.05}
    box(ax, cx, ys["start"], 2.6, 0.6, "Every 5 s", rounded=True, bold=True)
    box(ax, cx, ys["read"], w, h, "Read 4 detectors (noisy, may be missing)")
    box(ax, cx, ys["stm"], w, h, "Short-term memory: median of last 6 readings")
    diamond(ax, cx, ys["fault"], 3.8, 1.15, "3 missing readings\nin a row?")
    box(ax, cx, ys["state"], w, h, "Build state s = (green q bin, red q bin,\nelapsed bin)")
    diamond(ax, cx, ys["conf"], 3.8, 1.15, "State visited\n>= 20 times?")
    box(ax, cx, ys["argmax"], w, h, "ML picks a* = argmax Q(s, a)\n(KEEP or SWITCH)")
    diamond(ax, cx, ys["over"], 3.8, 1.25, "a* = KEEP and fuzzy\nurgency >= 0.75?")
    box(ax, cx, ys["hard"], w, h, "Hard limits: no SWITCH before 10 s,\nforce SWITCH at 90 s")
    box(ax, cx, ys["act"], w, h, "Execute action + write decision log")
    box(ax, cx, ys["learn"], w, h, "Reward r = -(total queue) - switch cost\n(training only: Q-update)")
    box(ax, rx, ys["fault"], bw, 0.95, "FALLBACK: fixed-time\n30 s plan + alert")
    box(ax, rx, ys["conf"], bw, 0.95, "LOW CONFIDENCE:\nfuzzy decides (>= 0.5)")
    box(ax, rx, ys["over"], bw, 0.95, "SAFETY OVERRIDE:\nSWITCH + log event")
    halves = {"start": 0.3, "read": h / 2, "stm": h / 2, "fault": 0.575, "state": h / 2, "conf": 0.575,
              "argmax": h / 2, "over": 0.625, "hard": h / 2, "act": h / 2, "learn": h / 2}
    order = list(ys)
    for a, b in zip(order, order[1:]):
        arrow(ax, cx, ys[a] - halves[a], cx, ys[b] + halves[b])
    labels = {"fault": ("Yes", "No"), "conf": ("No", "Yes"), "over": ("Yes", "No")}
    for k, (right, down) in labels.items():
        arrow(ax, cx + 1.9, ys[k], rx - bw / 2, ys[k])
        ax.text(cx + 2.0, ys[k] + 0.18, right, fontsize=FS - 0.5)
        ax.text(cx + 0.14, ys[k] - halves[k] - 0.2, down, fontsize=FS - 0.5, va="center")
    # override -> hard limits (straight down then left)
    line(ax, [rx, rx], [ys["over"] - 0.475, ys["hard"]])
    arrow(ax, rx, ys["hard"], cx + w / 2, ys["hard"])
    # low confidence -> own lane -> joins the same line into hard limits
    line(ax, [rx + bw / 2, lane_lc, lane_lc, rx], [ys["conf"], ys["conf"], ys["hard"], ys["hard"]])
    # fallback -> own lane -> execute (skips ML and hard-limit step)
    line(ax, [rx + bw / 2, lane_fb, lane_fb], [ys["fault"], ys["fault"], ys["act"]])
    arrow(ax, lane_fb, ys["act"], cx + w / 2, ys["act"])
    # loop back
    line(ax, [cx - w / 2, 0.75, 0.75], [ys["learn"], ys["learn"], ys["read"]])
    arrow(ax, 0.75, ys["read"], cx - w / 2, ys["read"])
    ax.text(0.53, (ys["learn"] + ys["read"]) / 2, "next decision", rotation=90, fontsize=FS - 0.5,
            ha="center", va="center")
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/decision_flow.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    sequence()
    decision_flow()
    print("diagrams saved")
