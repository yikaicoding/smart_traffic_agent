"""
demo.py - Live demo + EXECUTION LOG of the full agent (used for the viva).

Runs the full agent on held-out seed 101, stress day, and writes:
  outputs/decision_log.csv     every decision (state, Q-values, urgency, who decided)
  outputs/knowledge_base.json  long-term memory after the day (demand profile + events)
  outputs/execution_log.txt    readable excerpts: normal ML decisions, low-confidence
                               deferral during the unseen surge, sensor-fault fallback
Run:  python demo.py
"""
import csv
import json

import numpy as np

from evaluate import run_day
from q_agent import QAgent

HEADER = "%-8s %-3s %4s | %-11s | %3s %3s | %-9s | %7s %8s %6s | %5s | %-6s | %-16s" % (
    "time", "grn", "el_s", "read N/S/E/W", "G_q", "R_q", "state", "Q_KEEP", "Q_SWITCH", "visits",
    "urg", "ACTION", "decided_by")


def fmt(r):
    return "%-8s %-3s %4s | %-11s | %3s %3s | %-9s | %7s %8s %6s | %5s | %-6s | %-16s %s" % (
        r["time"], r["green"], r["elapsed_s"], r["read"], r["est_green_q"], r["est_red_q"],
        str(r["state"]).replace(" ", ""), r["Q_KEEP"], r["Q_SWITCH"], r["visits"], r["urgency"],
        r["action"], r["decided_by"], r["note"])


def window(records, start, n):
    idx = next(i for i, r in enumerate(records) if r["time"] >= start)
    return records[idx:idx + n]


def main():
    cfg = json.load(open("config.json"))
    agent = QAgent(cfg, np.random.default_rng(0))
    agent.load("outputs/q_table.npz")
    m, ctrl, _ = run_day(cfg, "agent", agent, 101, "stress", log=True)
    recs = ctrl.records

    with open("outputs/decision_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    ctrl.kb.save("outputs/knowledge_base.json")

    lines = ["SMART TRAFFIC AGENT - EXECUTION LOG (seed 101, stress day, full agent)",
             "Q-table: outputs/q_table.npz | config: config.json",
             "Columns: G_q/R_q = estimated queue on green/red side (median of last 6 readings),",
             "state = (green_q_bin, red_q_bin, elapsed_bin), urg = fuzzy urgency, '-' = missing reading", ""]
    sections = [("A. Morning peak - normal ML decisions", "08:00:00", 14)]
    lc = [r for r in recs if r["decided_by"] == "LOW_CONFIDENCE"]
    if lc:
        sections.append(("B. Unseen surge - ML has low confidence, fuzzy supervisor decides", lc[0]["time"], 10))
    h = cfg["evaluation"]["sensor_fault_window_hour"]
    sections.append(("C. East detector outage - fallback to fixed timing", "%02d:00:00" % h, 8))
    sections.append(("D. Detector recovery - ML control resumes", "%02d:10:00" % h, 6))
    for title, start, n in sections:
        lines += [title, HEADER, "-" * len(HEADER)] + [fmt(r) for r in window(recs, start, n)] + [""]

    lines.append("KNOWLEDGE-BASE EVENTS (first 6):")
    for e in ctrl.kb.events[:6]:
        lines.append("  %s  %-16s %s" % (e["time"], e["type"], e["detail"]))
    lines += ["", "DAY SUMMARY: " + ", ".join("%s=%s" % (k, round(v, 2)) for k, v in m.items()),
              "DECISION SOURCES: " + ", ".join("%s=%d" % kv for kv in ctrl.counts.items())]
    text = "\n".join(lines)
    open("outputs/execution_log.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
