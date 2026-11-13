"""
knowledge_base.py - LONG-TERM MEMORY of the agent.

Stores three things and updates them in real time from environment feedback:
  1. policy        : the learned Q-table (owned by QAgent, referenced here)
  2. demand_profile: running average of observed vehicles per hour, per axis
  3. event_log     : sensor faults, fallbacks, safety overrides (audit trail
                     that human operators review)
"""
import json
from collections import defaultdict


class KnowledgeBase:
    def __init__(self):
        self.demand_sum = defaultdict(lambda: [0.0, 0.0])   # hour -> [NS, EW]
        self.demand_n = defaultdict(int)
        self.events = []

    def update_demand(self, hour, ns_queue, ew_queue):
        self.demand_sum[hour][0] += ns_queue
        self.demand_sum[hour][1] += ew_queue
        self.demand_n[hour] += 1

    def expected_demand(self, hour):
        n = self.demand_n.get(hour, 0)
        if n == 0:
            return None
        s = self.demand_sum[hour]
        return round(s[0] / n, 2), round(s[1] / n, 2)

    def log_event(self, t, kind, detail):
        self.events.append({"time": clock(t), "type": kind, "detail": detail})

    def save(self, path):
        out = {
            "demand_profile_avg_queue": {h: self.expected_demand(h) for h in sorted(self.demand_n)},
            "event_counts": {k: sum(1 for e in self.events if e["type"] == k)
                             for k in sorted({e["type"] for e in self.events})},
            "events": self.events[:200],
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


def clock(t):
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)
