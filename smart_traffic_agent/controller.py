"""
controller.py - The AGENT LOOP: perceive -> remember -> reason -> act -> learn.

Modes
  "fixed" : fixed-time baseline (30 s / 30 s), no intelligence
  "ql"    : Q-learning policy only (+ hard min/max green limits)
  "agent" : full agent = Q-learning + fuzzy safety supervisor + sensor-fault fallback
Training uses mode "ql" with exploration and Q-updates switched on.
"""
import numpy as np

from env import PHASE_APPROACHES
from fuzzy_safety import urgency
from knowledge_base import KnowledgeBase, clock
from perception import ShortTermMemory
from q_agent import ACTIONS

KEEP, SWITCH = 0, 1


class TrafficAgent:
    def __init__(self, cfg, mode, qagent=None, train=False, log=False, kb=None):
        self.cfg, self.mode, self.q, self.train, self.log_on = cfg, mode, qagent, train, log
        self.dt = cfg["simulation"]["decision_interval_s"]
        self.min_g = cfg["safety"]["min_green_s"]
        self.max_g = cfg["safety"]["max_green_s"]
        self.thr = cfg["safety"]["fuzzy_override_threshold"]
        self.fixed = [cfg["fixed_time_baseline"]["green_ns_s"], cfg["fixed_time_baseline"]["green_ew_s"]]
        self.penalty = cfg["q_learning"]["switch_penalty"]
        self.stm = ShortTermMemory(cfg)
        self.kb = kb or KnowledgeBase()
        self.records = []
        self.counts = {"ML": 0, "SAFETY_OVERRIDE": 0, "FALLBACK": 0, "LOW_CONFIDENCE": 0,
                       "MIN_GREEN_BLOCK": 0, "MAX_GREEN_RULE": 0, "FIXED": 0}
        self.td_errors = []
        self._in_fault = False

    # ------------------------------------------------------------------
    def run(self, env, sensors, start_s, end_s):
        env.reset(start_s)
        pending = None                                   # (s, a, r) waiting for s'
        switches = 0
        while env.t < end_s:
            # 1. PERCEIVE (noisy, possibly missing readings)
            readings = sensors.read(env.queue_lengths(), env.t)
            # 2. SHORT-TERM MEMORY -> robust estimate
            self.stm.update(readings)
            est = {a: self.stm.estimate(a) for a in readings}
            g_ap, r_ap = PHASE_APPROACHES[env.phase], PHASE_APPROACHES[1 - env.phase]
            gq, rq = sum(est[a] for a in g_ap), sum(est[a] for a in r_ap)
            elapsed = env.green_elapsed
            faulty = self.stm.faulty_approaches()
            # 3. LONG-TERM MEMORY update from feedback
            self.kb.update_demand(env.hour(), est["N"] + est["S"], est["E"] + est["W"])

            s = self.q.state(gq, rq, elapsed) if self.q is not None else None
            if pending is not None and self.train:        # learn from previous step
                ps, pa, pr = pending
                self.td_errors.append(abs(self.q.update(ps, pa, pr, s)))

            # 4. REASON + DECIDE
            info, u, note = None, None, ""
            if self.mode == "fixed":
                action, source = (SWITCH if elapsed >= self.fixed[env.phase] else KEEP), "FIXED"
            elif self.mode == "agent" and faulty:
                action = SWITCH if elapsed >= self.fixed[env.phase] else KEEP
                source = "FALLBACK"
                if not self._in_fault:
                    self.kb.log_event(env.t, "SENSOR_FAULT", "No data from %s; fixed-time fallback, operator alerted" % ",".join(faulty))
                    self._in_fault = True
            else:
                if self._in_fault:
                    self.kb.log_event(env.t, "SENSOR_RECOVERED", "Detectors back online; ML control resumed")
                    self._in_fault = False
                action = self.q.act(s, explore=self.train)
                source = "ML"
                info = self.q.explain(s)
                if self.mode == "agent":
                    u = urgency(rq, elapsed)
                    if not info["confident"]:
                        action, source = (SWITCH if u >= 0.5 else KEEP), "LOW_CONFIDENCE"
                    elif action == KEEP and u >= self.thr and elapsed >= self.min_g:
                        action, source = SWITCH, "SAFETY_OVERRIDE"
                        self.kb.log_event(env.t, "SAFETY_OVERRIDE",
                                          "ML wanted KEEP; fuzzy urgency %.2f (red queue %.0f, red wait %ds)" % (u, rq, elapsed))

            # 5. HARD CONSTRAINTS (always win)
            if action == SWITCH and elapsed < self.min_g:
                action, note = KEEP, "blocked by min-green"
                self.counts["MIN_GREEN_BLOCK"] += 1
            elif action == KEEP and elapsed >= self.max_g:
                action, source = SWITCH, "MAX_GREEN_RULE"
            self.counts[source] = self.counts.get(source, 0) + 1

            if self.log_on:
                self.records.append({
                    "time": clock(env.t), "green": "NS" if env.phase == 0 else "EW", "elapsed_s": elapsed,
                    "true_NS": len(env.queues["N"]) + len(env.queues["S"]),
                    "true_EW": len(env.queues["E"]) + len(env.queues["W"]),
                    "read": "/".join("-" if readings[a] is None else str(readings[a]) for a in "NSEW"),
                    "est_green_q": gq, "est_red_q": rq, "state": s,
                    "Q_KEEP": info["Q_KEEP"] if info else "", "Q_SWITCH": info["Q_SWITCH"] if info else "",
                    "visits": info["visits"] if info else "", "urgency": round(u, 2) if u is not None else "",
                    "action": ACTIONS[action], "decided_by": source, "note": note})

            # 6. ACT on the environment
            if action == SWITCH:
                env.switch()
                switches += 1
            env.advance(self.dt)

            # 7. REWARD (feedback from the environment)
            if self.q is not None:
                reward = -sum(env.queue_lengths().values()) - (self.penalty if action == SWITCH else 0)
                pending = (s, action, reward)

        waits = env.waits or [0]
        return {
            "avg_wait_s": float(np.mean(waits)),
            "p95_wait_s": float(np.percentile(waits, 95)),
            "avg_queue_veh": float(np.mean(env.queue_samples)),
            "max_queue_veh": int(np.max(env.queue_samples)),
            "throughput_veh": env.departed,
            "wasted_green_s": env.wasted_green_s,
            "switches": switches,
        }
