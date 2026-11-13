"""
fuzzy_safety.py - Fuzzy-logic SAFETY SUPERVISOR (zero-order Sugeno inference).

Purpose: stop the red side from being starved if the learned policy keeps
holding green. It does NOT replace the ML policy; it can only force a SWITCH.

Inputs
  red_queue  : estimated vehicles waiting on the red approaches (veh)
  red_wait   : how long the red approaches have been red (s)
Output
  urgency    : 0..1 (how strongly the red side needs green)

Rule base (AND = min, aggregation = weighted average of singletons)
               red_wait SHORT   MEDIUM   LONG
  red_queue LOW         LOW     LOW      MEDIUM
  red_queue MEDIUM      LOW     MEDIUM   HIGH
  red_queue HIGH        MEDIUM  HIGH     HIGH
Output singletons: LOW = 0.2, MEDIUM = 0.5, HIGH = 0.9
"""


def _tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    return (x - a) / (b - a) if x <= b else (c - x) / (c - b)


def _left(x, a, b):          # 1 below a, falls to 0 at b
    return 1.0 if x <= a else 0.0 if x >= b else (b - x) / (b - a)


def _right(x, a, b):         # 0 below a, rises to 1 at b
    return 0.0 if x <= a else 1.0 if x >= b else (x - a) / (b - a)


def queue_sets(q):
    return {"LOW": _left(q, 0, 6), "MEDIUM": _tri(q, 3, 8, 13), "HIGH": _right(q, 10, 16)}


def wait_sets(w):
    return {"SHORT": _left(w, 0, 30), "MEDIUM": _tri(w, 15, 40, 65), "LONG": _right(w, 50, 80)}


RULES = {
    ("LOW", "SHORT"): "LOW",     ("LOW", "MEDIUM"): "LOW",     ("LOW", "LONG"): "MEDIUM",
    ("MEDIUM", "SHORT"): "LOW",  ("MEDIUM", "MEDIUM"): "MEDIUM", ("MEDIUM", "LONG"): "HIGH",
    ("HIGH", "SHORT"): "MEDIUM", ("HIGH", "MEDIUM"): "HIGH",   ("HIGH", "LONG"): "HIGH",
}
SINGLETON = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.9}


def urgency(red_queue, red_wait, explain=False):
    qs, ws = queue_sets(red_queue), wait_sets(red_wait)
    num = den = 0.0
    fired = []
    for (ql, wl), out in RULES.items():
        strength = min(qs[ql], ws[wl])
        if strength > 0:
            fired.append((ql, wl, out, round(strength, 3)))
            num += strength * SINGLETON[out]
            den += strength
    u = num / den if den else 0.0
    return (u, fired, qs, ws) if explain else u


if __name__ == "__main__":
    # Worked example used in the report (Task 1, Section 1.2)
    u, fired, qs, ws = urgency(12, 55, explain=True)
    print("red_queue=12 veh, red_wait=55 s")
    print("queue memberships:", {k: round(v, 3) for k, v in qs.items()})
    print("wait memberships :", {k: round(v, 3) for k, v in ws.items()})
    for f in fired:
        print("  IF queue is %s AND wait is %s THEN urgency is %s  (strength %.3f)" % f)
    print("urgency = %.3f" % u)
