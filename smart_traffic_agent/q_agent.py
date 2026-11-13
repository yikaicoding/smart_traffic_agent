"""
q_agent.py - The core ML component: a TABULAR Q-LEARNING policy.

State  (from the point of view of the CURRENT green phase, so one table
        serves both N-S and E-W):
    (green_queue_bin, red_queue_bin, elapsed_bin)  -> 4 x 4 x 4 = 64 states
Actions: 0 = KEEP current green, 1 = SWITCH phase
Update  (Watkins Q-learning, one transition at a time, i.e. batch size 1):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""
import numpy as np

ACTIONS = ["KEEP", "SWITCH"]


def to_bin(value, edges):
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


class QAgent:
    def __init__(self, cfg, rng):
        q = cfg["q_learning"]
        self.alpha = q["learning_rate_alpha"]
        self.gamma = q["discount_gamma"]
        self.epsilon = q["epsilon_start"]
        self.eps_min = q["epsilon_min"]
        self.eps_decay = q["epsilon_decay_per_episode"]
        self.q_edges = q["queue_bins"]
        self.t_edges = q["elapsed_bins_s"]
        self.low_conf = q["low_confidence_visit_threshold"]
        nq, nt = len(self.q_edges) + 1, len(self.t_edges) + 1
        self.Q = np.zeros((nq, nq, nt, 2))
        self.N = np.zeros((nq, nq, nt, 2), dtype=int)       # visit counts (confidence)
        self.rng = rng

    def state(self, green_q, red_q, elapsed):
        return (to_bin(green_q, self.q_edges), to_bin(red_q, self.q_edges), to_bin(elapsed, self.t_edges))

    @staticmethod
    def valid_actions(s):
        # elapsed bin 0 means green < min_green (10 s): SWITCH is not allowed there.
        # Masking it stops the agent bootstrapping from an action it can never take.
        return [0] if s[2] == 0 else [0, 1]

    def act(self, s, explore):
        valid = self.valid_actions(s)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        if len(valid) == 1:
            return 0
        q = self.Q[s]
        return int(np.argmax(q)) if q[0] != q[1] else 0      # tie -> KEEP

    def update(self, s, a, r, s_next):
        best_next = max(self.Q[s_next][a2] for a2 in self.valid_actions(s_next))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += self.alpha * td_error
        self.N[s][a] += 1
        return td_error

    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def explain(self, s):
        q = self.Q[s]
        visits = int(self.N[s].sum())
        can_switch = len(self.valid_actions(s)) == 2
        return {"Q_KEEP": round(float(q[0]), 2),
                "Q_SWITCH": round(float(q[1]), 2) if can_switch else "n/a",
                "margin": round(abs(float(q[0] - q[1])), 2) if can_switch else "n/a", "visits": visits,
                "confident": visits >= self.low_conf}

    def save(self, path):
        np.savez(path, Q=self.Q, N=self.N)

    def load(self, path):
        d = np.load(path)
        self.Q, self.N = d["Q"], d["N"]
