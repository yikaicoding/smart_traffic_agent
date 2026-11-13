"""
env.py - Simulated 4-way signalised junction (the agent's ENVIRONMENT).

Traffic is SYNTHETIC: vehicles arrive per approach (N, S, E, W) using the
hourly rates in config.json. Phase 0 = North-South green, Phase 1 = East-West green.
Every switch costs `lost_time_per_switch_s` seconds (amber + all-red) where no car moves.
"""
from collections import deque

APPROACHES = ["N", "S", "E", "W"]
PHASE_APPROACHES = {0: ("N", "S"), 1: ("E", "W")}


class Junction:
    def __init__(self, cfg, rng, surge=None):
        sim = cfg["simulation"]
        self.cfg = cfg
        self.rng = rng
        self.sat_flow = sim["saturation_flow_veh_per_s"]
        self.lost_time = sim["lost_time_per_switch_s"]
        self.profile = {int(h): v for h, v in cfg["arrival_profile_veh_per_s_per_approach"].items()
                        if not h.startswith("_")}
        # Optional unseen-demand surge: (hour, multiplier for E-W) used only in testing
        self.surge = surge

    # ------------------------------------------------------------------
    def reset(self, start_s):
        self.t = start_s                                   # seconds since midnight
        self.queues = {a: deque() for a in APPROACHES}     # arrival timestamps (FIFO)
        self.credit = {a: 0.0 for a in APPROACHES}         # fractional discharge capacity
        self.phase = 0
        self.green_elapsed = 0                             # seconds current phase has been green
        self.lost_remaining = 0
        self.waits = []                                    # wait time of every departed vehicle
        self.departed = 0
        self.queue_samples = []                            # total queue each second
        self.wasted_green_s = 0                            # green on an empty approach while others wait

    def hour(self):
        return int(self.t // 3600)

    def arrival_rates(self):
        ns, ew = self.profile.get(self.hour(), [0.03, 0.03])
        if self.surge and self.hour() == self.surge[0]:
            ew *= self.surge[1]
        return {"N": ns, "S": ns, "E": ew, "W": ew}

    def queue_lengths(self):
        return {a: len(q) for a, q in self.queues.items()}

    def switch(self):
        self.phase = 1 - self.phase
        self.green_elapsed = 0
        self.lost_remaining = self.lost_time

    # ------------------------------------------------------------------
    def advance(self, seconds):
        """Run the physical world forward `seconds` (1 s resolution)."""
        for _ in range(seconds):
            rates = self.arrival_rates()
            for a in APPROACHES:                                    # 1. arrivals
                if self.rng.random() < rates[a]:
                    self.queues[a].append(self.t)

            if self.lost_remaining > 0:                             # 2. amber/all-red
                self.lost_remaining -= 1
            else:                                                   # 3. discharge on green
                green = PHASE_APPROACHES[self.phase]
                red = PHASE_APPROACHES[1 - self.phase]
                if sum(len(self.queues[a]) for a in green) == 0 and \
                   sum(len(self.queues[a]) for a in red) > 0:
                    self.wasted_green_s += 1
                for a in green:
                    self.credit[a] = min(self.credit[a] + self.sat_flow, 1.0)
                    if self.queues[a] and self.credit[a] >= 1.0:
                        arrived = self.queues[a].popleft()
                        self.waits.append(self.t - arrived)
                        self.departed += 1
                        self.credit[a] -= 1.0
                self.green_elapsed += 1

            self.queue_samples.append(sum(len(q) for q in self.queues.values()))
            self.t += 1
