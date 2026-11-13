"""
perception.py - Noisy sensors + SHORT-TERM MEMORY (the perception window).

Uncertainty modelled here:
  * noisy readings      -> Gaussian noise on every queue count
  * missing data        -> random dropouts (reading = None)
  * tool/API failure    -> forced outage window on one detector (evaluation only)
The short-term memory keeps the last N readings per approach and estimates the
true queue with the MEDIAN of valid readings, which is robust to single noisy spikes.
"""
from collections import deque
from statistics import median

from env import APPROACHES


class SensorArray:
    def __init__(self, cfg, rng, outage=None):
        s = cfg["sensor"]
        self.noise = s["noise_std_veh"]
        self.p_drop = s["dropout_probability"]
        self.rng = rng
        self.outage = outage            # (approach, start_s, end_s) or None

    def read(self, true_queues, t):
        readings = {}
        for a in APPROACHES:
            if self.outage and a == self.outage[0] and self.outage[1] <= t < self.outage[2]:
                readings[a] = None                                  # detector/API offline
            elif self.rng.random() < self.p_drop:
                readings[a] = None                                  # random packet loss
            else:
                noisy = true_queues[a] + self.rng.normal(0, self.noise)
                readings[a] = max(0, round(noisy))
        return readings


class ShortTermMemory:
    def __init__(self, cfg):
        self.window = cfg["memory"]["short_term_window_readings"]
        self.fault_k = cfg["sensor"]["fault_after_consecutive_dropouts"]
        self.buffer = {a: deque(maxlen=self.window) for a in APPROACHES}

    def update(self, readings):
        for a, r in readings.items():
            self.buffer[a].append(r)

    def estimate(self, a):
        valid = [r for r in self.buffer[a] if r is not None]
        return median(valid) if valid else 0

    def faulty_approaches(self):
        """An approach is flagged FAULTY if its last k readings are all missing."""
        faulty = []
        for a, buf in self.buffer.items():
            recent = list(buf)[-self.fault_k:]
            if len(recent) == self.fault_k and all(r is None for r in recent):
                faulty.append(a)
        return faulty

    def snapshot(self):
        return {a: list(b) for a, b in self.buffer.items()}
