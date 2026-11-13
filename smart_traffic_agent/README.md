# Smart Traffic Agent – 5023CMD CW2

Adaptive traffic-signal agent for one 4-way junction. The core ML component is a **tabular Q-learning policy**. A **fuzzy safety supervisor**, a **sensor-fault fallback** and **hard min/max green limits** wrap around it.

> **Data:** the traffic is **simulated** (synthetic hourly arrival rates in `config.json`). No real field data is used.

## Files
| File | Role |
|---|---|
| `config.json` | All hyperparameters, safety limits, demand profile, test settings |
| `env.py` | Junction simulator (arrivals, discharge, lost time on switch) |
| `perception.py` | Noisy and missing sensors + short-term memory (median of last 6 readings) |
| `q_agent.py` | Q-learning policy (64 states × 2 actions), visit counts for confidence |
| `fuzzy_safety.py` | Fuzzy rule base (9 rules), urgency 0–1. Run it directly for the worked example |
| `knowledge_base.py` | Long-term memory: demand profile + event log |
| `controller.py` | Agent loop: perceive → remember → reason → act → learn |
| `train.py` | Training, validation curve, α/γ sweep |
| `evaluate.py` | Fixed-time vs Q-learning vs full agent, 5 held-out seeds, normal + stress day |
| `demo.py` | Execution log (decision trace) for the viva |
| `diagrams.py`, `plot_fuzzy.py` | Report figures |

## Run (Python 3.10+)
```
pip install -r requirements.txt
python train.py        # ~2 min: outputs/q_table.npz, learning curves, sweep
python evaluate.py     # ~15 s: comparison table + charts
python demo.py         # decision trace -> outputs/execution_log.txt
python fuzzy_safety.py # fuzzy worked example
```

## How one decision is made (every 5 s)
1. Read the 4 detectors (noise σ = 1 vehicle, 2% dropout)
2. Short-term memory: median of the last 6 readings per approach
3. If an approach has 3 missing readings in a row → **FALLBACK** (fixed 30 s plan), log the event, alert the operator
4. State = (green-queue bin, red-queue bin, elapsed-green bin)
5. If the state has been visited < 20 times in training → **LOW_CONFIDENCE**: the fuzzy supervisor decides
6. Otherwise the ML picks `argmax Q(s, a)`. If ML says KEEP but fuzzy urgency ≥ 0.75 → **SAFETY_OVERRIDE** to SWITCH
7. Hard limits: no switch before 10 s of green, forced switch at 90 s
8. Reward = −(total queue) − 2 if switched. During training, apply the Q-update
