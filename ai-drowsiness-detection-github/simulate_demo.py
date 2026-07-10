"""
simulate_demo.py
-----------------
A camera-free, network-free demo you can run anywhere (including this sandbox)
to SEE the real decision logic (utils.DrowsinessStateMachine) work end to end.

It does NOT fake the alert output — it feeds synthetically generated
EAR / MAR / head-pitch signals (representing 60 seconds of driving) through
the exact same DrowsinessStateMachine class used by main.py, frame by frame,
at a simulated 30 FPS. This proves the thresholding/state-machine logic is
correct before you ever plug in a webcam.

Scenario simulated:
    0s  - 15s : normal driving (occasional natural blinks)
    15s - 22s : a yawn                                   -> should NOT alone trigger yet
    22s - 35s : normal driving, one more yawn at ~30s     -> 2 yawns/60s SHOULD trigger
    35s - 45s : back to normal
    45s - 50s : eyes slowly close and STAY closed         -> SHOULD trigger eyes_closed
    50s - 60s : head drops forward (nod-off)              -> SHOULD trigger head_nod

Run:
    python simulate_demo.py
Outputs:
    - console log of every state transition (AWAKE <-> DROWSY) with the reason
    - assets/simulation_result.png  (EAR/MAR/Pitch traces + alert regions)
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import DrowsinessStateMachine

FPS = 30
DURATION_S = 60
N = FPS * DURATION_S
t = np.linspace(0, DURATION_S, N)

# ---------------------------------------------------------------
# 1. Build synthetic EAR signal: baseline ~0.30 with natural blinks,
#    plus a sustained closed-eye episode at 45-50s.
# ---------------------------------------------------------------
ear = np.full(N, 0.30) + np.random.normal(0, 0.01, N)


def add_blink(signal, center_s, width_s=0.15, depth=0.24):
    idx = (np.abs(t - center_s)).argmin()
    width = int(width_s * FPS)
    for i in range(max(0, idx - width), min(N, idx + width)):
        d = 1 - abs(i - idx) / width
        signal[i] -= depth * d
    return signal


for blink_time in [2, 5, 9, 12, 18, 24, 28, 33, 38, 41]:
    ear = add_blink(ear, blink_time)

# sustained eye closure: 45s-50s (drowsy micro-sleep)
closed_start, closed_end = int(45 * FPS), int(50 * FPS)
ear[closed_start:closed_end] = 0.09 + np.random.normal(0, 0.005, closed_end - closed_start)

# ---------------------------------------------------------------
# 2. Build synthetic MAR signal: baseline ~0.25, with 2 yawns.
# ---------------------------------------------------------------
mar = np.full(N, 0.25) + np.random.normal(0, 0.01, N)


def add_yawn(signal, start_s, dur_s=4.0, peak=0.85):
    start_idx = int(start_s * FPS)
    dur = int(dur_s * FPS)
    for i in range(dur):
        idx = start_idx + i
        if idx >= N:
            break
        phase = np.sin(np.pi * i / dur)  # smooth rise and fall
        signal[idx] = 0.25 + (peak - 0.25) * phase
    return signal


mar = add_yawn(mar, 16)   # yawn #1 ~16-20s
mar = add_yawn(mar, 30)   # yawn #2 ~30-34s

# ---------------------------------------------------------------
# 3. Build synthetic head pitch (degrees): baseline ~0 (looking at road),
#    with a forward nod-off starting at 50s.
# ---------------------------------------------------------------
pitch = np.full(N, 0.0) + np.random.normal(0, 1.0, N)
nod_start = int(50 * FPS)
for i in range(nod_start, N):
    progress = min(1.0, (i - nod_start) / (5 * FPS))
    pitch[i] -= 22 * progress  # chin drops ~22 degrees

# ---------------------------------------------------------------
# 4. Run the REAL state machine frame-by-frame
# ---------------------------------------------------------------
sm = DrowsinessStateMachine()
states = []
reason_log = []
prev_state = "AWAKE"

print(f"{'time(s)':>8} | {'EAR':>6} | {'MAR':>6} | {'pitch':>7} | state   | reasons")
print("-" * 70)

for i in range(N):
    now = t[i]
    # calibrate on the first 1 second, like the real app does
    if i < FPS:
        sm.calibrate_baseline(pitch[i])

    state, reasons = sm.update(ear[i], mar[i], pitch[i], now=now)
    states.append(1 if state == "DROWSY" else 0)
    reason_log.append(reasons)

    if state != prev_state:
        print(f"{now:8.2f} | {ear[i]:6.2f} | {mar[i]:6.2f} | {pitch[i]:7.1f} | {state:<7} | {reasons}")
        prev_state = state

states = np.array(states)

# ---------------------------------------------------------------
# 5. Plot everything
# ---------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

axes[0].plot(t, ear, color="#2563eb")
axes[0].axhline(0.21, color="red", linestyle="--", linewidth=1, label="EAR threshold")
axes[0].set_ylabel("EAR")
axes[0].set_title("Eye Aspect Ratio (lower = eyes more closed)")
axes[0].legend(loc="upper right", fontsize=8)

axes[1].plot(t, mar, color="#059669")
axes[1].axhline(0.6, color="red", linestyle="--", linewidth=1, label="MAR threshold (yawn)")
axes[1].set_ylabel("MAR")
axes[1].set_title("Mouth Aspect Ratio (higher = mouth more open)")
axes[1].legend(loc="upper right", fontsize=8)

axes[2].plot(t, pitch, color="#7c3aed")
axes[2].set_ylabel("Pitch (deg)")
axes[2].set_title("Head Pitch (drop = nodding off / looking down)")

axes[3].fill_between(t, states, color="#dc2626", step="mid", alpha=0.6)
axes[3].set_ylim(-0.1, 1.1)
axes[3].set_yticks([0, 1])
axes[3].set_yticklabels(["AWAKE", "DROWSY"])
axes[3].set_xlabel("Time (seconds)")
axes[3].set_title("Final Alert Output (DrowsinessStateMachine)")

plt.tight_layout()
plt.savefig("assets/simulation_result.png", dpi=140)
print("\nSaved plot -> assets/simulation_result.png")

drowsy_pct = 100 * states.mean()
print(f"Total time flagged DROWSY: {drowsy_pct:.1f}% of the 60s simulation")
