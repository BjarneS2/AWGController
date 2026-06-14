import numpy as np
import warnings
import matplotlib.pyplot as plt
from Controller import AWGController

ctrl = AWGController(serial_number=0, realtime_priority=False)
ctrl.DDS_TIMER_MIN_NS = 6.4  # override for offline testing

T   = 100
t   = np.linspace(0.0, 6.4e-6, T)
s   = np.linspace(0.0, 1.0, T)
pos = 6.0 * (10*s**3 - 15*s**4 + 6*s**5)  # min-jerk STA 0 → 6 µm
amplitudes = np.ones_like(pos) * 0.8  # constant amplitude (80% of max) for all waypoints

tick_s = 6.4 * 1e-9
ticks = np.round(t / tick_s).astype(np.int64)
unique_ticks, _ = np.unique(ticks, return_index=True)

result = ctrl.plan(t, pos)

tick_s   = ctrl.DDS_TIMER_MIN_NS * 1e-9
t_q      = result.time_arr
pos_q    = result.position_arr[0]
freq_q   = result.freqs_kt[0]
dt_q     = np.diff(t_q)
n_unique = len(t_q)

print(np.any(t_q<0), np.any(pos_q<0), np.any(freq_q<0), np.any(dt_q<0))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"MovePlanner sanity check  |  T_in={T}, T_quantized={n_unique}, tick={ctrl.DDS_TIMER_MIN_NS} ns")

ax = axes[0, 0]
ax.plot(t * 1e6, pos, "C0", lw=1, label="input")
ax.plot(t_q * 1e6, pos_q, "C1--", lw=1.2, label="quantized")
ax.set_xlabel("time (µs)")
ax.set_ylabel("position (µm)")
ax.set_title("Trajectory")
ax.legend()

ax = axes[0, 1]
ax.plot(t_q * 1e6, freq_q / 1e6, "C2", lw=1.2)
ax.set_xlabel("time (µs)")
ax.set_ylabel("frequency (MHz)")
ax.set_title("DDS frequency")

ax = axes[1, 0]
ax.plot(t_q[:-1] * 1e6, dt_q * 1e9, color="C3", lw=1.0)
ax.axhline(tick_s * 1e9, color="k", ls="--", lw=1, label=f"1 tick = {tick_s*1e9:.1f} ns")
ax.set_xlabel("time (µs)")
ax.set_ylabel("Δt (ns)")
ax.set_title("Time step over time")
ax.legend()

ax = axes[1, 1]
cmd_types = [c["type"] for c in result.command_chain]
labels, counts = np.unique(cmd_types, return_counts=True)
ax.bar(labels, counts, color="C4")
ax.set_ylabel("count")
ax.set_title(f"Command chain  ({len(result.command_chain)} total)")
ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()
