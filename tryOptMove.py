import numpy as np
import matplotlib.pyplot as plt
from Controller import AWGController, arc_length_spacing, resample_curvature_weighted, uniform_sampling, load_control_protocol
from pathlib import Path

PROTOCOL_PATH = Path(r"C:\\dev\\GitHub\\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\\scripts\\results\\control3d_thermal_2026-05-07_15-36-53.h5")


t_us, ux_um, uy_um, ua = load_control_protocol(PROTOCOL_PATH)
print(np.max(ua), np.min(ua))
total_time = t_us[-1] 
T_CTRL = len(t_us) // 10
t = np.linspace(0.0, total_time, T_CTRL) # or use 10000 hardcoded
pos_data = np.array(ux_um) # since we only do 1D, I ignore y component
POS_FIN = pos_data[-1]
POS_START = pos_data[0]
resamp_pos, resamp_time = resample_curvature_weighted(pos_data, t_us, T_CTRL, alpha=1.0)


ctrl = AWGController(serial_number=24909, realtime_priority=False, f_start_hz = 91.0e6, max_channel_amp_v = 0.65)
ctrl.DDS_TIMER_MIN_NS = 6.4
result_opt = ctrl.plan(t_us, pos_data, amplitudes=ua)



T   = 1000
# total_time = 15000e-6
t   = np.linspace(0.0, total_time, T)
s   = np.linspace(0.0, 1.0, T)
pos = (POS_FIN-POS_START) * (10*s**3 - 15*s**4 + 6*s**5) + POS_START
result = ctrl.plan(t, pos)
newp, newt = resample_curvature_weighted(result.position_arr[0], result.time_arr, T_CTRL, alpha=1.0)

# velocity info (µm/µs = m/s)
for label, times, positions in [
    ("Polynomial trajectory", t, pos),
    ("Protocol trajectory",   t_us, pos_data),
]:
    v = np.abs(np.diff(positions) / np.diff(times))  # m/s
    print(f"{label}:")
    print(f"  avg speed : {v.mean():.4f} m/s")
    print(f"  max speed : {v.max():.4f} m/s")

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
ax1.plot(t*1e-3, pos, 'x-', label='original position', color='blue')
ax1.plot(newt*1e-3, newp+1, 'x-', label='spacing: curvature weighted', color='green')
ax1.plot(t_us*1e-3, pos_data, 'x-', label='original position (from protocol)', color='orange')
ax1.plot(resamp_time*1e-3, resamp_pos-1, 'x-', label='spacing: curvature weighted (from protocol)', color='red')

ax2 = ax1.twinx()
ax2.plot(t_us*1e-3, ua, label='amplitude', color='purple', alpha=0.6)
ax2.plot(result_opt.time_arr*1e-3, result_opt.amps_kt[0], label='amplitude', color='magenta', alpha=0.9)
ax2.set_ylabel('amplitude')
ax2.legend(loc='upper right')

ax1.set_xlabel('time [ms]')
ax1.set_ylabel('position [$\\mu m$]')
ax1.legend(loc='upper left')
plt.tight_layout()  
plt.show()



