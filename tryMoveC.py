import numpy as np
import matplotlib.pyplot as plt
from Controller import AWGController, arc_length_spacing, resample_curvature_weighted, uniform_sampling
import similaritymeasures

ctrl = AWGController(serial_number=24909, realtime_priority=False, f_start_hz = 91.0e6, max_channel_amp_v = 0.65)
ctrl.DDS_TIMER_MIN_NS = 6.4  # override for offline testing

T   = 1000
total_time = 15000e-6
t   = np.linspace(0.0, total_time, T)   # 50 µs — satisfies 44.8 ns/segment exec floor
s   = np.linspace(0.0, 1.0, T)
pos = 35.2 * (10*s**3 - 15*s**4 + 6*s**5)  # min-jerk STA 0 → 6 µm
# pos = 35.2 * np.sin(4*np.pi*s)  # sinusoidal
# pos = 35.2*s
# First plan the move to see the quantized time/freq/...
result = ctrl.plan(t, pos)
r = np.array([35.2 * (10*s**3 - 15*s**4 + 6*s**5) for s in np.linspace(0.0, 1.0, 50)])
ti = np.linspace(0.0, total_time, 50)
result2 = ctrl.plan(ti, r)
# print(result.time_arr*1e3)
# print(np.diff(result.freqs_kt)[0])
print("peak velocity :   {:.2f} $\\mu m/ms$ ".format(np.max(np.diff(result.position_arr)[0]/(np.diff(result.time_arr)*1e3))))
print("average velocity :   {:.2f} $\\mu m/ms$ ".format(np.mean(np.diff(result.position_arr)[0]/(np.diff(result.time_arr)*1e3))))

newp2, newt2 = arc_length_spacing(result.position_arr[0], result.time_arr, 50)
uniformp, uniformt = uniform_sampling(result.position_arr[0], result.time_arr, 50)
uniformp_plan, uniformt_plan = result2.position_arr[0], result2.time_arr
newp, newt = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=1.0)
newpa, newta = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=0.01)
newpb, newtb = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=0.1)
newpc, newtc = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=0.25)
newpd, newtd = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=0.75)
newpe, newte = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=0.9)
newp10, newt10 = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=10.0)
newp100, newt100 = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=100.0)

pos_nnorm, tim_nnorm = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=1.0, normalize=False)
pos_nnorm1, tim_nnorm1 = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=5.0, normalize=False)
pos_nnorm2, tim_nnorm2 = resample_curvature_weighted(result.position_arr[0], result.time_arr, 50, alpha=10.0, normalize=False)

uniform_spacing = np.diff(uniformp)
uniform_spacing_plan = np.diff(uniformp_plan)
spacing = []
distance_to_uniform = []
distance_to_uniform_planned = []
for arr in [newpa, newpb, newpc, newpd, newpe, newp, newp2, uniformp, newp10, newp100, pos_nnorm, pos_nnorm1, pos_nnorm2]:
    s = np.diff(arr)
    spacing.append(s)
    distance_to_uniform.append(np.sqrt(np.mean((s - uniform_spacing)**2)))
    distance_to_uniform_planned.append(np.sqrt(np.mean((s - uniform_spacing_plan)**2)))


method_labels = ['P 0.01', 'P 0.1', 'P 0.25', 'P 0.75', 'P 0.9', 'P 1.0', 'Arc Length', 'Uniform', 'P 10.0', 'P 100.0', 'P 1.0 no norm', 'P 5.0 no norm', 'P 10.0 no norm']

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
ax1.plot(uniformt*1e3, uniformp-2, 'x', label='spacing: uniform sampled', color='green')
ax1.plot(newt2*1e3, newp2+2, 'x', label='spacing: Arc Length', color='green')
ax1.plot(result.time_arr*1e3, result.position_arr[0], label='position')
ax1.plot(newta*1e3, newpa+4, 'x', label='spacing: Curvature Weighted P 0.01', color='red')
ax1.plot(newtb*1e3, newpb+6, 'x', label='spacing: Curvature Weighted P 0.1', color='yellow')
ax1.plot(newtc*1e3, newpc+8, 'x', label='spacing: Curvature Weighted P 0.25', color='pink')
ax1.plot(newtd*1e3, newpd+10, 'x', label='spacing: Curvature Weighted P 0.75', color='purple')
ax1.plot(newte*1e3, newpe+12, 'x', label='spacing: Curvature Weighted P 0.9', color='blue')
ax1.plot(newt*1e3, newp+14, 'x', label='spacing: Curvature Weighted P 1.0', color='orange')
ax1.plot(newt10*1e3, newp10+16, 'x', label='spacing: Curvature Weighted P 10.0', color='black')
ax1.plot(newt100*1e3, newp100+18, 'x', label='spacing: Curvature Weighted P 100.0', color='magenta')
ax1.plot(tim_nnorm*1e3, pos_nnorm-6, 'x', label='spacing: Curvature Weighted P 1.0 no norm', color='cyan')
ax1.plot(tim_nnorm1*1e3, pos_nnorm1-8, 'x', label='spacing: Curvature Weighted P 5.0 no norm', color='brown')
ax1.plot(tim_nnorm2*1e3, pos_nnorm2-10, 'x', label='spacing: Curvature Weighted P 10.0 no norm', color='gray')
ax1.set_xlabel('time [ms]')
ax1.set_ylabel('position [$\\mu m$]')
ax1.legend()

# ax2.bar(method_labels, -np.array(distance_to_uniform), color= "blue", label='distance to uniform spacing (actual)')
# ax2.bar(method_labels, distance_to_uniform_planned, color= "orange", label='distance to uniform spacing (planned)', alpha=0.7)
# ax2.set_xlabel('Method')
# ax2.set_ylabel('RMS distance to uniform spacing [$\\mu m$]')
# ax2.tick_params(axis='x', rotation=45)
# ax2.legend()

# methods_data = [
#     (newpa, newta), (newpb, newtb), (newpc, newtc), (newpd, newtd),
#     (newpe, newte), (newp, newt), (newp2, newt2), (uniformp, uniformt),
#     (newp10, newt10), (newp100, newt100),
#     (pos_nnorm, tim_nnorm), (pos_nnorm1, tim_nnorm1), (pos_nnorm2, tim_nnorm2),
# ]

# frechet_vals, area_vals, dtw_vals, pcm_vals, cl_vals = [], [], [], [], []
# for p, t_arr in methods_data:
#     curve1 = np.column_stack((t_arr, p))
#     curve2 = np.column_stack((t, pos))
#     frechet_vals.append(similaritymeasures.frechet_dist(curve1, curve2))
#     area_vals.append(similaritymeasures.area_between_two_curves(curve1, curve2))
#     dtw_d, _ = similaritymeasures.dtw(curve1, curve2)
#     dtw_vals.append(dtw_d)
#     pcm_vals.append(similaritymeasures.pcm(curve1, curve2))
#     cl_vals.append(similaritymeasures.curve_length_measure(curve1, curve2))

# x = np.arange(len(method_labels))

# ax3b = ax3.twinx()
# ax3c = ax3.twinx()
# ax3d = ax3.twinx()
# ax3e = ax3.twinx()
# ax3c.spines['right'].set_position(('outward', 60))
# ax3d.spines['right'].set_position(('outward', 120))
# ax3e.spines['right'].set_position(('outward', 180))

# metrics = [
#     (ax3,  frechet_vals, 'Fréchet',      'tab:blue'),
#     (ax3b, area_vals,    'Area',         'tab:orange'),
#     (ax3c, dtw_vals,     'DTW',          'tab:green'),
#     (ax3d, pcm_vals,     'PCM',          'tab:red'),
#     (ax3e, cl_vals,      'Curve Length', 'tab:purple'),
# ]
# for ax_i, vals, label, color in metrics:
#     ax_i.plot(x, vals, 'o-', color=color, label=label)
#     ax_i.set_ylabel(label, color=color)
#     ax_i.tick_params(axis='y', labelcolor=color)

# ax3.set_xticks(x)
# ax3.set_xticklabels(method_labels, rotation=45, ha='right')
# ax3.set_xlabel('Method')
# lines  = [ax_i.get_lines()[0] for ax_i, *_ in metrics]
# labels = [line.get_label() for line in lines]
# ax3.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()


# fig2 = plt.figure(figsize=(12, 6))
# # now I wanna plot the time differences to see how the spacing is affected in each
# plt.plot(result.time_arr[1:]*1e3, np.diff(result.position_arr)[0]/(np.diff(result.time_arr)*1e3), marker=False, label='original')
# plt.plot(newta[1:]*1e3, np.diff(newpa)/(np.diff(newta)*1e3)+2, 'x', label='spacing: Curvature Weighted P 0.01', color='red')
# plt.plot(newtb[1:]*1e3, np.diff(newpb)/(np.diff(newtb)*1e3)+4, 'x', label='spacing: Curvature Weighted P 0.1', color='yellow')
# plt.plot(newtc[1:]*1e3, np.diff(newpc)/(np.diff(newtc)*1e3)+6, 'x', label='spacing: Curvature Weighted P 0.25', color='pink')
# plt.plot(newtd[1:]*1e3, np.diff(newpd)/(np.diff(newtd)*1e3)+8, 'x', label='spacing: Curvature Weighted P 0.75', color='purple')
# plt.plot(newte[1:]*1e3, np.diff(newpe)/(np.diff(newte)*1e3)+10, 'x', label='spacing: Curvature Weighted P 0.9', color='blue')
# plt.plot(newt[1:]*1e3, np.diff(newp)/(np.diff(newt)*1e3)+12, 'x', label='spacing: Curvature Weighted P 1.0', color='orange')
# plt.plot(newt2[1:]*1e3, np.diff(newp2)/(np.diff(newt2)*1e3)+14, 'x', label='spacing: Arc Length', color='green')
# plt.xlabel('time [ms]')
# plt.ylabel('speed [$\\mu m/ms$]')
# plt.legend()
# plt.show()

# fig3, axes3 = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
# fig3.suptitle('Position samples vs index (step uniformity)', fontsize=13, fontweight='bold')

# datasets = [
#     (newp2,  'Arc Length',               'green'),
#     (newpa,  r'Curvature Weighted α=0.01', 'red'),
#     (newpb,  r'Curvature Weighted α=0.1',  'goldenrod'),
#     (newpc,  r'Curvature Weighted α=0.25', 'hotpink'),
#     (newp,   r'Curvature Weighted α=0.5',  'orange'),
#     (newpd,  r'Curvature Weighted α=0.75', 'purple'),
#     (newpe,  r'Curvature Weighted α=0.9',  'royalblue'),
#     (result.position_arr[0], 'Original (dense)',    'gray'),
# ]

# for ax, (arr, label, color) in zip(axes3.flat, datasets):
#     ax.plot(arr, 'o-', color=color, lw=1.2, ms=3, label=label)
#     ax.set_title(label, fontsize=9)
#     ax.set_xlabel('Sample index', fontsize=8)
#     ax.set_ylabel('Position [μm]', fontsize=8)
#     ax.grid(True, alpha=0.3)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# fig3.tight_layout()
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(result.time_arr[1:]*1e3, np.diff(result.position_arr)[0]/(np.diff(result.time_arr)*1e3))
# ax.set_xlabel('time [ms]')
# ax.set_ylabel('speed [$\\mu m$/ms]')
# plt.show()

# ctrl.connect()
# n_segments = ctrl.move(t, pos, channel=0, amplitudes=1.0, hold = 0, back_and_forth=True, continuous=True, force_trigger=True)
# input("press enter to exit...\n")
# ctrl.disconnect()

