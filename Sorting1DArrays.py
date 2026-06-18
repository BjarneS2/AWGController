"""
Sorting1DArrays.py
Generate a 1D array of N sites (experimentally K atoms will be loaded into these sites at rndm probability),
assuming an image is taken and the occupancy is fed back as a binary mask; solve the resulting 1D
array with the sliding-window sorter (find_best_window / plan_moves) to find the moves to sort the K atoms
into a filled block of K adjacent sites. The target block is slid to the position where the most atoms are
already sitting in the right spot, which minimizes the number of moves.

For testing the script contains an "offline" test mode to generate a random mask and plot the results without
the need for the hardware connection.

@author: Bjarne Schümann
18.06.2026
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from Controller import AWGController, convert_position_to_freq



N            = 30        # number of trap sites in the 1D array (<= 20 on one channel)
SPACING_UM   = 3.0       # site separation [um]
CENTER_ARRAY = True      # center the sites about position 0 (= f_start)

STEP_TIME_S        = 20e-6   # speed: 3um/20us = 150 um/ms
WAYPOINTS_PER_STEP = 15      # min-jerk samples per single-site step
HOLD_S             = 2.5     # [s] hold at the sorted block for imaging [s]
LOAD_LEAD_S        = 0.5     # [s] plot-only lead: loading point drawn at -LOAD_LEAD_S

SERIAL        = 24909
F_START_HZ    = 91.0e6
MAX_AMPV      = 0.65
MAX_TOTAL     = 0.8
CHANNEL       = 0
FORCE_TRIGGER = True


def find_best_window(mask):
    """
    Find the best target window for the occupancy mask by:
    Assigning each atom an index for a window starting at "start" and the ith atom
    sitting at start+i, so it is already in the correct place iff occupied[i]-i == start
    Then just look for the windows in which most atoms are left untouched -> fewest moves.

    start: index of the first trap of best window.
    sites: (K,) target trap indices (start .. start+K-1).
    in_place: #atoms already in right place
    """
    mask = np.asarray(mask, dtype=bool)
    N = len(mask)
    occupied = np.flatnonzero(mask) # current trap index of each atom, in order
    K = occupied.size
    if K == 0:
        return 0, np.empty(0, dtype=int), 0

    disp = occupied - np.arange(K)
    lo, hi = 0, N - K # valid window starts keep the block in range
    disp = np.clip(disp, lo, hi)
    starts, counts = np.unique(disp, return_counts=True)
    best = int(np.argmax(counts))
    start = int(starts[best])
    in_place = int(counts[best])

    sites = start + np.arange(K)
    return start, sites, in_place

def plan_moves(mask):
    """
    Returns [(from_trap, to_trap),...] for atoms that need to move only.
    """
    _, sites, _ = find_best_window(mask)
    occupied = np.flatnonzero(np.asarray(mask, dtype=bool))
    moves = [(int(src), int(dst)) for src, dst in zip(occupied, sites) if src != dst]
    right = sorted((m for m in moves if m[1] > m[0]), key=lambda m: -m[0])
    left = sorted((m for m in moves if m[1] < m[0]), key=lambda m: m[0])
    return sites, left + right

def site_positions(n, spacing_um, center=True):
    x = np.arange(n) * float(spacing_um)
    if center:
        x = x - x.mean()
    return x

def min_jerk(p0, p1, n):
    """compute min jerk trajectory from p0->p1 in n steps"""
    s = np.linspace(0.0, 1.0, n)
    return p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)

def build_sort_trajectory(mask, positions_um, moves,
                          step_time_s=STEP_TIME_S, wpps=WAYPOINTS_PER_STEP):
    """
    Turn the ordered moves into one (K, T) trajectory on a single axis.
    Each move slides one atom straight from its source site to its target site
    with a minimum-jerk profile. The trajectories are built sequentially, so the
    first move starts from the initial state, the second from the state after the
    first move, and so on. Empty sites are turned OFF at the beginning.
    I think I could tehcnically also do 2 moves at the same time, but here it is
    sequentially just to be save.
    
    Returns (t, P, occ_cols):
        t        : (T,) waypoint times [s]
        P        : (K, T) per-tone positions [um]; row k tracks one physical atom
        occ_cols : the initially occupied columns, in tone order (ascending column)
    """
    mask = np.asarray(mask).astype(bool)
    occ_cols = [c for c in range(len(mask)) if mask[c]]
    K = len(occ_cols)
    if K == 0:
        raise ValueError("empty mask: no atoms loaded")

    tone_at_col = {occ_cols[i]: i for i in range(K)}      # column -> tone

    t = [0.0]
    P = [[positions_um[occ_cols[i]]] for i in range(K)]

    for sc, dc in moves:
        tone = tone_at_col.pop(sc)
        n_steps = abs(dc - sc)
        if n_steps == 0:
            continue
        n = max(2, n_steps * wpps)
        dur = n_steps * step_time_s
        seg_t = np.linspace(t[-1], t[-1] + dur, n)[1:]            # drop dup start
        seg_x = min_jerk(positions_um[sc], positions_um[dc], n)[1:]
        for j in range(K):
            if j == tone:
                P[j].extend(seg_x.tolist())
            else:
                P[j].extend([P[j][-1]] * len(seg_t))
        t.extend(seg_t.tolist())
        tone_at_col[dc] = tone

    return np.asarray(t), np.asarray(P), occ_cols

def plot_trajectories(t, P, positions_um, sites, n_moves):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for c in range(len(positions_um)):
        ax.axhline(positions_um[c], color="#e6e6e6", lw=0.7, zorder=0)
    for c in sites:
        ax.axhline(positions_um[c], color="#ffce85", lw=7, alpha=0.55, zorder=0)
    for k in range(P.shape[0]):
        ax.plot(t * 1e3, P[k], lw=1.7, zorder=3)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("position [$\\mu$m]")
    ax.set_title("1D sort: %d atoms, %d moves, step=%.2f ms/site, hold=%.1f s"
                 % (P.shape[0], n_moves, STEP_TIME_S * 1e3, HOLD_S))
    fig.tight_layout()
    plt.show()
    return fig

def build_freq_amp(mask, t, P, positions_um, ctrl, lead_s=LOAD_LEAD_S):
    """
    THIS IS JUST FOR PLOTTING SAKES!
    Per-core frequency/amplitude over the WHOLE channel (all N sites), with two
    loading waypoints prepended at -lead_s so the initialization phase is shown.

    Power is split equally over all N sites and held constant: every core sits at
    max_total/N during loading and the occupied cores stay there throughout the move
    (no power boost when the empties switch off — that wouldn't be repeatable in the
    experiment). Empty cores carry max_total/N only while loading and drop to 0 at the
    move start. A core's frequency is blanked (NaN) wherever its amplitude is 0, so an
    off tone leaves no line.

    Returns (t_ext, F, A): t_ext (T+2,) [s]; F, A each (N, T+2) [Hz], [frac].
    """
    mask = np.asarray(mask).astype(bool)
    positions_um = np.asarray(positions_um, dtype=float)
    P = np.asarray(P, dtype=float)
    n_sites = len(positions_um)
    occ_cols = [c for c in range(n_sites) if mask[c]]
    row_of_col = {c: k for k, c in enumerate(occ_cols)}    # site column -> tone row in P
    T = P.shape[1]
    tone_amp = ctrl.max_total_amplitude / n_sites          # equal, constant share

    P_all = np.empty((n_sites, T + 2))
    A_all = np.empty((n_sites, T + 2))
    for c in range(n_sites):
        if c in row_of_col:                                # occupied -> moves, always on
            traj = P[row_of_col[c]]
            P_all[c] = np.concatenate([traj[:1], traj[:1], traj])
            A_all[c] = tone_amp
        else:                                              # empty -> parked, on only at load
            P_all[c] = positions_um[c]
            A_all[c, :2] = tone_amp
            A_all[c, 2:] = 0.0

    F_all = convert_position_to_freq(P_all, f_start_hz=ctrl.f_start_hz,
                                     um_per_MHz=ctrl.um_per_MHz)
    F_all = np.where(A_all > 0, F_all, np.nan)             # frequency off when amplitude is 0

    t_ext = np.insert(np.asarray(t, dtype=float), 0, [t[0] - lead_s, t[0] - 6.4e-9])
    return t_ext, F_all, A_all

def plot_freq_amp(mask, t, P, positions_um, sites, ctrl, lead_s=LOAD_LEAD_S):
    """Frequency + amplitude of every core in the channel vs time, including the
    loading/initialization phase drawn at -lead_s. Occupied cores are solid, empty
    (load-only) cores dashed; their lines vanish when they switch off at the move."""
    positions_um = np.asarray(positions_um, dtype=float)
    n_sites = len(positions_um)
    t_ext, F, A = build_freq_amp(mask, t, P, positions_um, ctrl, lead_s=lead_s)
    occ = np.asarray(mask).astype(bool)
    tone_amp = ctrl.max_total_amplitude / n_sites

    fig, (ax_f, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6.4))

    site_f = convert_position_to_freq(positions_um, f_start_hz=ctrl.f_start_hz,
                                      um_per_MHz=ctrl.um_per_MHz)[0]
    for c in range(n_sites):
        ax_f.axhline(site_f[c] / 1e6, color="#e6e6e6", lw=0.7, zorder=0)
    for c in sites:
        ax_f.axhline(site_f[c] / 1e6, color="#ffce85", lw=7, alpha=0.55, zorder=0)

    for c in range(n_sites):
        kw = dict(lw=1.7) if occ[c] else dict(lw=1.3, ls="--", alpha=0.7)
        line, = ax_f.plot(t_ext * 1e3, F[c] / 1e6, zorder=3, **kw)
        ax_a.plot(t_ext * 1e3, A[c], color=line.get_color(), zorder=3, **kw)
        ax_f.plot(t_ext[:2] * 1e3, F[c, :2] / 1e6, "o", ms=5,
                  color=line.get_color(), zorder=4)
        ax_a.plot(t_ext[:2] * 1e3, A[c, :2], "o", ms=5,
                  color=line.get_color(), zorder=4)

    for ax in (ax_f, ax_a):
        ax.axvline(0.0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    ax_f.set_ylabel("frequency [MHz]")
    ax_a.set_ylabel("amplitude [frac]")
    ax_a.set_xlabel("time [ms]   (loading at %.0f ms)" % (-lead_s * 1e3))
    ax_f.set_title("1D sort: %d/%d cores on, constant %.3f/tone (max_total=%.2f)"
                   % (int(occ.sum()), n_sites, tone_amp, ctrl.max_total_amplitude))
    fig.tight_layout()
    plt.show()
    return fig

def replay_states(mask, moves):
    """Occupied-column lists before and after each move: states[0] is the initial
    load, states[k] the occupancy after the k-th move."""
    occ = [c for c in range(len(mask)) if mask[c]]
    states = [list(occ)]
    cur = list(occ)
    for sc, dc in moves:
        cur = sorted(dc if c == sc else c for c in cur)
        states.append(list(cur))
    return states

def plot_slider(mask, sites, moves):
    """Interactive 1D analog of HCA's plot_slider: a row of sites with a slider to
    step through the moves. Atoms inside the target block are green, outside red;
    the current move is drawn as an arrow from source to destination. Needs a GUI
    backend."""
    from matplotlib.patches import Circle
    from matplotlib.widgets import Slider

    mask = np.asarray(mask).astype(bool)
    n_sites = len(mask)
    states = replay_states(mask, moves)
    target = set(int(s) for s in sites)
    n = len(moves)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * n_sites), 2.8))
    plt.subplots_adjust(bottom=0.30)

    def render(step):
        ax.clear()
        if len(sites):
            ax.axvspan(sites[0] - 0.5, sites[-1] + 0.5, color="#ffce85",
                       alpha=0.45, zorder=0)
        for c in range(n_sites):
            ax.add_patch(Circle((c, 0), 0.16, facecolor="none",
                                edgecolor="#bbbbbb", lw=1.1, zorder=1))
        for c in states[step]:
            color = "#4c9f70" if c in target else "#d96459"
            ax.add_patch(Circle((c, 0), 0.33, facecolor=color, edgecolor="none", zorder=2))
        if step > 0:
            sc, dc = moves[step - 1]
            ax.annotate("", xy=(dc, 0), xytext=(sc, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.2,
                                        shrinkA=8, shrinkB=8), zorder=3)
            ax.add_patch(Circle((sc, 0), 0.30, facecolor="none",
                                edgecolor="#666666", lw=1.8, zorder=4))
        title = "initial" if step == 0 else \
            "after move %d:  col %d -> col %d" % (step, moves[step - 1][0], moves[step - 1][1])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1, n_sites)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_yticks([])
        ax.set_xticks(range(0, n_sites, max(1, n_sites // 15)))
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        fig.canvas.draw_idle()

    sax = plt.axes((0.16, 0.10, 0.68, 0.05))
    slider = Slider(sax, "step", 0, max(n, 1), valinit=0, valstep=1)
    slider.on_changed(lambda v: render(int(v)))
    render(0)
    plt.show()
    return fig, slider

def report_bandwidth(ctrl, t, P, n_sites=N):
    """Offline timing/FIFO sanity check via plan()."""
    res = ctrl.plan(t, P, channel=CHANNEL, hold=HOLD_S,
                    amplitudes=ctrl.max_total_amplitude / n_sites, skip_timing_check=True)
    K, T = res.freqs_kt.shape
    n_cmds = ctrl._estimate_n_commands(K, T)
    cap = ctrl._queue_max_actual or ctrl.SINGLE_MODE_QUEUE_MAX
    dt = np.diff(res.time_arr)
    print("  K=%d tones, T=%d waypoints, min dt=%.2f us"
          % (K, T, dt.min() * 1e6))
    print("  ~%d DDS commands vs FIFO %d  ->  %s"
          % (n_cmds, cap, "STREAMING" if n_cmds > cap else "single-shot"))
    print("  freq band used: [%.3f, %.3f] MHz"
          % (res.freqs_kt.min() / 1e6, res.freqs_kt.max() / 1e6))

def emulate(seed=0, fill_prob=0.6, mask=None):
    """
    Either pass a mask yourself or generate a random one given the 
    hardcoded variables above.
    """
    if mask is None:
        rng = np.random.default_rng(seed)
        mask = (rng.random(N) < fill_prob).astype(int)
        if mask.sum() == 0:
            mask[int(rng.integers(N))] = 1
    else:
        mask = np.asarray(mask).astype(int) # accept a list/array of 0/1
    n = len(mask)  # array size follows the given mask
    positions = site_positions(n, SPACING_UM, CENTER_ARRAY)

    print("loaded : " + "".join("#" if x else "." for x in mask)
          + "   (%d atoms)" % int(mask.sum()))
    sites, moves = plan_moves(mask)
    print("target : sites=%d..%d  block size=%d  moves=%d"
          % (sites[0], sites[-1], len(sites), len(moves)))
    for k, (sc, dc) in enumerate(moves, 1):
        print("  move %2d:  col %d -> col %d" % (k, sc, dc))

    plot_slider(mask, sites, moves)

    if not moves:
        print("already sorted - nothing to move.")
        return

    t, P, _ = build_sort_trajectory(mask, positions, moves)

    ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                         max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                         realtime_priority=False)
    print("playback: %.3f ms sort + %.1f s hold" % (t[-1] * 1e3, HOLD_S))
    report_bandwidth(ctrl, t, P, n_sites=n)

    # append the hold for visualization
    t_full = np.append(t, t[-1] + HOLD_S)
    P_full = np.concatenate([P, P[:, -1:]], axis=1)
    plot_trajectories(t_full, P_full, positions, sites, len(moves))
    plot_freq_amp(mask, t_full, P_full, positions, sites, ctrl)

def monte_carlo(trials=10000, fill_prob=0.5, seed0=71):
    counts, atoms = [], []
    for sd in range(seed0, seed0 + trials):
        rng = np.random.default_rng(sd)
        mask = (rng.random(N) < fill_prob).astype(int)
        if mask.sum() < 1:
            continue
        counts.append(len(plan_moves(mask)[1]))
        atoms.append(int(mask.sum()))
    counts, atoms = np.array(counts), np.array(atoms)
    print("MC over %d loads (p=%.2f, N=%d):  <atoms>=%.2f  <moves>=%.2f  max moves=%d  max atoms=%d"
          % (len(counts), fill_prob, N, atoms.mean(), counts.mean(), counts.max(), atoms.max()))
    # print moves per atoms and max atoms
    print("%.4f moves per atom on average"
          % (counts.mean()/atoms.mean()))

def acquire_mask(n):
    """Lab hook: replace with the camera readout -> length-n binary occupancy.
    Default reads the mask from the keyboard so the loop can be driven by hand."""
    raw = input("mask (%d bits, e.g. 1011..) or blank to quit: " % n).strip()
    if raw == "":
        return None
    bits = [int(ch) for ch in raw if ch in "01"]
    if len(bits) != n:
        print("  expected %d bits, got %d" % (n, len(bits)))
        return acquire_mask(n)
    return np.array(bits, int)

def run_in_the_lab_idea():
    positions = site_positions(N, SPACING_UM, CENTER_ARRAY)
    all_on = np.ones(N, dtype=bool)

    ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                         max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                         realtime_priority=True)
    ctrl.connect()
    try:
        ctrl.program_static(positions, all_on, channel=CHANNEL)   # initial N states
        while True:
            mask = acquire_mask(N)
            if mask is None:
                break
            if mask.sum() == 0:
                print("  no atoms - re-arming and waiting.")
                continue

            sites, moves = plan_moves(mask)
            print("  %d atoms, target sites %d..%d, %d moves"
                  % (int(mask.sum()), sites[0], sites[-1], len(moves)))

            if moves:
                t, P, _ = build_sort_trajectory(mask, positions, moves)
                ctrl.move(t, P, channel=CHANNEL, hold=HOLD_S,
                          amplitudes=ctrl.max_total_amplitude / N,   # equal, constant: matches loading
                          force_trigger=FORCE_TRIGGER)
                if FORCE_TRIGGER:
                    time.sleep(t[-1] + HOLD_S + 0.1)   # let sort + imaging finish
                else:
                    input("  armed, waiting for EXT0. press enter once imaged...\n")
            else:
                print("  already sorted.")
                time.sleep(HOLD_S)

            ctrl.stop()                                               # clear move/stream
            ctrl.program_static(positions, all_on, channel=CHANNEL)   # back to start
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    monte_carlo()
    emulate(seed=42, fill_prob=0.5)  #, mask=np.array([0,1,1,1,0,0,1,0,1,1,1,1,0,0,1,0,1]))

    
