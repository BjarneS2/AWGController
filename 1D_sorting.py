"""
1D_sorting.py

Atom rearrangement in a 1D optical tweezer array using an AWGController.

Given an array of trap frequencies or positions and a boolean occupancy array,
these functions set static tones and sort atoms into a compact target configuration.

Frequency / position convention
--------------------------------
Values > 1e5 are interpreted as frequencies in Hz.
Values <= 1e5 are interpreted as positions in µm.
The conversion is linear and monotonic, so ascending frequency == ascending position.
Core assignment always follows ascending frequency order: the lowest-frequency site
maps to the channel's first core, the next to the second core, and so on.
"""

import time
import warnings
import numpy as np
from typing import Literal
from Controller import AWGController, convert_position_to_freq


_FREQ_THRESHOLD = 1e5   # above this → Hz, at or below → µm


def _to_um(values: np.ndarray, ctrl: AWGController) -> np.ndarray:
    """Convert Hz → µm if values look like frequencies, otherwise return as-is."""
    if values.size > 0 and float(values.max()) > _FREQ_THRESHOLD:
        return (values - ctrl.f_start_hz) / 1e6 * ctrl.um_per_MHz
    return values.copy()


# ---------------------------------------------------------------------------
# Buffer check
# ---------------------------------------------------------------------------

def check_and_flush(ctrl: AWGController, flush: bool = False) -> bool:
    """
    Check whether the controller has pending DDS activity.

    Returns True if busy. If flush=True, stops all activity and returns False.
    Checks both the streaming thread and the on-chip DDS command FIFO.
    """
    streaming = ctrl._stream_thread is not None and ctrl._stream_thread.is_alive()
    replay    = ctrl._replay_mode_active

    if streaming or replay:
        if flush:
            ctrl.stop_replay() if replay else ctrl.stop()
            return False
        warnings.warn(
            "[sorting] Controller is busy (streaming or replay active). "
            "Pass flush=True to abort.",
            UserWarning, stacklevel=2,
        )
        return True

    if ctrl.card is not None:
        try:
            import spcm
            pending = int(ctrl.card.get_i(spcm.SPC_DDS_QUEUE_CMD_COUNT))
            if pending > 0:
                if flush:
                    ctrl.stop()
                    return False
                warnings.warn(
                    f"[sorting] DDS FIFO has {pending} pending commands. "
                    "Pass flush=True to clear.",
                    UserWarning, stacklevel=2,
                )
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _min_jerk(x0: float, x1: float, t: np.ndarray) -> np.ndarray:
    if abs(x1 - x0) < 1e-9:
        return np.full_like(t, x0)
    s = (t - t[0]) / (t[-1] - t[0])
    return x0 + (x1 - x0) * (10*s**3 - 15*s**4 + 6*s**5)


def _smooth_amp_profile(
    a_trap: float, a_transport: float, n: int, ramp_frac: float = 0.15
) -> np.ndarray:
    """Cosine-tapered amplitude ramp: a_trap → a_transport → a_trap."""
    out = np.full(n, float(a_transport))
    nr  = max(2, int(n * ramp_frac))
    tap = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nr)))
    out[:nr]  = a_trap + (a_transport - a_trap) * tap
    out[-nr:] = a_transport + (a_trap - a_transport) * tap
    return out


# ---------------------------------------------------------------------------
# Main sorting function
# ---------------------------------------------------------------------------

def sort_1d(
    ctrl: AWGController,
    freqs_or_positions: np.ndarray,
    occupancy: np.ndarray,
    channel: int = 0,
    target_offset_um: float = 0.0,
    target_spacing_um: float | None = None,
    mode: Literal['simultaneous', 'individual'] = 'simultaneous',
    total_time_s: float | None = None,
    speed_um_s: float | None = None,
    n_waypoints: int = 100,
    trap_amplitude: float | np.ndarray | None = None,
    transport_amplitude: float | np.ndarray | None = None,
    flush_buffer: bool = False,
    force_trigger: bool = False,
    blocking: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rearrange atoms from a sparse occupancy to a compact sorted target array.

    Accepts frequencies in Hz (values > 1e5) or positions in µm.
    Cores are assigned in ascending frequency order (lowest freq = core 0).
    Occupied atoms are mapped in order to target sites starting at target_offset_um.

    All N cores are programmed on each move() call so non-moving traps remain
    held. Empty sites have amplitude 0 throughout.

    Modes
    -----
    'simultaneous'  All atoms move at the same time (total_time_s required).
    'individual'    Atoms move one at a time at the given speed (speed_um_s required).
                    With blocking=True (default) each atom finishes before the next starts.

    Args:
        ctrl:                AWGController (connected).
        freqs_or_positions:  (N,) trap site frequencies [Hz] or positions [µm].
        occupancy:           (N,) bool — True where an atom is present.
        channel:             0 (X-AOD) or 1 (Y-AOD).
        target_offset_um:    Position of the first target site [µm].
        target_spacing_um:   Spacing between target sites [µm].
                             None → infer from median spacing of input sites.
        mode:                'simultaneous' or 'individual'.
        total_time_s:        Transport duration for simultaneous mode [s].
        speed_um_s:          Peak transport speed for individual mode [µm/s].
        n_waypoints:         DDS waypoints per trajectory.
        trap_amplitude:      Amplitude during static trapping. None → equal share
                             of max_total_amplitude. Scalar or (n_atoms,) array.
        transport_amplitude: Amplitude during transport. None → same as trap_amplitude.
                             Use >= trap_amplitude for stronger confinement during motion.
        flush_buffer:        Stop active streaming/replay before sorting.
        force_trigger:       Software-trigger (bypass EXT0).
        blocking:            In individual mode: sleep after each atom so it finishes
                             before the next move is programmed.

    Returns:
        final_positions_um:  (n_atoms,) target positions [µm] in sorted order.
        final_occupancy:     (N,) bool occupancy updated to reflect sorted state.
                             A target site sets the nearest input site to True when
                             within half a spacing; otherwise all False for that target.
    """
    if ctrl.card is None or ctrl.dds is None:
        raise RuntimeError("AWGController not connected. Call connect() first.")

    # if check_and_flush(ctrl, flush=flush_buffer):
    #     raise RuntimeError("[sorting] Controller busy. Pass flush_buffer=True to force.")

    v = np.asarray(freqs_or_positions, dtype=float)
    o = np.asarray(occupancy, dtype=bool)
    N = len(v)
    if len(o) != N:
        raise ValueError("freqs_or_positions and occupancy must have the same length.")

    pos_um = _to_um(v, ctrl)    # (N,) positions in µm

    n_atoms = int(o.sum())
    if n_atoms == 0:
        warnings.warn("[sorting] No atoms to sort.", UserWarning, stacklevel=2)
        return np.array([]), o.copy()

    # Ascending frequency order defines the core assignment
    sort_order  = np.argsort(pos_um)            # sort_order[rank] = input index
    sorted_pos  = pos_um[sort_order]            # (N,) ascending
    sorted_occ  = o[sort_order]                 # (N,) occupancy in core rank order

    occupied_ranks = np.where(sorted_occ)[0]    # core ranks that have atoms
    empty_ranks    = np.where(~sorted_occ)[0]
    initial_pos    = sorted_pos[occupied_ranks] # (K,) initial positions

    # Target geometry
    if target_spacing_um is None:
        diffs = np.diff(sorted_pos)
        target_spacing_um = float(np.median(diffs)) if len(diffs) > 0 else 1.0

    K          = n_atoms
    target_pos = target_offset_um + np.arange(K, dtype=float) * target_spacing_um

    target_freqs = convert_position_to_freq(
        target_pos[np.newaxis, :], f_start_hz=ctrl.f_start_hz, um_per_MHz=ctrl.um_per_MHz
    )[0]
    if np.any(target_freqs < ctrl.f_min_hz) or np.any(target_freqs > ctrl.f_max_hz):
        raise ValueError(
            f"Target positions leave AOD band "
            f"[{ctrl.f_min_hz/1e6:.1f}, {ctrl.f_max_hz/1e6:.1f}] MHz. "
            f"Freq range [{target_freqs.min()/1e6:.2f}, {target_freqs.max()/1e6:.2f}] MHz."
        )

    # Amplitudes
    def _parse_amp(amp: float | np.ndarray | None, label: str) -> np.ndarray:
        if amp is None:
            return np.full(K, ctrl.max_total_amplitude / K)
        if np.isscalar(amp):
            return np.full(K, float(amp))
        a = np.asarray(amp, dtype=float)
        if a.shape != (K,):
            raise ValueError(f"{label} must be scalar or ({K},). Got {a.shape}.")
        return a

    trap_amps  = _parse_amp(trap_amplitude,      "trap_amplitude")
    trans_amps = _parse_amp(transport_amplitude, "transport_amplitude")
    if transport_amplitude is None:
        trans_amps = trap_amps.copy()

    if np.any(trans_amps < trap_amps - 1e-10):
        warnings.warn(
            "[sorting] transport_amplitude < trap_amplitude for some atoms: "
            "traps will be weakened during transport.",
            UserWarning, stacklevel=2,
        )

    # (N,) static amplitude for all cores in sorted rank order (0 for empty)
    static_amps = np.zeros(N)
    for ki, rank in enumerate(occupied_ranks):
        static_amps[rank] = trap_amps[ki]

    # ---- Simultaneous mode -----------------------------------------------
    if mode == 'simultaneous':
        if total_time_s is None:
            raise ValueError("sort_1d: total_time_s required for mode='simultaneous'.")

        T        = n_waypoints
        time_arr = np.linspace(0.0, total_time_s, T)

        pos_arr  = np.tile(sorted_pos[:, np.newaxis], (1, T))     # (N, T) constant baseline
        amps_arr = np.tile(static_amps[:, np.newaxis], (1, T))

        for ki, rank in enumerate(occupied_ranks):
            pos_arr[rank]  = _min_jerk(initial_pos[ki], target_pos[ki], time_arr)
            amps_arr[rank] = _smooth_amp_profile(trap_amps[ki], trans_amps[ki], T)
        amps_arr[empty_ranks] = 0.0

        n_segs = ctrl.move(
            time_arr, pos_arr,
            channel=channel, amplitudes=amps_arr, force_trigger=force_trigger,
        )
        print(
            f"[sorting] Simultaneous: {K} atoms → {total_time_s*1e6:.1f} µs, "
            f"{n_segs} segments on CH{channel}.\n"
        )

    # ---- Individual mode -------------------------------------------------
    elif mode == 'individual':
        if speed_um_s is None:
            raise ValueError("sort_1d: speed_um_s required for mode='individual'.")

        displacements = np.abs(target_pos - initial_pos)
        min_duration  = ctrl.DDS_TIMER_MIN_NS * 1e-9 * n_waypoints
        durations     = np.maximum(displacements / speed_um_s, min_duration)

        current_sorted = sorted_pos.copy()  # updated as atoms move

        for ki, rank in enumerate(occupied_ranks):
            if displacements[ki] < 1e-3:
                print(f"[sorting] Atom {ki} (rank {rank}) already at target, skipping.\n")
                current_sorted[rank] = target_pos[ki]
                continue

            T_k    = n_waypoints
            time_k = np.linspace(0.0, durations[ki], T_k)

            pos_k     = np.tile(current_sorted[:, np.newaxis], (1, T_k))
            pos_k[rank] = _min_jerk(current_sorted[rank], target_pos[ki], time_k) # type: ignore

            amps_k = np.tile(static_amps[:, np.newaxis], (1, T_k))
            amps_k[empty_ranks] = 0.0
            amps_k[rank] = _smooth_amp_profile(trap_amps[ki], trans_amps[ki], T_k)

            n_segs = ctrl.move(
                time_k, pos_k,
                channel=channel, amplitudes=amps_k, force_trigger=force_trigger,
            )
            print(
                f"[sorting] Atom {ki} (rank {rank}): "
                f"{current_sorted[rank]:.2f} → {target_pos[ki]:.2f} µm, "
                f"{durations[ki]*1e6:.1f} µs, {n_segs} segments.\n"
            )
            current_sorted[rank] = target_pos[ki]

            if blocking:
                time.sleep(durations[ki] * 1.05 + 0.002)  # 5 % margin + 2 ms overhead

    else:
        raise ValueError(f"mode must be 'simultaneous' or 'individual'. Got '{mode}'.")

    # Map final target positions back to input grid sites
    final_occupancy = np.zeros(N, dtype=bool)
    half_spacing    = target_spacing_um * 0.5
    for k in range(K):
        dists   = np.abs(pos_um - target_pos[k])
        nearest = int(np.argmin(dists))
        if dists[nearest] < half_spacing:
            final_occupancy[nearest] = True

    return target_pos, final_occupancy
