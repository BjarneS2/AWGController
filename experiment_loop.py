"""
experiment_loop.py

1D tweezer sorting experiment loop.
"""

import importlib
import json
import socket
import numpy as np
from Controller import AWGController

_sorting = importlib.import_module("1D_sorting")
sort_1d = _sorting.sort_1d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERIAL          = 24909
AMP_V           = 0.65

CHANNEL         = 0
N_TRAPS         = 10
FREQ_START_HZ   = 91.0e6
FREQ_SPACING_HZ = 1.0e6
TRANSPORT_S     = 500e-6

HOST = "localhost"
PORT = 5005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trap_positions_um(ctrl):
    freqs = FREQ_START_HZ + np.arange(N_TRAPS) * FREQ_SPACING_HZ
    return (freqs - ctrl.f_start_hz) / 1e6 * ctrl.um_per_MHz


def _initialize(ctrl, positions_um):
    occupancy = np.ones(N_TRAPS, dtype=bool)
    ctrl.program_static(positions_um, occupancy)


def _recv_line(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("Client disconnected.")
        buf += chunk
    return buf.decode().strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(ctrl):
    positions_um = _trap_positions_um(ctrl)
    offset_um    = float(positions_um.min())
    spacing_um   = FREQ_SPACING_HZ / 1e6 * ctrl.um_per_MHz

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print(f"Connected: {addr}")

    try:
        cycle = 0
        while True:
            _initialize(ctrl, positions_um)
            conn.sendall(b"ready\n")

            raw       = _recv_line(conn)
            msg       = json.loads(raw)
            occupancy = np.array(msg["occupancy"], dtype=bool)

            if len(occupancy) != N_TRAPS:
                conn.sendall(f"error: expected {N_TRAPS} entries\n".encode())
                continue

            n_atoms = int(occupancy.sum())
            print(f"Cycle {cycle}: {n_atoms}/{N_TRAPS} atoms — sorting")

            sort_1d(
                ctrl, positions_um, occupancy,
                channel=CHANNEL,
                target_offset_um=offset_um,
                target_spacing_um=spacing_um,
                total_time_s=TRANSPORT_S,
                force_trigger=True,
            )
            conn.sendall(b"sorted\n")
            cycle += 1

            cmd = json.loads(_recv_line(conn)).get("cmd", "next")
            if cmd == "stop":
                conn.sendall(b"stopping\n")
                print("Stop received.")
                break

    finally:
        conn.close()
        server.close()
        ctrl.stop()
        print("Traps off.")


if __name__ == "__main__":
    ctrl = AWGController(
        serial_number=SERIAL,
        max_channel_amp_v=AMP_V,
        f_start_hz=FREQ_START_HZ,
    )
    ctrl.connect()
    try:
        run(ctrl)
    finally:
        ctrl.disconnect()
