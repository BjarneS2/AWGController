import spcm
from spcm import units
import numpy as np
import psutil
import os

p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)
card : spcm.Card

f_start_hz = 91e6      # start frequency [Hz]
f_stop_hz = 96e6     # stop  frequency [Hz]

amp_value = 0.4 * units.V 
threshold = 0.99 * units.V
if amp_value > threshold: 
    print("ERROR: you are exceeding the threshold of 1V")
    quit()

a_start_pct = 100.0      # start amplitude [%]
a_stop_pct = 100.0      # stop  amplitude [%]
T_move_s = 5000e-3    # total sweep duration [s]
N = 100        # number of waypoints  -> N-1 linear segments

# Min-jerk profile in [0, 1] -- S shaped curve with zero velocity and acceleration at start and end
s_arr        = np.linspace(0.0, 1.0, N)
sta_profile  = 10.0*s_arr**3 - 15.0*s_arr**4 + 6.0*s_arr**5
#sta_profile = s_arr
freqs_hz = f_start_hz  + (f_stop_hz  - f_start_hz)  * sta_profile   # (N,) Hz
amps_pct = a_start_pct + (a_stop_pct - a_start_pct) * sta_profile   # (N,) %

dt_s = T_move_s / (N - 1)
freq_slopes_hz_per_s = np.diff(freqs_hz) / dt_s      # (N-1,) Hz/s
amp_slopes_pct_per_s = np.diff(amps_pct) / dt_s      # (N-1,) %/s

print(any(freq_slopes_hz_per_s < 0))


print(f"STA move: {f_start_hz/1e6:.1f} -> {f_stop_hz/1e6:.1f} MHz, "
      f"{a_start_pct:.0f}% -> {a_stop_pct:.0f}%, "
      f"{T_move_s*1e6:.0f} us, {N} waypoints (dt = {dt_s*1e6:.2f} us).")

print(np.any(freqs_hz<0), np.any(amps_pct<0), np.any(freq_slopes_hz_per_s<0), np.any(amp_slopes_pct_per_s<0))

exit()
with spcm.Card(serial_number=24909) as card:

    print(f"Name: {card.product_name()}: SN={card.sn()}, num_channels={card.num_channels()}")
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # setup channels
    channels = spcm.Channels(card)
    for ch in channels:
        ch.enable(True)
        ch.diff(False)
        ch.output_load(50 * units.ohm)
        ch.amp(amp_value)   
        


    # setup trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0)
    trigger.ext0_mode(spcm.SPC_TM_POS)
    trigger.ext0_level0(0.5 * units.V)
    trigger.ext0_coupling(spcm.COUPLING_DC)
    trigger.termination(1)

    # communicate the setup to card
    card.write_setup()

    # set DDS mode
    dds = spcm.DDS(card, channels=channels)
    dds.reset()
    core0 = dds[0] # use core 0 for MVP

    core0.freq(f_start_hz * units.Hz) # lock to start value at t < 0
    core0.amp(a_start_pct * units.percent) # turn on at a_start_pct * units.percent, for now turned off since we only want output once we trigger
    core0.phase(0)
    core0.freq_slope(0)
    core0.amp_slope(0)
    
    
    # Read back the exact values
    freq = dds[0].get_freq(return_unit=units.MHz)
    amp = dds[0].get_amp(return_unit=units.dBm)
    phase = dds[0].get_phase(return_unit=units.rad)
    print(f"Generated core0 signal frequency: {freq} and amplitude: {amp} and phase: {phase}")
     
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD) # use internal trigg
    core0.freq(f_start_hz * units.Hz)
    core0.amp(a_start_pct * units.percent)
    core0.freq_slope(float(freq_slopes_hz_per_s[0]) * units.Hz / units.s)
    core0.amp_slope(float(amp_slopes_pct_per_s[0]) * units.percent / units.s)
    dds.exec_at_trg() # execute initial state on trigger

    dds.trg_timer(dt_s * units.s) # set trigger source to internal timer with the period corresponding to the segment duraction
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER) # the trigger will now always fire after the configured dt
    for i in range(1, N - 1): # we can now update the next segments parameters which will be called in sequence every dt
        core0.freq(float(freqs_hz[i]) * units.Hz)
        core0.amp(float(amps_pct[i]) * units.percent)
        core0.freq_slope(float(freq_slopes_hz_per_s[i]) * units.Hz / units.s)
        core0.amp_slope(float(amp_slopes_pct_per_s[i]) * units.percent / units.s)
        dds.exec_at_trg()

    # finally set the last point to be held and put everything to 0
    core0.freq(f_stop_hz * units.Hz)  # will hold that forever
    core0.amp(a_stop_pct * units.percent) # a_stop_pct * units.percent will hold that forever
    core0.freq_slope(0)
    core0.amp_slope(0)
    dds.exec_at_trg()

    dds.write_to_card()  # push these commands to the card

    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Initial state loaded. Send EXT0 to start STA sweep. Press Enter to exit.")