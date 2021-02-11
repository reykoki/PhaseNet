
import os
import numpy as np
from collections import namedtuple
from datetime import datetime, timedelta
from detect_peaks import detect_peaks
import json


def extract_picks(preds, fnames=None, t0=None, config=None):

    if preds.shape[-1] == 4:
        record = namedtuple("phase", ["fname", "t0", "p_idx", "p_prob", "s_idx", "s_prob", "ps_idx", "ps_prob"])
    else:
        record = namedtuple("phase", ["fname", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])

    picks = []
    for i, pred in enumerate(preds):

        if config is None:
            mph_p, mph_s, mpd = 0.3, 0.3, 30

        else:
            mph_p, mph_s, mpd = config.min_p_prob, config.min_s_prob, 0.5/config.dt

        if (fnames is None):
            fname = f"{i:04d}"
        else:
            if isinstance(fnames[i], str):
                fname = fnames[i]
            else:
                fname = fnames[i].decode()
            
        if (t0 is None):
            start_time = "0"
        else:
            if isinstance(t0[i], str):
                start_time = t0[i]
            else:
                start_time = t0[i].decode()

        p_idx, p_prob, s_idx, s_prob = [], [], [], []
        for j in range(pred.shape[1]):
            p_idx_, p_prob_ = detect_peaks(pred[:,j,1], mph=mph_p, mpd=mpd, show=False)
            s_idx_, s_prob_ = detect_peaks(pred[:,j,2], mph=mph_s, mpd=mpd, show=False)
            p_idx.append(list(p_idx_))
            p_prob.append(list(p_prob_))
            s_idx.append(list(s_idx_))
            s_prob.append(list(s_prob_))

        if pred.shape[-1] == 4:
            ps_idx, ps_prob = detect_peaks(pred[:,0,3], mph=0.3, mpd=mpd, show=False)
            picks.append(record(fname, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob), list(ps_idx), list(ps_prob)))
        else:
            picks.append(record(fname, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob)))

    return picks


def extract_amplitude(data, picks, window_p=8, window_s=5, config=None):
    record = namedtuple("amplitude", ["p_amp", "s_amp"])
    dt = 0.01 if config is None else config.dt
    window_p = int(window_p/dt)
    window_s = int(window_s/dt)
    amps = []
    for i, (da, pi) in enumerate(zip(data, picks)):
        p_amp, s_amp = [], []
        for j in range(da.shape[1]):
            amp = np.max(np.abs(da[:,j,:]), axis=-1)
            p_amp.append([np.max(amp[idx:idx+window_p]) for idx in pi.p_idx[j]])
            s_amp.append([np.max(amp[idx:idx+window_s]) for idx in pi.s_idx[j]])
        amps.append(record(p_amp, s_amp))
    return amps

def save_picks(picks, output_dir, amps=None):

    int2s = lambda x: ",".join(["["+",".join(map(str, i))+"]" for i in x])
    flt2s = lambda x: ",".join(["["+",".join(map("{:0.3f}".format, i))+"]" for i in x])
    sci2s = lambda x: ",".join(["["+",".join(map("{:0.3e}".format, i))+"]" for i in x])
    if amps is None:
        if hasattr(picks[0], "ps_idx"):
            with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tps_idx\tps_prob\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{int2s(pick.ps_idx)}\t{flt2s(pick.ps_prob)}\n")
                fp.close()
        else:
            with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\n")
                fp.close()
    else:
        with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
            fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tp_amp\ts_amp\n")
            for pick, amp in zip(picks, amps):
                fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{sci2s(amp.p_amp)}\t{sci2s(amp.s_amp)}\n")
            fp.close()

    return 0


def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
def save_picks_json(picks, output_dir, dt=0.01, amps=None):
    
    picks_ = []
    for pick, amplitude in zip(picks, amps):
        for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname, 
                               "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                               "prob": prob, 
                               "amp": amp,
                               "type": "p"})
        for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname, 
                               "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                               "prob": prob, 
                               "amp": amp,
                               "type": "s"})

    with open(os.path.join(output_dir, "picks.json"), "w") as fp:
        json.dump(picks_, fp)

    return 0

