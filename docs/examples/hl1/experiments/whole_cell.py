import numpy as np
import pandas as pd
import myokit
from ionchannelABC.experiment import Experiment


ap_desc = """Action potential and calcium transient characteristics
from paced whole cell simulation. 80pA/pF for 0.5ms at 1Hz for 100s.
"""

# AP measurements
mdp, mdp_sem, mdp_n = -67, 2, 25 # maximum diastolic potential
mdp_sd = np.sqrt(mdp_n)*mdp_sem
dvdt_max, dvdt_max_sem, dvdt_max_n = 107, 7, 11 # maximum upstroke
dvdt_max_sd = np.sqrt(dvdt_max_n)*dvdt_max_sem
amp, amp_sem, amp_n = 105, 2, 11 # maximum amplitude of AP
amp_sd = np.sqrt(amp_n)*amp_sem
apd90, apd90_sem, apd90_n = 42, 9, 7 # 90% repolarisation of AP
apd90_sd = np.sqrt(apd90_n)*apd90_sem
# CaT measurements
t2p, t2p_sem, t2p_n = 59, 2, 6 # CaT time to peak
t2p_sd = np.sqrt(t2p_n)*t2p_sem
CaTR50, CaTR50_sem, CaTR50_n = 157, 6, 6 # CaT time to 50% repolarisation
CaTR50_sd = np.sqrt(CaTR50_n)*CaTR50_sem
CaTR90, CaTR90_sem, CaTR90_n = 397, 14, 6 # CaT time to 90% repolarisation
CaTR90_sd = np.sqrt(CaTR90_n)*CaTR90_sem
ap_dataset = [np.asarray([[0], [mdp], [mdp_sd**2]]),
              np.asarray([[0], [dvdt_max], [dvdt_max_sd**2]]),
              np.asarray([[0], [amp], [amp_sd**2]]),
              np.asarray([[0], [apd90], [apd90_sd**2]]),
              np.asarray([[0], [t2p], [t2p_sd**2]]),
              np.asarray([[0], [CaTR50], [CaTR50_sd**2]]),
              np.asarray([[0], [CaTR90], [CaTR90_sd**2]])]

ap_protocol = myokit.pacing.blocktrain(
    period=1000, duration=2, limit=101, offset=2
)

ap_conditions = {'extra.Ca_o': 1.8e3,
                 'extra.K_o' : 4.0e3,
                 'extra.Na_o': 130e3,
                 'phys.T'    : 295}

def ap_sum_stats(data):
    output = []
    d = data.trim_left(1000*100, adjust=True)
    t = d['engine.time']
    v = d['membrane.V']
    CaT = d['calcium.Ca_i']
    # minimum diastolic potential
    mdp = np.min(v)
    # maximum upstroke gradient
    dvdt_max_idx = np.argmax(np.gradient(v, t))
    dvdt_max = np.max(np.gradient(v, t))
    # amplitude
    peak_idx = np.argmax(v)
    amp = np.max(v)-mdp
    # action potential duration (90% repolarisation)
    try:
        decay = d.trim_left(t[peak_idx])['membrane.V']
        apd90_idx = np.argwhere(decay < np.max(v)-0.9*amp)[0][0]
        apd90 = t[peak_idx+apd90_idx] - t[dvdt_max_idx]
    except:
        apd90 = float('inf')
    # CaT time to peak
    peak_cat_idx = np.argmax(CaT)
    cat_t2p = t[peak_cat_idx] - 2 # offset 2ms
    if cat_t2p < 0:
        cat_t2p = float('inf')
    # CaT time to repolarisation 50% and 90%
    peak_cat = np.max(CaT)
    try:
        decay = d.trim_left(t[peak_cat_idx])['calcium.Ca_i']
        cat_r50_idx = np.argwhere(decay < peak_cat - 0.5*CaT[0])[0][0]
        cat_r50 = t[peak_cat_idx+cat_r50_idx] - 2
        cat_r90_idx = np.argwhere(decay < peak_cat - 0.9*CaT[0])[0][0]
        cat_r90 = t[peak_cat_idx+cat_r90_idx] - 2
    except:
        cat_r50 = float('inf')
        cat_r90 = float('inf')
    return [mdp, dvdt_max, amp, apd90, cat_t2p, cat_r50, cat_r90]

ap = Experiment(
    dataset=ap_dataset,
    protocol=ap_protocol,
    conditions=ap_conditions,
    sum_stats=ap_sum_stats,
    description=ap_desc,
    Q10=None,
    Q10_factor=0.
)
