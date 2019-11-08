### Digitised data from [Firek1995]
import numpy as np

# Steady State Inactivation

def Inact_Firek():
    """
    Steady-State inactivation curve [Firek1995]
    cf Fig 3c
    """
    x = [-50, -35, -25, -15, -5, 5, 15]
    y = np.asarray([0.9478260869565218,
                    0.9356521739130435,
                    0.8852173913043478,
                    0.6869565217391305,
                    0.6278260869565218,
                    0.5478260869565218,
                    0.5252173913043479])
    ylower = np.asarray([0.8973913043478261,
                        0.9130434782608696,
                        0.8400000000000001,
                        0.6452173913043479,
                        0.5669565217391304,
                        0.5130434782608696,
                        0.5026086956521739])
    sem = np.abs(y-ylower)
    N = 6
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()
