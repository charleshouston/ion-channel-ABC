### Digitised data from [Firek1995]
import numpy as np

# Steady State Inactivation

def Inact_Firek():
    """
    Steady-State inactivation curve [Firek1995]
    cf Fig 3c
    """
    x = [-70, -60, -50, -35, -25, -15, -5, 5, 15]
    y = np.asarray([0.9608748465711434,
                    0.9081016796087487,
                    0.7481159254383455,
                    0.5451859907529303,
                    0.37155858279434495,
                    0.293448190334459,
                    0.1237161264062987,
                    0.04364693237682782,
                    0.0025725169964196937])

    ylower = np.asarray([0.9004413663276923,
                         0.828172399977105,
                         0.5064074434459646,
                         0.3502502559797507,
                         0.2097666609428961,
                         0.13164990873765436,
                         0.039888322871552084,
                         -0.001189272381533879,
                         -0.0403112459377124])
    sem = np.abs(y-ylower)
    N = 6
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()
