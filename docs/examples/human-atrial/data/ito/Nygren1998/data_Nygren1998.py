### Digitised data from [Nygren1998]
import numpy as np

# Inactivation kinetics

def Inact_Kin_Nygren():
    """
    Inactivation kinetics from [Nygren1998]
    cf Fig 4c
    """
    x = np.arange(0, 50, 10)
    y = np.asarray([12.535043950889689,
                    11.239082354992732,
                    15.040688796863812,
                    14.565827181187226,
                    16.203311113674975])
    return x.tolist(), y.tolist(), None


def Rec_Nygren():
    """
    Recovery data from [Nygren1998]
    cf Fig 4c
    """
    x = [-100, -80, -60]
    y = [15, 30, 387]
    return x, y, None
