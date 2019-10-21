### Digitised data from [Nygren1998]
import numpy as np

# Inactivation kinetics

def Inact_Kin_Nygren():
    """
    Inactivation kinetics from [Nygren1998]
    cf Fig 4c
    """
    x = np.arange(-10, 50, 10)
    y = np.asarray([285.36729423105726,
                    263.7475959994694,
                    462.3822288175494,
                    283.349325934525,
                    305.12410930146774,
                    301.4994103793058])
    return x.tolist(), y.tolist(), None


# Recovery (reactivation) kinetics)

def Rec_Nygren():
    """
    Recovery data from [Nygren1998]
    cf Fig 4d
    """
    x = [-90]
    y = [352.1414827445892]
    return x, y, None
