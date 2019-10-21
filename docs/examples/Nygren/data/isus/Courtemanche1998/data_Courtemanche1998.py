### Digitised data from [Courtemanche1998]
import numpy as np


# Deactivation kinetics

def Deact_Kin_Courtemanche():
    """
    Deactivation kinetics [Courtemanche1998]
    cf Fig 5b
    """
    x = [-80, -50, -40, -30]
    y = [4.939481200083289,
         10.226164706991584,
        12.980369202000404,
        14.727930575252707]
    return x, y, None


# Inactivation kinetics

def Inact_Kin_Courtemanche():
    """
    Inactivation kinetics [Courtemanche1998]
    cd Fig 5b
    """
    x = np.arange(0, 60, 10).tolist()
    y = np.asarray([0.9680028902642075,
                    0.7045411705570039,
                    0.5065423643617848,
                    0.35229022022556755,
                    0.24162765857183288,
                    0.13100436681222583])
    y = y*1000.
    return x, y.tolist(), None
