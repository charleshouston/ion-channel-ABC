### Digitised data from [Courtemanche1998]
import numpy as np

# Activation kinetics

def Act_Kin_Courtemanche():
    """
    Activation kinetics [Courtemanche1998]
    cf Fig 5d
    """
    x = np.arange(-30, 60, 10).tolist()
    y = [18.4251968503937,
         16.01049868766404,
         6.8241469816272975,
         4.199475065616795,
         2.939632545931758,
         2.047244094488189,
         1.5748031496063,
         1.2598425196850371,
         0.9973753280839901]
    return x, y, None


# Deactivation kinetics

def Deact_Kin_Courtemanche():
    """
    Deactivation kinetics [Courtemanche1998]
    cf Fig 5d
    """
    x = [-80, -70, -50, -40]
    y = [4.829396325459314,
         6.141732283464567,
         8.766404199475065,
        12.388451443569553]
    return x, y, None


# Inactivation kinetics

def Inact_Kin_Courtemanche():
    """
    Inactivation kinetics [Courtemanche1998]
    cd Fig 5d
    """
    x = np.arange(-30, 60, 10).tolist()
    y = [159.89485221084357,
        79.31808062472174,
        49.26704798438192,
        37.37027776826386,
        35.73826078021716,
        34.895708463198275,
        35.63208548823508,
        36.371887522690656,
        37.10483953830874]
    return x, y, None


# Recovery kinetics

def Rec_Courtemanche():
    """
    Recovery kinetics [Courtemanche1998]
    cd Fig 5d
    """
    x = np.arange(-80, -30, 10).tolist()
    y = [27.559055118110166,
        40.944881889763735,
        70.86614173228344,
        142.51968503937005,
        257.48031496062987]
    return x, y, None
