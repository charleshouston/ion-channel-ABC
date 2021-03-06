### Digitised data from [Mewes1994]
import numpy as np
# IV Curves

# Steady State Activation

def Act_Mewes():
    """
    Steady-State activation curve [Mewes1994]
    cf Fig 5c
    """

    x = np.arange(-35, 20, 5)
    y = np.asarray([0.016543331908527525,
                    0.05896692669373804,
                    0.11645116477879891,
                    0.2582616477879889,
                    0.4091218743321222,
                    0.5840724513784997,
                    0.7650472857448173,
                    0.8797486108142767,
                    0.9281897841419107,
                    0.9404854135499039,
                    0.9226664351357128])
    ylower = np.asarray([0.0014893673861935408,
                         0.03487657619149398,
                         0.08934868561658504,
                         0.2130863966659542,
                         0.3518981085702073,
                         0.5298541354990384,
                         0.7198720346227827,
                         0.8436097456721523,
                         0.9101303697371234,
                         0.9194071917076299,
                         0.8955639559734987])
    sem = np.abs(y-ylower)
    N = 13
    sd = np.sqrt(N)*sem
    return x.tolist(), y.tolist(), sd.tolist()
