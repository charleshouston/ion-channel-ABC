### Digitised data from Li and nattel paper from 1997 I_CaL model evaluation
## author Benjamin Marchand
import numpy as np
# IV Curves

def IV_Li_80():
     """
     IV curve for CaL - holding -80mV
     already converted to pA/pF
     Data from Li et al '97 Fig. 1C
     """
     x = [-70, -60, -50, -40, -30, -20, -10, 0,
         10, 20, 30, 40, 50, 60]

     y = np.asarray([0.0085, 0.0056, 0.0025, -0.1982, -0.7223,
         -2.4683, -6.5145, -10.0575, -10.7433, -9.3806,
         -7.1734, -4.8763, -2.7589, -0.3719])

     yupper = np.asarray([0.0085, 0.0056, 0.0025, -0.1802, -0.6863,
            -2.0910, -5.8676, -9.4466, -9.7730, -8.5181,
            -6.7241, -4.8763, -2.7589, -0.3719])

     sem = yupper - y
     N = 15
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def IV_Li_60():
     """
     IV curve for CaL - holding -60mV
     already converted to pA/pF
     Data from Li et al '97 Fig. 1C
     """
     x = [-50, -40, -30, -20, -10, 0,
         10, 20, 30, 40, 50, 60]

     y = np.asarray([0.021219056,0.018300518,-0.24539581,-1.1956093,-4.8025737,
                    -8.848952,-10.2042,-9.204793,-6.8803887,-3.9587488,
                    -2.1630037,-0.31918526])

     ylower = np.asarray([0.021219056,0.018300518,-0.24539581,-1.4977056,-5.351791,
                         -9.4118805,-10.774013,-10.001114,-7.4570847,-3.9587488,
                         -2.1630037,-0.31918526])

     sem =  y - ylower
     N = 9
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def IV_Li_40():
     """
     IV curve for CaL - holding -40mV
     already converted to pA/pF
     Data from Li et al '97 Fig. 1C
     """
     x = [-30, -20, -10, 0,10, 20, 30, 40, 50, 60]

     y = np.asarray([0.015400335,-0.47482225,-2.5783906,-5.258581,-6.3461127,
                    -6.0949717,-4.7316656,-3.1692936,-1.7236261,-0.06519903 ])

     ylower = np.asarray([0.015400335,-0.47482225,-2.5783906,-5.8215284,-6.9845753,
                         -6.8569856,-5.5142746,-3.6154995,-1.7236261,-0.06519903 ])

     sem =  y - ylower
     N = 10
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def IV_Li_all(Vhold : int):
    """
    Returns the IV curve (pA/pF) from Figure 1C
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    already converted to pA/pF
    """
    try:
        assert(Vhold in [-80,-60,-40])
    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == -80:
        x,y,sd = IV_Li_80()
    elif Vhold == -60:
        x,y,sd = IV_Li_60()
    elif Vhold == -40:
        x,y,sd = IV_Li_40()

    return x,y,sd


# Steady State Activation

def Act_Li():
     """
     Steady-State activation curve
     Data from Li et al '97 Fig. 2B
     """
     x = [-80, -70, -60, -50, -40, -30, -25, -20, -15,
         -10, -5, 0, 5, 10, 15, 20]
     y = np.asarray([0.0071,0.0071, 0.0061, 0.0119, 0.0332,
                     0.0836,0.1344, 0.2093, 0.3371, 0.5095,
                     0.6700, 0.8184,0.9275, 0.9749, 0.9864,
                     0.9961])
     yupper = np.asarray([0.0071,0.0071, 0.0061, 0.0119, 0.0332,
                          0.0836,0.1344, 0.2281, 0.3679, 0.5472,
                          0.7179,0.8646, 0.9651, 0.9749, 0.9864,
                          0.9961])
     sem = yupper - y
     N = 10
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

# Steady State Inactivation
def Inact_Li_1000():
     """
     Steady-State inactivation curve - 1000ms pre-pulse
     Data from Li et al '97 Fig. 2B
     """
     x = [-80, -70, -60, -50, -40, -30, -20, -10, 0,
          10, 20, 30, 40, 50]
     y = np.asarray([0.9942, 0.9829, 0.9887, 0.9809, 0.8361, 0.5698,
                     0.2795, 0.0627, 0.0292, 0.0094, 0.0152, 0.0262,
                     0.0577, 0.0875])
     yupper = np.asarray([0.9976, 0.9778, 0.9904, 0.9980, 0.8892, 0.6040,
                         0.3051, 0.0833, 0.0412, 0.0214, 0.0272, 0.0450,
                         0.0783, 0.1149])
     sem = yupper - y
     N = 8
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Inact_Li_300():
     """
     Steady-State inactivation curve - 300ms pre-pulse
     Data from Li et al '97 Fig. 2B
     """
     x = [-80, -70, -60, -50, -40, -30, -20, -10, 0,
          10, 20, 30, 40, 50]
     y = np.asarray([0.9828174,0.9644436,0.9913071,0.9699411,0.83782166,
                    0.6163692,0.36924195,0.16980544,0.08827661,0.05918199,
                    0.052713107,0.065289244,0.11422548,0.156603])
     ylower = np.asarray([0.9828174,0.941201,0.9597331,0.92109466,0.7883726,
                         0.58241284,0.30076206,0.12509272,0.058503658,0.034180954,
                         0.028853536,0.044436105,0.09809451,0.13933766])
     sem = y - ylower
     N = 5
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Inact_Li_150():
     """
     Steady-State inactivation curve - 150ms pre-pulse
     Data from Li et al '97 Fig. 2B
     """
     x = [-80, -70, -60, -50, -40, -30, -20, -10, 0,
          10, 20, 30, 40, 50]
     y = np.asarray([0.98222893,0.9626639,0.9883575,0.98900026,0.8694027,
                    0.618156,0.38531607,0.16980544,0.088893525,0.059784696,
                    0.053911414,0.0932825,0.15531485,0.2185526 ])
     ylower = np.asarray([0.98222893,0.94122225,0.959173,0.9443515,0.806269,
                         0.5383242,0.3483959,0.12512825,0.05909926,0.034180954,
                         0.03065455,0.045031704,0.13388032,0.18220674])
     sem = y - ylower
     N = 5
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def inact_Li_all(tstep : int):
    """
    Steady-State inactivation curve  from Figure 1C
    in Li and Nattel 1997 with tstep = 1000, 300, 150 mS

    """
    try:
        assert(tstep in [1000,300,150])
    except:
        raise Exception("Wrong tstep, ( = {} mS)".format(tstep))

    if tstep == 1000:
        x,y,sd = Inact_Li_1000()
    elif tstep == 300:
        x,y,sd = Inact_Li_300()
    elif tstep == 150:
        x,y,sd = Inact_Li_150()

    return x,y,sd

# import matplotlib.pyplot as plt
# for tstep in [1000,300,150]:
#     x,y,_ = inact_Li_all(tstep)
#     plt.plot(x,y)
# plt.legend(('1000','300','150'))
# plt.show()

# Time Constant Experiments
def Tau1_Li_80():
     """
     Time Constant data for tau 1 - -80mV HP
     Data from Li et al '97 Fig 3B
     """
     x = [-10, 0, 10, 20, 30]
     y = np.asarray([9.3351, 9.5781, 11.2823, 10.5040, 9.6271])
     yupper = np.asarray([10.3084, 10.6500, 12.4983, 11.5740, 11.6703])
     sem = yupper - y
     N = 12
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Tau1_Li_60():
     """
     Time Constant data for tau 1 - -60mV HP
     Data from Li et al '97 Fig 3B
     """
     x = [-10, 0, 10, 20, 30]
     y = np.asarray([11.250413,10.649782,12.664147,14.214826,16.567045])
     yupper = np.asarray([11.250413,10.649782,12.664147,16.32484,20.786776])
     sem = yupper - y
     N = 10
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Tau1_Li_40():
     """
     Time Constant data for tau 1 - -40mV HP
     Data from Li et al '97 Fig 3B
     """
     x = [-10, 0, 10, 20, 30]
     y = np.asarray([11.334802,10.607138,12.410981,14.214675,14.54217])
     yupper = np.asarray([11.334802,10.607138,12.410981,14.214675,14.54217]) # not sure there is any uncertainty :
     # hard to distinguish on the figure
     sem = yupper - y
     N = 8
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Tau1_Li_all(Vhold : int):
    """
    Returns Time Constant data for tau 1 from Figure 3B
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    """
    try:
        assert(Vhold in [-80,-60,-40])
    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == -80:
        x,y,sd = Tau1_Li_80()
    elif Vhold == -60:
        x,y,sd = Tau1_Li_60()
    elif Vhold == -40:
        x,y,sd = Tau1_Li_40()

    return x,y,sd

# import matplotlib.pyplot as plt
# for Vhold in [80,60,40]:
#     x,y,s_ = Tau1_Li_all(Vhold)
#     plt.plot(x,y)
# plt.legend(('80','60','40'))
# plt.show()

def Tau2_Li_80():
    """
    Time Constant data for tau 2 - -80mV HP
    Data from Li et al '97 Fig 3B
    """
    x = [-10, 0, 10, 20, 30]
    y = np.asarray([87.5782, 62.5797, 50.7630, 54.3979, 76.2176])
    yupper = np.asarray([94.4024, 67.1252, 58.0298, 54.8554, 92.1267])

    sem = yupper - y
    N = 12
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()

def Tau2_Li_60():
     """
     Time Constant data for tau 2 - -60mV HP
     Data from Li et al '97 Fig 3B
     """
     x = [-10, 0, 10, 20, 30]
     y = np.asarray([74.84698,62.14582,52.21172,61.257946,72.47672])
     yupper = np.asarray([74.84698,62.14582,52.21172,61.257946,72.47672]) # unable to distinguish the error bars

     sem = yupper - y
     N = 10
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Tau2_Li_40():
     """
     Time Constant data for tau 2 - -40mV HP
     Data from Li et al '97 Fig 3B
     """
     x = [-10, 0, 10, 20, 30]
     y = np.asarray([74.84839,61.74902,54.189396,60.86114,85.72313])
     yupper = np.asarray([74.84839,61.74902,54.189396,60.86114,92.24713])

     sem = yupper - y
     N = 8
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()


def Tau2_Li_all(Vhold : int):
    """
    Returns Time Constant data for tau 2 from Figure 3B
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    """
    try:
        assert(Vhold in [-80,-60,-40])
    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == -80:
        x,y,sd = Tau2_Li_80()
    elif Vhold == -60:
        x,y,sd = Tau2_Li_60()
    elif Vhold == -40:
        x,y,sd = Tau2_Li_40()

    return x,y,sd

# import matplotlib.pyplot as plt
# for Vhold in [80,60,40]:
#     x,y,s_ = Tau2_Li_all(Vhold)
#     plt.plot(x,y)
# plt.legend(('80','60','40'))
# plt.show()

def recov_Li_80():
     """
     recovery data for I2/I1 -80mV HP
     Data from Li et al '97 Fig 4B
     """
     x = [6.558458,16.384476,70.22334,114.97140,159.35185,
          196.94046,235.14172,268.28995,318.14355,440.4553,
          554.7499,674.71014,830.8994,1009.5474,2013.8258,
          3004.188]
     y = np.asarray([-6.7262736E-5,0.33562955,0.6879255,0.8347158,0.91641045,
                    0.95573455,0.9785908,0.98731524,0.9968731,1.0082126,
                    1.0101173,1.0096859,1.0101452,1.0067493,1.004213,
                    1.0008514])
     yupper = np.asarray([-6.7262736E-5,0.3819375,0.7326487,0.86688375,0.9454335,
                         0.9792601,1.0013157,1.0107998,1.0219014,1.0293446,
                         1.0101173,1.0316103,1.0297416,1.0067493,1.004213,
                         1.0008514])

     sem = yupper - y
     N = 11
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def recov_Li_60():
     """
     recovery data for I2/I1 -60mV HP
     Data from Li et al '97 Fig 4B
     """
     x = [4.5979223,26.117134,45.343216,107.52603,136.96907,
          182.42899,231.14479,273.9323,311.6376,431.382,
          591.28235,835.29315,1116.7,1506.2981,2206.4775,
          3514.7092]
     y = np.asarray([1.0007383E-4,0.14900172,0.3819867,0.5100057,0.6183217,
                    0.67100155,0.71114266,0.7606776,0.7968649,0.8772127,
                    0.9278763,0.9670232,0.9756935,1.0003675,0.99928963,
                    0.99921584])
     yupper = np.asarray([1.0007383E-4,0.14900172,0.419603,0.56021,0.68498564,
                         0.76048887,0.8053105,0.7606776,0.83686656,0.91564596,
                         0.95610696,0.99522924,1.0086211,1.0003675,0.99928963,
                         0.99921584])

     sem = yupper - y
     N = 7
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def recov_Li_40():
     """
     recovery data for I2/I1 -40mV HP
     Data from Li et al '97 Fig 4B
     """
     x = [60.449844,185.4865,308.53345,431.8546,670.7132,
          786.48395,906.0941,1158.5306,1386.1862,1755.9979,
          2106.3735,2579.4404,3192.3704,3791.325]
     y = np.asarray([0.20086457,0.43886638,0.580397,0.63958,0.7422377,
                    0.7794389,0.7884177,0.82602084,0.85492414,0.8865983,
                    0.9158625,0.94627184,0.96611273,0.9866967])
     yupper = np.asarray([0.20086457,0.47179395,0.62038225,0.67877287,0.7767501,
                         0.8076696,0.815856,0.85658765,0.87609714,0.9085637,
                         0.9378197,0.96824545,0.96611273,0.9866967])

     sem = yupper - y
     N = 8
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()


def Recov_Li_all(Vhold : int):
    """
    Returns recovery data for I2/I1 from Figure 4B
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    """
    try:
        assert(Vhold in [-80,-60,-40])
    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == -80:
        x,y,sd = recov_Li_80()
    elif Vhold == -60:
        x,y,sd = recov_Li_60()
    elif Vhold == -40:
        x,y,sd = recov_Li_40()

    return x,y,sd


def TauF_Recov_Li():
    """
    Data for fast recovery time constant for i_CaL
    from text on H230 in Li1997.
    Errors reported as mean +/- SEM for 7,8 cells
    """
    x = [-60, -40]
    y = [32, 169]
    sem = np.array([8, 37])
    N = [7, 8]
    sd = np.sqrt(N)*sem

    return x, y, sd.tolist()

def TauS_Recov_Li():
    """
    Data for slow recovery time constant for i_CaL
    from text on H230 in Li1997.
    Errors reported as mean +/- SEM for 11,7,8 cells
    """
    x = [-80, -60, -40]
    y = [55, 242, 1384]
    sem = np.array([10, 51, 219])
    N = [11, 7, 8]
    sd = np.sqrt(N)*sem

    return x, y, sd.tolist()


# import matplotlib.pyplot as plt
# for Vhold in [80,60,40]:
#     x,y,s_ = Recov_Li_all(Vhold)
#     plt.plot(x,y)
# plt.legend(('80','60','40'))
# plt.show()

def relat_curr_Li_2_0():
     """
     relative current as function of the pulse number
     data for -40mV HP and 2.0Hz
     Data from Li et al '97 Fig 5C
     """
     x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
     y = np.asarray([1.0012542,0.65933526,0.5895303,0.5552531,0.5068999,
                    0.50781155,0.4725323,0.4462981,0.4482151,0.4199685,
                    0.4098194,0.40268308,0.41164017,0.40149108,0.40843078])
     ylower = np.asarray([1.0012542,0.58158517,0.5131229,0.48387176,0.4335103,
                         0.43844426,0.40919393,0.37290603,0.37582675,0.34557107,
                         0.33039427,0.3202435,0.3251758,0.31804115,0.3340317])

     sem = y - ylower
     N = 13
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_Li_1_0():
     """
     relative current as function of the pulse number
     data for -40mV HP and 1.0Hz
     Data from Li et al '97 Fig 5C
     """
     x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
     y = np.asarray([0.9969004,0.7293747,0.6538748,0.6165824,0.5692329,
                    0.5460156,0.52279586,0.5096331,0.49747497,0.4823023,
                    0.48421517,0.4609946,0.4699492,0.45376623,0.46372452])
     yupper = np.asarray([0.9969004,0.77662545,0.7172132,0.67388767,0.64061505,
                         0.6204139,0.60222185,0.5830277,0.5668447,0.5587072,
                         0.55559653,0.5414251,0.541328,0.5301735,0.5351042])

     sem = yupper - y
     N = 13
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_Li_0_5():
     """
     relative current as function of the pulse number
     data for -40mV HP and 0.5Hz
     Data from Li et al '97 Fig 5C
     """
     x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
     y = np.asarray([0.99891114,0.87314105,0.83283085,0.78749293,0.77935785,
                    0.7601613,0.74498695,0.72981185,0.73273176,0.74068016,
                    0.7375678,0.7324447,0.73536545,0.7322539,0.7231102])
     ylower = np.asarray([0.99891114,0.84700227,0.7986474,0.75130206,0.7270778,
                         0.72698236,0.70577586,0.69060236,0.69251776,0.70448595,
                         0.69835585,0.6982621,0.69916797,0.6920366,0.6889276])

     sem = y - ylower
     N = 13
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_Li_0_1():
     """
     relative current as function of the pulse number
     data for -40mV HP and 0.1Hz
     Data from Li et al '97 Fig 5C
     """
     x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
     y = np.asarray([0.9989128,0.9777005,0.9796158,0.9795196,0.9754052,
                    0.9793321,0.97521687,0.9791438,0.986086,0.96085554,
                    0.9708138,0.97474074,0.9666024,0.9755545,0.98651737])
     ylower = np.asarray([0.9989128,0.9777005,0.9796158,0.9553907,0.953287,
                         0.9562069,0.94807106,0.95300335,0.9579373,0.9276791,
                         0.9376358,0.94860196,0.9344289,0.94639874,0.95635456])

     sem = y - ylower
     N = 13
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def Relat_curr_Li_all(frequency : float):
    """
    Returns relative current as function of the pulse number from Figure 5C
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    """
    try:
        assert(frequency in [2.0,1.0,0.5,0.1])
    except:
        raise Exception("Wrong frequency, ( = {} Hz)".format(frequency))

    if frequency == 2.0:
        x,y,sd = relat_curr_Li_2_0()
    elif frequency == 1.0:
        x,y,sd = relat_curr_Li_1_0()
    elif frequency == 0.5:
        x,y,sd = relat_curr_Li_0_5()
    elif frequency == 0.1:
        x,y,sd = relat_curr_Li_0_1()

    return x,y,sd


def Use_Inact_Li_80():
    """Pulse constants for use-dependent inactivation (Fig 5D).

    Given as mean +/- SE.

    x: Hz
    y: Pulse constant (1/k) see text.
    """
    x = [0.5, 1.0, 2.0]
    y = np.asarray([8.412698412698413,
                    3.682539682539683,
                    4.253968253968255])
    sem = np.asarray([9.238095238095239,
                      4.19047619047619,
                      4.825396825396825])
    sem = np.abs(y-sem)
    N = 8
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def Use_Inact_Li_60():
    """Pulse constants for use-dependent inactivation (Fig 5D).

    Given as mean +/- SE.

    x: Hz
    y: Pulse constant (1/k) see text.
    """
    x = [0.5, 1.0, 2.0]
    y = np.asarray([2.6984126984126973,
                    3.238095238095239,
                    3.1746031746031744])
    sem = np.asarray([3.3968253968253963,
                      3.841269841269842,
                      3.904761904761904])
    sem = np.abs(y-sem)
    N = 10
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def Use_Inact_Li_40():
    """Pulse constants for use-dependent inactivation (Fig 5D).

    Given as mean +/- SE.

    x: Hz
    y: Pulse constant (1/k) see text.
    """
    x = [0.5, 1.0, 2.0]
    y = np.asarray([1.9682539682539677,
                    2.000000000000001,
                    1.6825396825396814])
    sem = np.asarray([2.4126984126984135,
                      2.53968253968254,
                      2.1269841269841274])
    sem = np.abs(y-sem)
    N = 10
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

# import matplotlib.pyplot as plt
# for frequency in [2.0,1.0,0.5,0.1]:
#     x,y,s_ = Relat_curr_Li_all(frequency)
#     plt.plot(x,y)
# plt.legend(('2.0','1.0','0.5','0.1'))
# plt.show()

def relat_curr_ss_Li_80():
     """
     relative current as function of the Frequency (Hz)
     data for -80mV HP
     Data from Li et al '97 Fig 6B
     """
     x = [0.2,0.5,1,2]
     y = np.asarray([1,0.97614056,0.9490366,0.87891024])
     ylower = np.asarray([1,0.939259,0.8953926,0.8210704])

     sem = y - ylower
     N = 8
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_ss_Li_60():
     """
     relative current as function of the Frequency (Hz)
     data for -60mV HP
     Data from Li et al '97 Fig 6B
     """
     x = [0.2,0.5,1,2]
     y = np.asarray([1,0.87806815,0.6858296,0.63749784])
     ylower = np.asarray([1,0.82945347,0.6858296,0.63749784])

     sem = y - ylower
     N = 10
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_ss_Li_40():
     """
     relative current as function of the Frequency (Hz)
     data for -40mV HP
     Data from Li et al '97 Fig 6B
     """
     x = [0.2,0.5,1,2]
     y = np.asarray([1,0.7364104,0.4720861,0.406147])
     ylower = np.asarray([1,0.70120144,0.39748573,0.3323858])

     sem = y - ylower
     N = 13
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def relat_curr_ss_Li_all(Vhold : int):
    """
    Returns relative current as function of the Frequency (Hz) from Figure 6B
    in Li and Nattel 1997 with Vhold = -80, -60, -40 mV
    """
    try:
        assert(Vhold in [80,60,40])
    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == 80:
        x,y,sd = relat_curr_ss_Li_80()
    elif Vhold == 60:
        x,y,sd = relat_curr_ss_Li_60()
    elif Vhold == 40:
        x,y,sd = relat_curr_ss_Li_40()

    return x,y,sd

# import matplotlib.pyplot as plt
# for Vhold in [80,60,40]:
#     x,y,s_ = relat_curr_ss_Li_all(Vhold)
#     plt.plot(x,y)
# plt.legend(('80','60','40'))
# plt.show()

def IV_Li_36deg():
     """
     IV curve for CaL - holding -80mV at 36°C
     Data from Li et al '97 Fig. 1C
     """
     x = [-70, -60, -50, -40, -30, -20, -10, 0,
         10, 20, 30, 40, 50, 60]

     y = np.asarray([-0.008752454,-0.017321125,-0.008942166,-0.14611204,-1.684228,
                    -3.9925282,-9.197569,-13.163163,-13.675742,-11.479001,
                    -7.83745,-4.76955,-1.6496687,-0.026008366])

     yupper = np.asarray([-0.008752454,-0.017321125,-0.008942166,-0.14611204,-1.2487197,
                         -2.7532248,-6.4365516,-10.470933,-11.308362,-9.770294,
                         -6.5903134,-3.9658973,-1.3255295,-0.026008366])

     sem = yupper - y
     N = 6
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

def IV_Li_RTdeg():
     """
     IV curve for CaL - holding -80mV at Room Temperature (20°C)
     Data from Li et al '97 Fig. 1C
     """
     x = [-70, -60, -50, -40, -30, -20, -10, 0,
         10, 20, 30, 40, 50, 60]

     y = np.asarray([-0.008752454,-0.017321125,-0.008942166,-0.14611204,-1.2317721,
                    -2.2489302,-5.0264916,-7.1284323,-7.290668,-5.752457,
                    -3.8808508,-2.256906,-1.1033605,-0.01703655])

     yupper = np.asarray([-0.008752454,-0.017321125,-0.008942166,-0.14611204,-0.8303014,
                         -1.3776292,-3.120824,-4.941066,-5.2823186,-4.1971326,
                         -2.7533433,-1.5649068,-0.65873826,-0.01703655])

     sem = yupper - y
     N = 6
     sd = np.sqrt(N)*sem
     return x, y.tolist(), sd.tolist()

