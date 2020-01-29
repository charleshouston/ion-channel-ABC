import numpy as np
# digitilized data from Nygren 1998 by Benjamin Marchand 07/2019

"""
NB : Na_c in the nygren_Na.mmt file corresponds to Na_o in the Sakakibara paper according to equation (3) in Sakakibara paper.
"""

# figure 1B
def IV_Sakakibara():
   """
   Data points in IV curve for i_Na in human atrial cells
   from fig. 1B in Sakakibara 1992

   Already converted to pA/pF ( cell capacitance = 0.1161 nF)

   No errors reported in data.
   """

   x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
   y = [0.0, 0.0, 0.0, -0.1555, -1.3371, -6.6232, -14.6457, -17.0089,
        -13.7750, -7.1207, 2.0212, 11.5362, 21.5487]
   return x, y, None

# figure 1B
# def IV_Sakakibara():
#     """
#     Data points in IV curve for i_Na (nA) in human atrial cells
#     from fig. 1B in Sakakibara 1992

#     No errors reported in data.
#     """

#     x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
#     y = [0,0,0,-0.023617903,-0.14940734,
#         -0.76687074,-1.6884719,-1.9643742,-1.5946794,-0.81199026,
#         0.25222388,1.3464108,2.5195153]

#     return x, y, None


# figure 3A
def IV_Sakakibara_fig3A_2():
    """
    IV curve (nA) data from Figure 3A in Sakakibara 1992 with [Na+]o = 2mM
    No error reported
    Already converted to pA/pF ( cell capacitance = 0.1161 nF)
    digitilized the 04/07/19 by Benjamin Marchand
    """
    x = [-100.05277,-90.02083,-80.25387,-70.022995,-60.125435,
        -50.103706,-40.20615,-30.306698,-20.517422,-10.796095,
        -0.80107594,9.591806,19.985445,30.13301,40.60368]

    y = [-0.040171184,-0.04726624,-0.03767147,-0.05315843,-0.093163274,
        -0.32303375,-0.3630386,-0.36178875,0.13461158,0.5980551,
        1.2345805,1.8543221,2.4905655,3.5560458,4.423269]

    y = np.asarray(y)/0.1161 # converting to pA/pF
    y = y.tolist()

    return x , y , None

def IV_Sakakibara_fig3A_5():
    """
    IV curve (nA) data from Figure 3A in Sakakibara 1992 with [Na+]o = 5mM
    No error reported
    Already converted to pA/pF ( cell capacitance = 0.1161 nF)
    digitilized the 04/07/19 by Benjamin Marchand
    """
    x = [-100.11921,-90.087265,-80.18857,-70.35669,-60.14095,
        -50.00375,-40.259872,-30.298147,-20.716883,-10.588613,
        -0.65019345,9.621544,19.828358,29.975168,40.581738]

    y = [-0.040124197,-0.047219254,-0.062471278,-0.085927255,-0.4314519,
        -1.0409596,-1.5346723,-1.6242298,-1.3174602,-0.67277783,
        0.178319,1.0539337,1.9625992,3.0115776,3.944714]

    y = np.asarray(y)/0.1161 # converting to pA/pF
    y = y.tolist()

    return x , y , None

def IV_Sakakibara_fig3A_20():
    """
    IV curve (nA) data from Figure 3A in Sakakibara 1992 with [Na+]o = 20mM
    No error reported
    Already converted to pA/pF ( cell capacitance = 0.1161 nF)
    digitilized the 04/07/19 by Benjamin Marchand
    """
    x = [-100.11921,-90.087265,-80.3222,-70.1657,-60.301064,
        -50.18293,-40.455853,-30.70895,-20.923836,-10.911792,
        -0.97412926,9.379557,19.483387,29.70874,39.92766]

    y = [-0.040124197,-0.047219254,-0.078879185,-0.26758894,-1.0254257,
        -3.4996932,-5.80866,-6.236365,-5.830725,-4.8229074,
        -3.9883125,-2.774456,-1.2138723,0.099089295,1.2717849]

    y = np.asarray(y)/0.1161 # converting to pA/pF
    y = y.tolist()

    return x , y , None

def IV_Sakakibara_fig3A_all(Na : int):
    """
    Returns the IV curve (nA) data from Figure 3A in Sakakibara 1992 with [Na+]o = 2,5 or 20 mM

    """
    try:
        assert(Na in [2,5,20])
    except:
        raise Exception("Wrong Na concentration, ( = {} mM)".format(Na))

    if Na == 2:
        x,y,_ = IV_Sakakibara_fig3A_2()
    elif Na == 5:
        x,y,_ = IV_Sakakibara_fig3A_5()
    elif Na == 20:
        x,y,_ = IV_Sakakibara_fig3A_20()

    return x,y,None

# figure 2
def Act_Sakakibara():
    """
    Activation data from Figure 2 in Sakakibara 1992

    error reported as + or - SEM (N=46)

    digitilized the 04/07/19 by Benjamin Marchand
    """

    x = [-100,-90,-80,-70,-60,
        -50,-40,-30,-20,-10,
        0,10,20]

    y = np.asarray([0.0023289563,0.0020447162,0.0052403254,0.018904742,0.036657896,
                    0.17249742,0.44329727,0.70652646,0.87842613,0.9642359,
                    0.9895305,1.0043688,1.0046475])

    ylower = np.asarray([0.0023289563,0.0020447162,0.0052403254,0.018904742,0.036657896,
                        0.15296747,0.41819438,0.6863299,0.87842613,0.9642359,
                        0.9895305,1.0043688,1.0046475])
    sem = y - ylower
    N = 46
    sd = np.sqrt(N)*sem

    return x , y.tolist(), sd.tolist()

# figure 7
def Act_Sakakibara_fig7():
    """Data points for activation curve of i_Na (gNa/gNaMax).
    It's probably betteer to use the Act_Sakakibara function for the figure 2 (one more point)
    Extracted from figure 7 in Nygren 1992. Reported as mean /pm SEM for n=46 cell.
    """

    x = [-90.1665,-80.1659,-70.3305,-60.2444,-50.3087,
        -40.2702,-30.1487,-19.9573,-9.77818,0.391688,
        10.6441,20.5609]

    y = np.asarray([0.00264315,0.00679469,0.0212131,0.0404606,0.172631,
                    0.440670,0.705689,0.877712,0.964592,0.986858,
                    1.00369,1.00482])

    ylower = np.asarray([0.00264315, 0.00679469, 0.0212131, 0.0404606, 0.172631,
                        0.44067, 0.685762, 0.853256, 0.9446645, 0.986858,
                        1.00369, 1.00482])

    sem = y - ylower
    N = 46
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def Inact_Sakakibara():
    """Data points for inactivation curve of i_Na (h infinity).

    Extracted from figure 7 in Sakakibara 1992. Reported as mean /pm SEM for n=46 cell.
    """
    x = [-140,-130,-120,-110,-100,
        -90,-80,-70,-60,-50,
        -40]

    y = np.asarray([1.00242,1.00234,0.986563,0.904362,0.669383,
                    0.313029,0.0859017,0.0260446,0.0229492,0.0174390,
                    0.00589079])

    ylower = np.asarray([1.00242, 1.00234, 0.986563, 0.904362, 0.669383,
                        0.2933275, 0.0541988, -0.00257837, 0.0114761, 0.017439,
                        0.00589079])


    sem = y-ylower
    N = 46
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

# figure 3B
def Reversal_potential_Sakakibara():
    """
    Reversal potential data from Figure 3B in Sakakibara 1992
    x corresponds to [Na+]o and y to the Reversal potential
    Error reported as + or - SEM (N=4)
    digitilized the 08/07/19 by Benjamin Marchand
    """
    x = [2,5,10,20]
    y = np.asarray([-26.78964,-3.5984309,16.029884,28.931164])

    ylower = np.asarray([-29.294262,-5.9605193,13.596818,26.498098])

    sem = y-ylower
    N = 4
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

# Figure 5B
def TauF_Inactivation_Sakakibara():
    """
    Data for tau_f (fast) for i_Na
    Data from Sakakibara 1992 Fig 5B

    Errors reported as mean + or - SEM for 8 cells
    """
    x = [-50, -40, -30, -20]

    y = np.array([9.31, 5.56, 3.68, 2.62])

    sem = np.array([0.63, 0.32, 0.23, 0.14])
    N = 8
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def TauS_Inactivation_Sakakibara():
    """
    Data for tau_s (slow) for i_Na
    Data from Sakakibara 1992 Fig 5B

    Errors reported as mean + or - SEM for 8 cells
    """
    x = [-50, -40, -30, -20]

    y = np.array([59.2, 40.7, 16.9, 11.9])

    sem = np.array([0.6, 5.7, 2.0, 1.0])
    N = 8
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def Rel_Tf_Sakakibara():
    """
    Data for percentage fast inactivation for i_Na
    Data from Sakakibara 1992 Figure 5B

    Errors reported as mean + or - SEM for 8 cells
    """
    x = [-50, -40, -30, -20]

    y = [0.88, 0.91, 0.92, 0.91]

    sem = np.array([0.04, 0.02, 0.01, 0.02])
    N = 8
    sd = np.sqrt(N)*sem
    return x, y, sd.tolist()

# figure 6
def Time_course_Inactivation_Sakakibara_100():
    """
    Data for the time course inactivation at Vcond = -100mV for i_Na
    Data from Sakakibara 1992 Fig 6

    No Errors reported
    """
    x = [13.618383,48.543922,104.8536,209.70401,307.70993,
        406.94058,507.34506,608.8216,808.44904,1009.2071,
        1506.5621,2008.5006]

    y = [1.0001771,0.9633226,0.8897682,0.84734863,0.7554476,
        0.7322206,0.73836476,0.70938736,0.6887862,0.67025256,
        0.6111179,0.59258205]

    return x, y, None

def Time_course_Inactivation_Sakakibara_80():
    """
    Data for the time course inactivation at Vcond = -80mV for i_Na
    Data from Sakakibara 1992 Fig 6

    No Errors reported
    """
    x = [5.7145677,29.230057,59.41954,107.55055,209.5316,
        309.55435,406.0836,507.48373,605.4515,806.0798,
        1007.6961]

    y = [0.9936339,0.8555152,0.6790062,0.49132293,0.27539366,
        0.1996656,0.13169993,0.11845193,0.10217777,0.08887822,
        0.068500906]

    return x, y, None

def Time_course_Inactivation_Sakakibara_all(Vcond : int):
    """
    Returns the time course inactivation from Figure 6 in Sakakibara 1992 with Vcond = -80 or -100 mV

    """
    try:
        assert(Vcond in [-80,-100])
    except:
        raise Exception("Wrong Vcond, ( = {} mV)".format(Vcond))

    if Vcond == -80:
        x,y,_ = Time_course_Inactivation_Sakakibara_80()
    elif Vcond == -100:
        x,y,_ = Time_course_Inactivation_Sakakibara_100()

    return x,y,None

#figure 8A
def Recovery_Sakakibara_100():
    """
    Recovery data from Figure 8A with Vhold = -100 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [12.087913,21.428572,25.824175,29.67033,32.967033,
        38.46154,43.956043,47.252747,58.241756,74.72527,
        99.45055,148.9011,200.54945,299.45056,396.7033,
        496.15384,597.25275,696.1539,800.0,898.901,
        1000.54944]

    y = [0.003197442,0.039168663,0.06235012,0.07993605,0.11350919,
        0.13669065,0.16466826,0.19664268,0.2669864,0.3261391,
        0.42605916,0.57633895,0.65067947,0.7665867,0.8465228,
        0.90407676,0.911271,0.9376499,0.940048,0.95683455,
        0.9688249]

    return x,y,None

def Recovery_Sakakibara_120():
    """
    Recovery data from Figure 8A with Vhold = -120 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [2,10.439561,13.736263,18.681318,
        22.527473,28.021978,32.967033,38.46154,43.956043,
        47.802197,58.241756,72.52747,98.35165,147.8022,
        197.25275,297.8022,395.05493,495.6044,596.7033,
        696.1539,798.3516,899.45056,1000.54944]

    y = [0.044764187,0.17585932,0.2797762,0.39568347,
        0.47881696,0.5115907,0.56754595,0.60751396,0.6434852,
        0.6722622,0.7282174,0.76978415,0.8201439,0.87929654,
        0.90407676,0.9416467,0.95363706,0.960032,0.97202235,
        0.9816147,0.9880096,0.9872102,0.9872102]

    return x,y,None

def Recovery_Sakakibara_140():
    """
    Recovery data from Figure 8A with Vhold = -140 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [2,8.791209,13.186813,18.681318,
        21.978022,26.923077,32.967033,37.912086,41.758244,
        49.45055,56.043957,71.42857,98.9011,147.8022,
        197.8022,297.8022,396.15384,495.6044,597.25275,
        697.8022,799.45056,898.9011,1002.1978]

    y = [0.27657872,0.54996,0.66746604,0.7553957,
        0.80255795,0.8321343,0.853717,0.86970425,0.8832934,
        0.8960831,0.9072742,0.92645884,0.95043963,0.9672262,
        0.9760192,0.99040765,0.9976019,0.9984013,1.0047961,
        1.0071943,1.0079936,1.0071943,1.008793]

    return x,y,None

def Recovery_Sakakibara_all(Vhold : int):
    """
    Returns the Recovery data from Figure 8A with Vhold between -140 and -100
    digitilized by Benjamin Marchand
    """
    try:
        assert(Vhold in [-140,-120,-100])

    except:
        raise Exception("Wrong Vhold, ( = {} mV)".format(Vhold))

    if Vhold == -140:
        x,y,_ = Recovery_Sakakibara_140()
    elif Vhold == -120:
        x,y,_ = Recovery_Sakakibara_120()
    elif Vhold == -100:
        x,y,_ = Recovery_Sakakibara_100()

    return x,y,None


def TauF_Recovery():
    """
    Data for fast recovery time constant
    from Sakakibara Fig 9.
    Errors reported as mean + or - SEM for 12,11,11,12,4 cells.
    """
    x = [-140, -120, -110, -100, -90]
    y = np.array([7.182494525483941,
         21.080778238617775,
         44.85613625942195,
         71.15848950350686,
         94.1204967268066])

    sem = np.array([7.921098705433488,
           22.607420491967787,
           48.78194467418629,
           77.38627950439239,
           102.35792127703817])
    sem = np.abs(y-sem)
    N = [12, 11, 11, 12, 4]
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def TauS_Recovery():
    """
    Data for slow recovery time constant
    from Sakakibara Fig 9.
    Errors reported same as previous.
    """
    x = [-140, -120, -110, -100, -90]
    y = np.array([73.17664087948957,
         200.27163030670607,
         288.08070424186946,
         503.9983907936142,
         735.1852092122435])
    sem = np.array([63.62733372428446,
                    166.9829252154457,
                    257.591290037445,
                    444.3992111457802,
                    596.0787555812192])
    sem = np.abs(y-sem)
    N = [12, 11, 11, 12, 4]
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()
