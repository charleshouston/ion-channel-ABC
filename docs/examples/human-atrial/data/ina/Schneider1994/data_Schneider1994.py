import numpy as np
# digitilized data from Schneider 1994 by Benjamin Marchand 07/2019

# figure 1B
def IV_Schneider():
    """
    Data points in IV curve for i_Na (nA) in human atrial cells
    from fig. 1B in Schenider 1994. No error reported.

    """
    x = [-85, -75, -65, -55, -45, -35, -25, -15, -5,
           5, 15, 25, 35, 45, 55]
    y = [-0.012040616536735271,-0.05741623375231519,-0.6072115075423579,
        -3.3836990774815483,-10.698462945627867,-12.088600358717478,
        -10.74769727674839,-8.44011490392517,-6.426723737521705,
        -3.2367821184339327,-0.6350085391909559,2.5547187069020936,
        5.324417798675175,8.135848166753608,10.527393295841879]
    return x, y, None

# figure 3C
def TauM_Activation_Schneider():
    """
    Data for tau_m for i_Na from Schneider 1994 figure 3C

    Errors reported as mean + or - SD for n=23

    digitilized the 01/07/19 by Benjamin Marchand
    updated for corrected test pulses by Charles Houston 2019-09-20
    """
    x = [-65, -55, -45, -35, -25, -15, -5, 5, 15]

    y = np.asarray([0.2802096,0.39335337,0.3340908,0.23605047,0.1959645,
                    0.16590016,0.1929139,0.17548873,0.15936272])

    ylower = np.asarray([-0.010131356,0.1502184,0.09879723,0.06263277,0.097489536,
                     0.09139314,0.15936644,0.06917234,0.12407236])

    sd = np.abs(y-ylower)
    return x, y.tolist(), sd.tolist()

# figure 3B
def TauF_Inactivation_Schneider():
    """
    Data for tau_f (fast) for i_Na from Schneider 1994 figure 3B

    Errors reported as mean + or - SD for n=23

    digitilized the 01/07/19 by Benjamin Marchand
    updated for corrected test pulses by Charles Houston 2019-09-20
    """
    x = [-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75,
         85, 95, 105]

    y = np.asarray([1.6562053,1.7388325,1.2643846,0.8249009,0.62296253,0.5015341,
                    0.358666,0.29900438,0.2890414,0.27903038,0.27574855,
                    0.273827,0.28131306,0.2646089,0.26001486,0.24330468,
                    0.22935106,0.2300779])

    ylower = np.asarray([1.4159275,1.4676893,0.9127194,0.5873136,0.47397462,
                    0.3847658,0.24457617,0.21712212,0.23132537,0.22534105,
                    0.23548757,0.23088151,0.22896595,0.22434789,0.22378056,
                    0.20303167,0.18774183,0.18175751])

    sd = np.abs(y-ylower)
    return x, y.tolist(), sd.tolist()

#figure 5B
def TauS_Inactivation_Schneider():
    """
    Data for tau_s (slow) for i_Na from Schneider 1994 figure 5B

    Errors reported as mean + or - SD for n=40

    digitilized the 01/07/19 by Benjamin Marchand
    """
    x =  [-115,-105,-95,-85,
          -75,-65,-55]

    y = np.asarray([76.365364,78.771286,74.83804,68.60557,
                    43.78963,29.352915,20.07485])

    ylower = np.asarray([47.15486,55.65152,44.25995,42.377945,
                     26.698391,13.380271,12.305942]) # lower point of the error bar

    sd = np.abs(y-ylower)
    return x, y.tolist(), sd.tolist()

# figure 4
def Inact_Schneider_32():
    """
    Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 32 ms
    n = 7

    no error reported

    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [-135.30276,-129.67128,-123.79758,-118.10554,-112.231834,-106.23702,
        -100.302765,-94.67128,-88.61591,-82.86332,-77.05017,-70.99481,
        -65.0,-59.307957,-53.373703,-47.5,-41.626297,-35.81315,
        -29.939445,-24.247404,-18.252596,-12.5,-6.444637,-0.8131488,
        4.818339]

    y = [1.0011135,1.0005568,1.0,0.99498886,0.99832964,0.9821826,
        0.9599109,0.94320714,0.903118,0.844098,0.798441,0.6937639,
        0.6330735,0.5545657,0.45935413,0.37082404,0.27449888,0.18875279,
        0.1091314,0.06013363,0.033407573,0.013919822,0.0061247214,0.004454343,
        0.0022271716]

    return x,y,None

def Inact_Schneider_64():
    """
    Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 64 ms
    n = 6

    no error reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [-135.30276,-129.67128,-123.61591,-117.98443,-111.92907,-106.17647,
        -100.24222,-94.610725,-88.79758,-82.86332,-76.80796,-70.8737,-64.939445,
        -59.247406,-53.31315,-47.439445,-41.68685,-35.81315,-29.939445,-24.247404,
        -18.313148,-12.5,-6.565744,-0.8737024,4.878893]


    y = [0.99944323,0.9955457,0.9916481,0.97995543,0.967706,0.9482183,
        0.91258353,0.87861913,0.8006682,0.7243875,0.6252784,0.52115816,0.422049,
        0.33017817,0.23218262,0.15868597,0.1013363,0.059020046,0.033964366,0.012806236,
        0.0105790645,0.014476615,0.015033407,0.018374166,0.016703786]

    return x,y,None

def Inact_Schneider_128():
    """
    Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 128 ms
    n = 6

    no error reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [-135.36333,-129.73183,-123.918686,-118.22665,-112.231834,-106.17647,
        -100.302765,-94.48962,-88.67647,-82.439445,-76.6263,-70.99481,-65.0,
        -59.18685,-53.373703,-47.439445,-41.444637,-35.570934,-29.818338,-24.186852,
        -18.131489,-12.439446,-6.3840833,-0.6314879,4.9394464]


    y = [1.0,0.99832964,0.9888641,0.97884184,0.9688196,0.95768374,0.8997773,
        0.8402004,0.766147,0.6408686,0.5044543,0.3674833,0.25,0.15534522,0.08741648,
        0.049554564,0.008908686,0.005567929,0.0061247214,0.004454343,0.0033407572,
        0.0022271716,0.0022271716,0.0022271716,5.567929e-4]

    return x,y,None

def Inact_Schneider_256():
    """
    Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 256 ms
    n = 6

    no error reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [-135.24222,-129.79239,-123.918686,-118.16609,-112.110725,-106.17647,
        -100.18166,-94.42907,-88.55537,-82.802765,-76.747406,-70.93426,-64.87889,
        -59.126297,-53.31315,-47.31834,-41.50519,-35.752594,-29.757786,-23.884083,
        -18.192041,-12.378893,-6.5051904,-0.7525951,4.9394464]


    y = [0.9977728,0.9933185,0.98608017,0.97939867,0.9292873,0.89587975,
        0.86302894,0.78452116,0.68318486,0.5573497,0.4148107,0.25334075,0.1492205,
        0.07182628,0.04844098,-0.0027839644,0.0016703786,0.0033407572,0.0022271716,
        0.0027839644,0.0022271716,0.0016703786,0.0033407572,0.00389755,0.00389755]

    return x,y,None

def Inact_Schneider_512():
    """
    Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 512 ms
    n = 6

    no error reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [-135.42387,-129.73183,-123.97924,-118.10554,-112.05017,-106.17647,
        -100.18166,-94.55017,-88.61591,-82.62111,-76.868515,-70.81315,-64.81834,
        -59.065742,-53.252594,-47.37889,-41.565742,-35.570934,-29.757786,-24.247404,
        -18.192041,-12.378893,-6.5051904,-0.9342561,4.9394464]


    y = [1.0005568,0.9927617,0.98608017,0.9682628,0.96213806,0.905902,0.85634744,
        0.7717149,0.6636971,0.5233853,0.37694877,0.21659243,0.1247216,0.067371935,
        0.041759465,-5.567929e-4,0.0066815144,0.0066815144,0.004454343,0.0066815144,
        0.0061247214,0.0072383075,0.008908686,0.016703786,0.016703786]

    return x,y,None

def Inact_Schneider_all(tprepulse : int):
    """
    Returns the Data points for inactivation curve of i_Na (h infinity) from fig 4
    prepulse duration 512 ms

    """
    try:
        assert(tprepulse in [32,64,128,256,512])

    except:
        raise("Wrong vhold, ( = {})".format(tprepulse))

    if tprepulse == 32:
        x,y,_ = Inact_Schneider_32()
    elif tprepulse == 64:
        x,y,_ = Inact_Schneider_64()
    elif tprepulse == 128:
        x,y,_ = Inact_Schneider_128()
    elif tprepulse == 256:
        x,y,_ = Inact_Schneider_256()
    elif tprepulse == 512:
        x,y,_ = Inact_Schneider_512()

    return x,y,None


# Table 1
def Inact_Schneider_Vh():
    """
    Inflexion point from inactivation curves at different prepulse
    durations.

    Errors as SD. From Table 1.
    """

    x = [32, 64, 128, 256, 512]
    y = [-61.7, -66.6, -70.4, -73.3, -72.2]
    sd = [6.4, 5.7, 4.7, 4.7, 2.6]
    return x, y, sd

def Inact_Schneider_k():
    """Slope factor from inactivation curve fits with varying prepulse
    duration.

    Errors as SD. From Table 1.
    """
    x = [32, 64, 128, 256, 512]
    y = [5.1, 5.3, 5.2, 5.0, 4.9]
    sd = [1.7, 1.0, 0.7, 0.7, 0.5]
    return x, y, sd


# figure 5A
def Reduction_Schneider_65():
    """
    Recovery data from Figure 5A with Vinact = -65 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [8.743764,16.193748,19.507229,24.475739,33.99996,
        46.834675,63.40293,89.90158,126.74833,176.42209,
        246.366,346.4942,487.5943,686.60205,970.8316]

    y = [0.98814225,0.9762854,0.96442544,0.95138055,0.9229179,
        0.89089906,0.829227,0.75985175,0.67150444,0.57249045,
        0.46993282,0.4012053,0.2755688,0.15946512,0.016141161]

    return x,y,None

def Reduction_Schneider_75():
    """
    Recovery data from Figure 5A with Vinact = -75 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [15.361528,23.638813,34.398857,46.814354,66.26448,
        90.6814,127.09631,177.16962,248.34335,348.46985,
        489.96478,688.1217,969.4144]

    y = [0.9899265,0.97807026,0.9638434,0.9472453,0.923535,
        0.8915247,0.8535921,0.79372424,0.72200966,0.6580271,
        0.5845853,0.533724,0.50486976]

    return x,y,None

def Reduction_Schneider_85():
    """
    Recovery data from Figure 5A with Vinact = -85 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [15.36217,24.466114,36.8814,46.400063,64.19623,
        90.67862,127.08968,176.73906,249.14008,348.0057,
        489.89142,690.09924,971.7973]

    y = [0.98814714,0.97807086,0.9620659,0.9490243,0.92353344,
        0.8992352,0.87197876,0.8405803,0.8068262,0.7980027,
        0.7880249,0.78520775,0.7794854]

    return x,y,None

def Reduction_Schneider_95():
    """
    Recovery data from Figure 5A with Vinact = -95 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [14.943601,20.324263,32.32205,47.215385,64.17569,
        91.478966,128.29407,178.76134,247.84116,349.60043,
        490.65518,690.449,972.56024]

    y = [1.0017885,0.9928958,0.9875666,0.98223954,0.9804728,
        0.97396874,0.9734029,0.9681023,0.96756035,0.96407706,
        0.96418166,0.96255046,0.95801467]

    return x,y,None

def Reduction_Schneider_105():
    """
    Recovery data from Figure 5A with Vinact = -105 mV

    No errors reported
    digitilized the 08/07/19 by Benjamin Marchand
    """

    x = [18.251093,26.939888,35.213753,48.036484,65.82366,
        91.88406,126.21619,177.50777,249.48314,349.5861,
        491.46774,691.26044,972.54205]

    y = [1.006536,1.0006112,0.9982449,0.9994406,0.99886066,
        0.9976938,1.0000917,1.0030953,1.0025556,1.003816,
        1.0051074,1.0064418,1.0084296]

    return x,y,None

def Reduction_Schneider_all(Vinact : int):
    """
    Returns the Reduction of the i_Na peak current data from Figure 5A with Vinact between -105 and -65

    """
    try:
        assert(Vinact in [-105,-95,-85,-75,-65])

    except:
        raise("Wrong Vinact, ( = {})".format(Vinact))

    if Vinact == -105:
        x,y,_ = Reduction_Schneider_105()
    elif Vinact == -95:
        x,y,_ = Reduction_Schneider_95()
    elif Vinact == -85:
        x,y,_ = Reduction_Schneider_85()
    elif Vinact == -75:
        x,y,_ = Reduction_Schneider_75()
    elif Vinact == -65:
        x,y,_ = Reduction_Schneider_65()
    return x,y,None


# figure 6
def Recovery_Schneider_75():
    """
    Recovery data from Figure 6 with Vhold = -75 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """


    x = [1.9999113,2.8368287,4.002183,5.656388,7.979884,
        11.257825,15.739073,22.204203,31.098347,43.55496,
        61.66985,86.371956,122.068596,171.2765,242.94131,
        340.8718,478.27234,678.4105,957.06775]

    y = [0.015055522,0.0147306705,0.013307083,0.01847228,0.022173138,
        0.025507964,0.028841723,0.033640675,0.04429528,0.056414016,
        0.056454983,0.068939745,0.082889706,0.089518175,0.104200624,
        0.11412338,0.12843852,0.13214003,0.14169739]


    return x,y,None

def Recovery_Schneider_85():
    """
    Recovery data from Figure 6 with Vhold = -85 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """


    x = [2.0180357,2.8367448,4.0020776,5.6254973,7.907448,
        11.256933,15.851812,22.000694,31.035515,43.700985,
        61.091663,86.49018,119.380806,167.49161,238.42236,
        336.94565,472.76062,663.3173,945.99835]

    y = [0.027135637,0.02497956,0.022457877,0.03311291,0.043767944,
        0.052960344,0.066177815,0.085982144,0.117501415,0.15194872,
        0.18236883,0.2226733,0.26626906,0.30108196,0.33406642,
        0.36265767,0.379169,0.3967784,0.40706912]

    return x,y,None

def Recovery_Schneider_95():
    """
    Recovery data from Figure 6

    Vhold = -95 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """


    x = [2.0032268,2.834315,4.0207086,5.6573195,7.938326,11.138813,
        15.626384,21.628338,30.419765,42.789165,60.67295,84.88165,
        120.378174,168.44931,240.26764,336.24655,473.16193,667.65424,
        947.20306]

    y = [0.03416171,0.038089674,0.05850174,0.084406376,0.11525541,
        0.14940114,0.21981047,0.27702984,0.37326393,0.4513662,
        0.5492505,0.6289998,0.6977634,0.73630404,0.75012404,
        0.7721812,0.7805035,0.78388137,0.78945833]


    return x,y,None

def Recovery_Schneider_105():
    """
    Recovery data from Figure 6 with Vhold = -105 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [2.0105047,2.867146,4.0445147,5.6534324,7.873399,
        11.104783,15.492207,22.167637,31.039038,44.098167,
        60.536694,85.0794,121.11095,171.47789,238.86885,
        336.98886,476.27,679.257,953.12225]

    y = [0.067398705,0.09708909,0.13336678,0.19965799,0.28388435,
        0.3750668,0.48601365,0.60501623,0.7060807,0.7829887,
        0.83427054,0.90056235,0.9295205,0.9379802,0.9413135,
        0.94611245,0.954938,0.95571184,0.9458689]

    return x,y,None

def Recovery_Schneider_115():
    """
    Recovery data from Figure 6 with Vhold = -115 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [2.0058124,2.8210201,3.941625,5.567115,7.931059,
        11.10456,15.719857,22.104898,30.753572,43.25592,
        60.976986,85.02802,120.91444,171.20338,239.78423,
        331.49136,476.54965,668.9403,955.3748]

    y = [0.12391375,0.1937928,0.27553052,0.38450226,0.5171939,
        0.6331925,0.70351154,0.8340054,0.89509845,0.93774474,
        0.95930785,0.9843836,0.9901352,0.9910546,0.9968044,
        0.99684256,1.0025954,0.9868228,1.0066304]

    return x,y,None

def Recovery_Schneider_125():
    """
    Recovery data from Figure 6 with Vhold = -125 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [1.970509,2.7953246,3.9825203,5.563751,7.8923855,
        11.050929,15.713047,21.719032,30.616953,43.25373,
        60.512028,85.366806,120.214424,169.29097,239.68134,
        335.09863,475.32074,668.18335,942.6288]

    y = [0.25173,0.35191748,0.46484342,0.5940188,0.7047481,
        0.8044948,0.8537309,0.91087,0.9315547,0.9553143,
        0.97555876,0.9873123,0.9935752,0.9921514,1.0200107,
        1.0123636,1.0186273,1.002562,1.0161457]

    return x,y,None

def Recovery_Schneider_135():
    """
    Recovery data from Figure 6 with Vhold = -135 mV

    No errors reported
    digitilized the 01/07/19 by Benjamin Marchand
    """

    x = [1.9940444,2.8222792,3.9877973,5.613986,7.932807,
        11.048715,15.557898,21.748865,30.460009,42.893604,
        60.511707,85.214836,120.43695,170.21446,239.2625,
        336.3286,477.93134,663.35974,947.81793]

    y = [0.4056112,0.5414499,0.63373065,0.7432145,0.81755966,
        0.87396765,0.9029245,0.9373709,0.95498013,0.9652693,
        0.9773889,0.97706324,0.98039824,0.98995584,0.9984147,
        0.9980888,1.0036206,1.0025612,0.99674666]

    return x,y,None


def Recovery_Schneider_all(Vhold : int):
    """
    Returns the Recovery data from Figure 6 with Vhold between -135 and -75

    """
    try:
        assert(Vhold in [-135,-125,-115,-105,-95,-85,-75])

    except:
        raise("Wrong vhold, ( = {})".format(Vhold))

    if Vhold == -135:
        x,y,_ = Recovery_Schneider_135()
    elif Vhold == -125:
        x,y,_ = Recovery_Schneider_125()
    elif Vhold == -115:
        x,y,_ = Recovery_Schneider_115()
    elif Vhold == -105:
        x,y,_ = Recovery_Schneider_105()
    elif Vhold == -95:
        x,y,_ = Recovery_Schneider_95()
    elif Vhold == -85:
        x,y,_ = Recovery_Schneider_85()
    elif Vhold == -75:
        x,y,_ = Recovery_Schneider_75()

    return x,y,None

# Table 2
def Recovery_Schneider_tau_r1():
    """Time constants from double exponential fit to recovery curve."""

    x = [-135,-125,-115,-105,-95]
    y = [1.6,4.7,6.5,9.6,15.9]
    sd = [0.2,3.1,4.1,2.5,9.4]
    return x, y, sd

def Recovery_Schneider_tau_r2():
    """Time constants from double exponential fit to recovery curve."""

    x = [-135,-125,-115,-105,-95,-85]
    y = [8.6,19.8,33.5,48.7,53.2,80.6]
    sd = [2.9,10.2,17.2,23.1,33.3,47.6]

    return x, y, sd
