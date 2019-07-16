import numpy as np
### Digitised data for human atrial myocytes.
# Author: Lukas Engelbert 2019
# Amended: Charles Houston 2019-06-12

# IV curve

def IV_Schneider():
    """
    Data points in IV curve for i_Na in human atrial cells
    from fig. 1B in Schenider 1994. No error reported.

    Already converted to nA/pF
    """
    x = [-85, -75, -65, -55, -45, -35, -25, -15, -5,
           5, 15, 25, 35, 45, 55]
    y = [-0.012040616536735271,
        -0.05741623375231519,
        -0.6072115075423579,
        -3.3836990774815483,
        -10.698462945627867,
        -12.088600358717478,
        -10.74769727674839,
        -8.44011490392517,
        -6.426723737521705,
        -3.2367821184339327,
        -0.6350085391909559,
        2.5547187069020936,
        5.324417798675175,
        8.135848166753608,
        10.527393295841879]
    #y = [-0.3969, -0.1152, -4.8705, -29.7854, -92.2440,
    #     -104.0544, -92.6832, -72.7459, -54.5716, -28.0836,
    #     -5.3740, 22.6277, 46.0932, 71.0738, 91.0111]
    return x, y, None

def IV_Sakakibara():
    """
    Data points in IV curve for i_Na in human atrial cells
    from fig. 1B in Sakakibara 1992

    Already converted to nA/pF

    No errors reported in data.
    """

    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    y = [0.0, 0.0, 0.0, -0.1555, -1.3371, -6.6232, -14.6457, -17.0089,
         -13.7750, -7.1207, 2.0212, 11.5362, 21.5487]
    return x, y, None


# Activation

def Act_Sakakibara():
    """
    Activation curve of i_Na current
    Data from Sakakibara 1992 - Fig. 2.

    Data expressed as mean \pm SEM for 46 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    y = np.array([0.0000, 0.0012, 0.0038, 0.0203, 0.0376, 0.1744, 0.4454,
         0.7081, 0.8790, 0.9671, 0.9913, 1.0071, 1.0076])
    N = 46
    sem = np.array([0.0000, 0.0019, 0.0045, 0.0217, 0.0382, 0.1938, 0.4691,
            0.7248, 0.8915, 0.9678, 0.9920, 1.0064, 1.0062])
    sem = np.abs(y-sem)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()

# Inactivation

def Inact_Sakakibara():
    """
    Inactivation curve for i_Na
    Data from Sakakibara 1992 - Fig 7.

    Data expressed as mean \pm SEM for 46 cells.
    """
    x = [-140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40]
    y = np.array([1.0000, 1.0011, 0.9849, 0.9041, 0.6699, 0.3157, 0.0901,
         0.0299, 0.0284, 0.0229, 0.0093])
    N = 46
    sem = np.array([1.0000, 1.0011, 0.9849, 0.9041, 0.6699, 0.3357, 0.1208,
            0.0566, 0.0444, 0.0229, 0.0093])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()

def Inact_Schneider_32():
    """Inactivation curve for i_Na from Schneider 1994 - Fig 4.

    32ms prepulse duration.
    """
    x = np.linspace(-135, 5, 25)
    y = [0.99883,
         0.99888,
         0.99776,
         0.99431,
         0.99786,
         0.98043,
         0.95833,
         0.94323,
         0.90015,
         0.84309,
         0.80002,
         0.69284,
         0.63229,
         0.55308,
         0.45756,
         0.37019,
         0.27467,
         0.18964,
         0.11043,
         0.06036,
         0.03477,
         0.00102,
         -0.00010,
         -0.00005,
         0.00117]
    return x, y, None

def Inact_Schneider_64():
    """Inactivation curve for i_Na from Schneider 1994 - Fig 4.

    64ms prepulse duration.
    """
    x = np.linspace(-135, 5, 25)
    y = [0.99767,
         0.99538,
         0.99194,
         0.97217,
         0.96523,
         0.94779,
         0.91055,
         0.87796,
         0.79992,
         0.72305,
         0.62636,
         0.52151,
         0.42250,
         0.32814,
         0.23145,
         0.15924,
         0.10218,
         0.05910,
         0.03584,
         0.01141,
         0.00913,
         0.00335,
         0.00340,
         0.00462,
         0.00700]
    return x, y, None

def Inact_Schneider_128():
    """Inactivation curve for i_Na from Schneider 1994 - Fig 4.

    128ms prepulse duration.
    """
    x = np.linspace(-135, 5, 25)
    y = [0.99767,
         0.99422,
         0.98494,
         0.96868,
         0.96640,
         0.94779,
         0.89773,
         0.83833,
         0.76729,
         0.64030,
         0.50515,
         0.36767,
         0.25233,
         0.15448,
         0.08926,
         0.04968,
         0.01010,
         0.00316,
         0.00321,
         0.00326,
         0.00097,
         0.00102,
         0.00224,
         0.00228,
         0.00233]
    return x, y, None

def Inact_Schneider_256():
    """Inactivation curve for i_Na from Schneider 1994 - Fig 4.

    256ms prepulse duration.
    """
    x = np.linspace(-135, 5, 25)
    y = [1.00000,
         0.99072,
         0.98378,
         0.96518,
         0.92910,
         0.89535,
         0.86160,
         0.78472,
         0.68337,
         0.55638,
         0.41424,
         0.25228,
         0.14860,
         0.07289,
         0.04730,
         -0.00043,
         0.01010,
         0.00316,
         0.00204,
         0.00442,
         -0.00019,
         0.00219,
         0.00340,
         0.00578,
         0.00466]
    return x, y, None

def Inact_Schneider_512():
    """Inactivation curve for i_Na from Schneider 1994 - Fig 4.

    512ms prepulse duration.
    """
    x = np.linspace(-135, 5, 25)
    y = [0.99767,
         0.98956,
         0.98728,
         0.96984,
         0.95940,
         0.90933,
         0.85344,
         0.77423,
         0.66356,
         0.52258,
         0.37461,
         0.21848,
         0.12646,
         0.06823,
         0.04380,
         -0.00277,
         0.00894,
         0.00083,
         0.00204,
         0.00209,
         -0.00019,
         0.00452,
         0.00340,
         0.00695,
         0.01166]
    return x, y, None

# Time constant activation

def TauM_Activation_Schneider():
    """
    Data for tau_m for i_Na
    Data from Schneider '94 figure 3C

    Errors reported as mean \pm SD for n=23
    """
    x = [-65, -55, -45, -35, -25, -15, -5, 5, 15]
    y = np.array([0.2793, 0.3933, 0.3348, 0.2357, 0.1969,
         0.1676, 0.1939, 0.1760, 0.1611])
    #N = 23
    sd = np.array([0.5670, 0.6416, 0.5766, 0.4136, 0.2996,
            0.2464, 0.2339, 0.2834, 0.1993])
    sd = np.abs(y-sd)
    return x, y.tolist(), sd.tolist()

# Time constants inactivation

def TauH_Inactivation_Schneider():
    """
    Data for tau_h for i_Na
    Data from Schneider '94 figure 3B

    Data reported as mean \pm SD
    """
    x = [-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45,
         55, 65, 75, 85, 95, 105]
    y = np.asarray([1.6611, 1.7457, 1.2700, 0.8196, 0.6182, 0.4986, 0.3556,
         0.2923, 0.2894, 0.2729, 0.2738, 0.2748, 0.2836, 0.2690,
         0.2623, 0.2437, 0.2292, 0.2263])
    sd = np.asarray([1.8750, 2.0161, 1.6124, 1.0511, 0.7661, 0.6075, 0.4626,
           0.3799, 0.3458, 0.3312, 0.3128, 0.3137, 0.3303, 0.3157,
           0.3011, 0.2826, 0.2681, 0.2730])
    sd = np.abs(y-sd)

    return x, y.tolist(), sd.tolist()

def TauH_Inactivation_Sakakibara():
    """
    Data for tau_h for i_Na
    Data from Sakakibara 1992 Fig 5B

    Errors reported as mean \pm SEM for 8 cells
    """
    x = [-50, -40, -30, -20]
    y = np.array([9.31, 5.56, 3.68, 2.62])
    N = 8
    sem = np.array([0.63, 0.32, 0.23, 0.14])
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def TauJ_Inactivation_Sakakibara():
    """
    Data for tau_j for i_Na
    Data from Sakakibara 1992 Fig 5B

    Errors reported as mean \pm SEM for 8 cells
    """
    x = [-50, -40, -30, -20]
    y = np.array([59.2, 40.7, 16.9, 11.9])
    N = 8
    sem = np.array([0.6, 5.7, 2.0, 1.0])
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def Rel_Tf_Sakakibara():
    """
    Data for percentage fast inactivation for i_Na
    Data from Sakakibara 1992 Figure 5B

    Errors reported as mean \pm SEM for 8 cells
    """
    x = [-50, -40, -30, -20]
    y = [0.88, 0.91, 0.92, 0.91]
    N = 8
    sem = np.array([0.04, 0.02, 0.01, 0.02])
    sd = np.sqrt(N)*sem
    return x, y, sd.tolist()

def TauH_Inactivation_Sakakibara_Depol():
    """
    Data for fast inactivation time constant
    from Sakakibara Fig 9.

    Errors reported as mean \pm SEM for 12,11,11,12,4 cells.
    """
    x = [-140, -120, -110, -100, -90]
    y = [7.3419060624046795,
    20.79767110039968,
    45.124433704328595,
    70.37630653239292,
    93.05720409296985]
    return x, y, None

def TauJ_Inactivation_Sakakibara_Depol():
    """
    Data for slow inactivation time constant
    from Sakakibara Fig 9.

    Errors reported same as previous.
    """
    x = [-140, -120, -110, -100, -90]
    y = [73.10895701613708,
    201.90499294522908,
    284.4750075672543,
    497.3832737043119,
    737.304795871682]
    return x, y, None
