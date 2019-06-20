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

def TauJ_Inactivation_Sakakibara():
    """
    Data for tau_j for i_Na
    Data from Sakakibara 1992 Fig 5B

    Errors reported as mean \pm SEM for 8 cells
    """
    x = [-50, -40, -30, -20]
    y = np.array([9.31, 5.56, 3.68, 2.62])
    N = 8
    sem = np.array([0.63, 0.32, 0.23, 0.14])
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()

def TauH_Inactivation_Sakakibara():
    """
    Data for tau_h for i_Na
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

