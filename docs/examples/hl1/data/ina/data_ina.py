import numpy as np

### Digitised data for HL-1 i_Na channel.
# Charles Houston

# I-V curves.

def IV_Dias():
    """Data points in IV curve for i_Na in HL1-6 subclone.

    Data from figure 6 in Dias 2014. Reported as mean \pm SEM for n=12 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    y  = np.asarray([0.1865311861209591, -0.11120577919429309, 0.1503001913230264,
          0.13101595215640316, -4.915903186106931, -47.392361805051536,
          -80.65124628778395, -80.67023834150868, -75.94004822228533,
          -58.64062476259929, -39.66434905174137, -25.99620626422211,
          -12.327479105819037, 24.245957030040174, 49.366308213566526])
    N = 12
    sem = np.asarray([-7.075445787328732, -4.300560645425946, -4.319844884592598,
            -4.058923284959093, -8.82592876986422, -69.73841221755893,
            -99.08697893109215, -96.59229962798946, -86.83389023878559,
            -67.29983251930469, -46.64816548448434, -31.583084099151364,
            -17.355698375799562, 15.027506337502246, 38.75237985042443])
    sem = np.abs(y-sem)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()


def IV_Nakajima():
    """Data points in IV curve for i_Na in HL-1.

    Data from figure 4A in Nakajima 2009. Reported as mean \pm SEM for
    n=17 cells.
    """
    x = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y  = np.asarray([-1.2973226211598643, -1.0570961503957221, -1.3277247263810494,
          -4.4075576643208905, -20.769372620163068, -80.81004166583597,
          -141.10638743246744, -160.02247762205695, -160.29310619804227,
          -146.0043659416678, -124.56315663563325, -99.54646039751995,
          -74.52926576423914, -48.23593030441977, -26.028189230677214,
          -0.500637945814475])
    N = 17
    sem = np.asarray([-6.405873088654516, -6.1658958154741725, -6.435278403540593,
            -7.983293793983364, -26.389276529574758, -89.23939913478597,
            -155.4103287414525, -176.11466079224894, -174.5977950997787,
            -158.7762405055720, -136.0578935826638, -109.50813380913459,
            -82.19308825581626, -55.13247343553758, -28.327036941049812,
            -6.886575227766585])
    sem = np.abs(y-sem)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()


# Activation.

def Act_Nakajima():
    """Activation of i_Na channel in HL-1.

    Data from figure 4B in Nakajima 2009. Reported as mean \pm SEM for
    n=17 cells.
    """
    x = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10]
    y = np.asarray([-3.5219059432956E-4, 0.0036995596059203883, 0.01646568946763116,
         0.07559319181297108, 0.3332689208942523, 0.5634207991921433,
         0.8225458081527448, 0.9266009867570104, 0.9712325735782252,
         0.9796508638019754])
    N = 17
    sem = np.asarray([0.02863340699207395, 0.03123899404392705, -9.287858151343364E-4,
            0.058198716530205585, 0.25065373431115545, 0.48804889527472406,
            0.7022618116310164, 0.9454447419190959, 0.9828361628055564,
            1.0144273474438132])
    sem = np.abs(sem-y)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()


# Inactivation.

def Inact_Nakajima():
    """Inactivation data for HL-1 cells.

    Data from figure 4D in Nakajima 2009. Reported as mean \pm SEM for
    N=19 cells.
    """
    x = [-130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30]
    y = np.asarray([0.9787835926449786, 0.9830268741159829, 0.9844413012729843,
         0.9816124469589815, 0.9420084865629419, 0.7270155586987269,
         0.3465346534653464, 0.10325318246110315, 0.032531824611032434,
         0.026874115983026803, 0.021216407355021172])
    N = 19
    sem = np.asarray([0.9971711456859971, 1.004243281471004, 1.004243281471004,
            1.007072135785007, 0.966053748231966, 0.758132956152758,
            0.38896746817538885, 0.12729844413012725, 0.05516265912305496,
            0.03960396039603942, 0.04526166902404505])
    sem = np.abs(y-sem)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()


# Recovery.

def Recovery_Zhang():
    """Recovery of i_Na channel in HL-1 cells.

    Extracted from figure 4F in Zhang 2013. Data reported as mean \pm SEM for
    an unreported number of cells.

    Assume 10 cells to generate approximate SD.
    """
    x = [2, 10, 20, 30, 40, 50, 60, 70, 80]
    y = np.asarray([0.0, 0.7037787468481477, 0.8894270109940083, 0.9240166438077417,
         0.9389746360072241, 0.9493922229447606, 0.9643502151442429,
         0.974773595262337, 0.9776412196383027])
    N = 10#None
    sem = np.asarray([0.0, 0.7279478961340657, 0.9105750166191867, 0.9663169999435165,
            0.9888249547045695, 0.9947108261509964, 1.006647674689739,
            1.0140513594422327, 1.016913190637641])
    sem = np.abs(y-sem)
    sd = np.sqrt(N) * sem
    return x, y.tolist(), sd.tolist()

# Current trace.

def Trace_Nakajima():
    """Current trace from HL-1 cells.

    Data from highest peak curve in Figure 4C of control cells in
    Nakajima 2009. Stimulated from HP of -120mV to step of -20mV.
    Current in pA and time in ms.
    """
    x = [-29.855072463768114, -29.79710144927536, -29.681159420289855,
         -29.565217391304348, -29.463768115942027, -29.27536231884058,
         -29.07246376811594, -28.782608695652172, -28.217391304347824,
         -27.333333333333332, -26.3768115942029, -25.15942028985507,
         -23.89855072463768, -22.73913043478261, -20.217391304347824]
    y = [3.318681318681319, 2.0329670329670333, 0.1098901098901095,
         -0.604395604395604, -0.7692307692307692, -0.3736263736263741,
         0.5494505494505493, 1.5054945054945055, 2.4505494505494507,
         2.9450549450549453, 3.1428571428571432, 3.21978021978022,
         3.2527472527472527, 3.274725274725275, 3.285714285714286]
    x0 = x[0]
    x = [xi - x0 for xi in x]
    y0 = y[0]
    y = [yi - y0 for yi in y]
    return x, y, None

# Activation time constant

def TauAct_Dias():
    """Activation time constants.

    Generated by fitting single exponential function to current traces
    in Dias 2014 Fig 6. Voltage step values inferred from peak of trace.
    Standard deviations assumed 10%.
    See Matlab script file for details.
    """
    x = [-40]
    y = [0.3590]
    sd = np.asarray(y)*0.1
    return x, y, sd.tolist()

# Inactivation time constant

def TauInact_Dias():
    """Inactivation time constants.

    Generated by fitting single exponential function to current traces
    in Dias 2014 Fig 6. Voltage step values inferred from peak of trace.
    Standard deviations assumed 10%.
    See Matlab script file for details.
    """
    x = [-40]
    y = [2.7090]
    sd = np.asarray(y)*0.1
    #x = [-40, -30, -20]
    #y = [2.1329, 2.5981, 3.2569]
    #y = [1.2873, 1.6258, 2.4537, 2.9704]
    # assume 50% SD
    #sd = np.asarray(y)*0.5
    #sd = [0.0127,0.0383,0.3030,0.2347]
    return x, y, sd.tolist()
