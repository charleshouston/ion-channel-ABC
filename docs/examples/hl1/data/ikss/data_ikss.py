import numpy as np
### Digitised data for HL-1 i_ss current.


# IV curve

def IV_Lu():
    """IV curve for i_ss in HL-1 cells in Lu 2016.

    Extracted from Figure 3A as 'IKsus'.

    Recorded at 35C.

    Values are mean \pm SEM for n=14 cells.
    """
    x = np.arange(-40, 70, 10)
    y = np.asarray([0.02191780821917888,
                    0.5260273972602754,
                    1.1835616438356205,
                    2.147945205479454,
                    3.090410958904112,
                    4.558904109589042,
                    6.0493150684931525,
                    7.517808219178084,
                    9.139726027397261,
                   11.112328767123289,
                   12.909589041095892])
    N = 14
    sem = np.asarray([0.04383561643836131,
                      0.6356164383561662,
                      1.3150684931506902,
                      2.41095890410959,
                      3.397260273972604,
                      5.15068493150685,
                      6.838356164383564,
                      8.613698630136987,
                      10.564383561643838,
                      12.778082191780824,
                      14.75068493150685])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem
    return x.tolist(), y.tolist(), sd.tolist()


# Activation time constant

def ActTau_Xu():
    """Activation time constants from Xu 1999 in mouse
    ventricular cells.

    Using for HL-1 as i_to in mouse atrial cells (which HL-1 are derived
    from) is similar to i_to_f in ventricular cells [Xu 1999], and no
    existing data for HL-1.

    Errors are sem for n=4 cells (see table iii and p669 Xu 1999).

    Recorded at room temperature.

    Figure 8B for 4 cells as mean \pm SEM in Xu 1999.
    """
    x = np.arange(0, 70, 10)
    y = np.asarray([52.28865080530902,
                    29.094803280941257,
                    22.135454920227254,
                    16.598511161549872,
                    15.329255059539328,
                    14.06009372764018,
                    10.949928211640618])
    N = 4
    sem = np.asarray([48.27191440363538,
                      28.341759975738853,
                      20.126991949279038,
                      15.510645052763266,
                      13.404568867070708,
                      11.884266739955557,
                      8.439373190483188])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem
    return x.tolist(), y.tolist(), sd.tolist()
