import numpy as np
### Digitised data for HL-1 i_K1 channel.


# I-V curves.

def IV_Goldoni():
    """IV curve for i_K1 in HL-1 cell.

    Data reported as single points in figure 3 from Goldoni 2010.
    """
    x = [-150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40,
         -30, -20, -10, 0, 10, 20, 30]
    y = [-8.610370907694442, -7.640677170562069, -7.095740481955988,
         -6.417375164012246, -5.690280874305283, -5.048312940699572,
         -4.066458295552736, -3.4365872651824665, -2.466744183565707,
         -1.7640357146666945, -1.0006507152534052, -0.6013675687784685,
         -0.28718944347844655, 0.05139583755587118, 0.26854271785626693,
         0.43702463117246193, 0.532775780591618, 0.5435072485412249,
         0.42070341252146815]
    # Max current at -150mV reported as -42.4 \pm 9.4 pA/pF
    peak_out_curr = y[0]
    y = [yi * -42.4 / peak_out_curr for yi in y]
    N = 10
    return x, y, None

def Ko_Goldoni():
    """Ko dependence of ik1 in HL-1 cell.

    Data reported as single points at -150mV from figure 2 in Goldoni 2010
    and normalised to maximum value at 100mM external Ko. This was done as
    values are given in nA, but model output is pA/pF."""
    x = [100, 40, 20]
    y = np.asarray([-2.86764705882353,
                    -0.41911764705882293,
                    -0.1544117647058818])
    y = y / y[0]
    return x, y.tolist(), None

