# Digitised data from Sun et al, 1997
# i_CaL in human atrial myocytes

import numpy as np

def inact_tauf_Sun():
    """Fast time constant of inactivation (cf Fig 4B)"""

    x = [-10, 0, 10, 20, 30]
    y = np.array([14.025313529445754,
                  10.858232676414488,
                  15.490955325666064,
                  20.12830145061551,
                  25.25342426168875])
    sem = np.array([15.976420273940917,
                    12.809339420909666,
                    16.958908859735303,
                    21.591631508986893,
                    26.71675432006012])
    sem = np.abs(y-sem)
    N = 10
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()


def inact_taus_Sun():
    """Slow time constant of inactivation (cf Fig 4B)"""

    x = [-10, 0, 10, 20, 30]
    y = np.array([514.5409115712685,
                  300.03733593727554,
                  264.86113903328635,
                  302.4727878457164,
                  478.22740457796044])
    sem = np.array([463.63766908871594,
                    275.7747206984693,
                    240.64447571728078,
                    287.92900427927293,
                    427.30118613400737])
    sem = np.abs(y-sem)
    N = 10
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()


def rel_inact_Sun():
    """Fraction of inactivation I_Ca (cf Fig 4B)"""

    x = [-10, 0, 10, 20, 30]
    y = np.array([0.7464030326527094,
                  0.831834237959852,
                  0.8688234111599322,
                  0.8694207518451511,
                  0.853051319605985])
    sem = np.array([0.7778782918353866,
                    0.8584963096981708,
                    0.8930616581947675,
                    0.893693460842595,
                    0.8821716780103959])

    sem = np.abs(y-sem)
    N = 10
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()


