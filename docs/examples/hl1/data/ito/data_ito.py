import numpy as np
### Digitised data for HL-1 i_to channel.


# I-V curves

def Peak_Yang():
    """Data points for peak outward rapidly-inactivating
    4-AP sensitive current in HL-1 cells.

    Figure 8B from 9 cells as mean \pm SEM in Yang 2005.
    """
    x = [-60, -40, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = np.asarray([2.0137931034482506,
         2.5855799373040327,
         3.1623824451410485,
         3.791849529780535,
         5.803134796238226,
         7.127272727272697,
         11.897178683385533,
         23.563636363636306,
         38.67836990595609,
         67.58620689655169,
         104.76990595611282])
    sem = np.asarray([-2.811285266457702,
           -1.549843260188112,
           -0.28589341692790526,
           -0.3435736677116097,
           1.6677115987460525,
           4.368652037617494,
           8.446394984325963,
           17.356739811912178,
           29.02319749216298,
           47.58620689655169,
           70.28714733542319])
    sem = np.abs(y-sem)
    N = 9
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()


def SS_Yang():
    """Data points for steady-state outward rapidly-inactivating
    4-AP sensitive current in HL-1 cells.

    Figure 8C from 9 cells as mean \pm SEM in Yang 2005.
    """
    x = [-60, -40, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = np.asarray([0.1769911504424755,
                    1.327433628318584,
                    2.8318584070796433,
                    4.86725663716814,
                    6.814159292035395,
                    8.141592920353983,
                    9.292035398230084,
                    10.265486725663717,
                    11.150442477876103,
                    12.920353982300883,
                    15.221238938053094])
    sem = np.asarray([-0.3539823008849545,
                      0.7964601769911468,
                      2.389380530973451,
                      4.424778761061948,
                      6.194690265486724,
                      6.902654867256636,
                      7.610619469026549,
                      8.23008849557522,
                      9.02654867256637,
                      10.442477876106192,
                      12.212389380530972])
    sem = np.abs(y-sem)
    N = 9
    sd = np.sqrt(N)*sem
    return x, y.tolist(), sd.tolist()


def TauInact_Yang():
    """Time constants for inactivation of iTo in HL1 from Yang 2005.

    Time constants obtained from single exponential fit to normalised
    current traces in Figure 8A.
    """
    x = [20, 30, 40, 50, 60]
    y = [11.5099, 12.1994, 14.1953, 16.8687, 16.4331]
    return x, y, None


def IV_Kao():
    """Data points in IV curve for i_to in HL-1.

    Data from figure 6 in Kao 2016. Reported as mean \pm SEM for
    n=15 cells.
    """
    x = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = [0.0, 0.27223230490018047, 0.5626134301270405, 0.7622504537205081,
         1.0526315789473681, 1.7059891107078045, 2.2323049001814876,
         2.940108892921959, 3.7749546279491835, 4.519056261343012,
         5.372050816696914]
    N = 15
    errs = [0.07259528130671455, 0.3992740471869318, 0.7622504537205081,
            0.9981851179673313, 1.3611615245009077, 1.978221415607985,
            2.649727767695099, 3.3030852994555353, 4.192377495462795,
            4.918330308529946, 5.8439201451905625]

    return x, y, errs, N

def IV_Lu():
    """Data points in IV curve for i_to in HL-1.

    Recorded at 35C.

    Data from figure 3A in Lu 2016.
    Error bars are SEM for n=14 cells.
    """
    x = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = np.asarray([0.03659423788829841, 0.9650303127850197, 1.58526959600837,
         2.513705670905088, 3.724710553141259, 4.601830570309563,
         5.812775363485162, 7.357785288910344, 8.645794302269431,
         9.625666613015717, 11.196244433714252])
    N = 14
    sem = np.asarray([0.01823021905769906, 1.1462287650876135, 1.8902551931321323,
            2.7879063899847143, 4.0438756855575875, 4.967091097640507,
            6.24862460844372, 8.016446967510184, 9.40032714326951,
            10.553919839208014, 12.270523963779041])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()


# Activation rate constant

def ActTau_Xu():
    """Activation time constant of i_to_f in mouse ventricular cells.

    Using for HL-1 as i_to in mouse atrial cells (which HL-1 are derived
    from) is similar to i_to_f in ventricular cells [Xu 1999], and no
    existing data for HL-1.

    Errors are sem for n=4 cells (see table iii and p669 Xu 1999).

    Data from figure 8B in Xu 1999.
    """
    x = [0, 10, 20, 30, 40, 50, 60]
    y = np.asarray([11.633340923253925, 8.67159412300466, 5.954232778869674,
         4.052088826039039, 3.1280259814965916, 2.4485283543454415,
         1.9322539667321905])
    N = 4
    sem = np.asarray([12.505602626942945, 9.272287669427037, 6.609631724650356,
            4.598438454993236, 3.647400021571613, 2.940570209604168,
            2.4783085459535243])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()


# Inactivation.

def Inact_Xu():
    """Inactivation of ito_f in mouse ventricular cells.

    Using for HL-1 as i_to in mouse atrial cells (which HL-1 are derived
    from) is similar to i_to_f in ventricular cells [Xu 1999], and no
    existing data for HL-1.

    Data from figure 9 in Xu 1999. Reported as mean \pm SEM in
    n=7 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]
    y = np.asarray([1.0064935064935066, 0.9653679653679654, 0.9718614718614719,
         0.9848484848484849, 1.0216450216450217, 1.0151515151515151,
         0.9848484848484849, 0.8095238095238095, 0.24025974025974017,
         0.032467532467532534])
    N = 7
    sem = np.asarray([1.0173160173160174, 1.0541125541125542, 1.0389610389610389,
            1.0043290043290043, 1.0454545454545454, 1.0411255411255411,
            0.9978354978354979, 0.8484848484848486, 0.28787878787878785,
            0.04761904761904767])
    sem = np.abs(y-sem)
    sd = np.sqrt(N)*sem

    return x, y.tolist(), sd.tolist()
