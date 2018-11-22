### Digitised data for HL-1 i_to channel.


# I-V curves

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

    Data from figure 3A in Lu 2016.
    """
    x = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = [0.03659423788829841, 0.9650303127850197, 1.58526959600837,
         2.513705670905088, 3.724710553141259, 4.601830570309563,
         5.812775363485162, 7.357785288910344, 8.645794302269431,
         9.625666613015717, 11.196244433714252]
    N = 14
    errs = [0.01823021905769906, 1.1462287650876135, 1.8902551931321323,
            2.7879063899847143, 4.0438756855575875, 4.967091097640507,
            6.24862460844372, 8.016446967510184, 9.40032714326951,
            10.553919839208014, 12.270523963779041] 
    return x, y, errs, N 

# Activation rate constant

def ActTau_Xu():
    """Activation time constant of ito in mouse ventricular cells.

    Data from figure 8B in Xu 1999.
    """
    x = [0, 10, 20, 30, 40, 50, 60]
    y = [11.633340923253925, 8.67159412300466, 5.954232778869674,
         4.052088826039039, 3.1280259814965916, 2.4485283543454415,
         1.9322539667321905]
    errs = [12.505602626942945, 9.272287669427037, 6.609631724650356,
            4.598438454993236, 3.647400021571613, 2.940570209604168,
            2.4783085459535243] 
    return x, y, errs, None

# Inactivation.

def Inact_Xu():
    """Inactivation of ito in mouse ventricular cells.

    Data from figure 9 in Xu 1999. Reported as mean \pm SEM in
    n=7 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]
    y = [1.0064935064935066, 0.9653679653679654, 0.9718614718614719,
         0.9848484848484849, 1.0216450216450217, 1.0151515151515151,
         0.9848484848484849, 0.8095238095238095, 0.24025974025974017,
         0.032467532467532534]
    N = 7
    errs = [1.0173160173160174, 1.0541125541125542, 1.0389610389610389,
            1.0043290043290043, 1.0454545454545454, 1.0411255411255411,
            0.9978354978354979, 0.8484848484848486, 0.28787878787878785,
            0.04761904761904767]
    return x, y, errs, N