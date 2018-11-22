### Digitised data for HL-1 i_Kur channel.


# I-V curves.

def IV_Maharani():
    """Data points in IV curve for i_Kur.

    Data from figure 2B in Maharani 2015. Reported as mean \pm SD with no
    n number in the publication.
    """
    x = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    y = [-0.03234152652004951, -0.03234152652004951, -0.03234152652004951,
         0.679172056921086, 1.8758085381630032, 2.3285899094437283,
         3.10478654592497, 3.783958602846056, 3.8486416558861585,
         4.98059508408797, 5.9831824062095755, 6.985769728331178,
         7.988357050452784]
    errs = [0.0, -0.03234152652004951, 0.0, 1.131953428201811,
            2.4256144890038804, 2.943078913324708, 3.654592496765847,
            4.333764553686933, 4.463130659767142, 5.5627425614489,
            6.565329883570506, 7.600258732212161, 8.764553686934025]
    return x, y, errs, None


# Activation time constants.

def ActTau_Xu():
    """Data points for activation time constants of i_Kur.

    Determined by single exponential fit to rising phase from
    depolarising step. Reported as i_K,slow in figure 8B in Xu 1999
    as mean \pm SEM.
    """
    x = [0, 10, 20, 30, 40, 50, 60]
    y = [11.001327963485771, 6.687798095181137, 4.395491176586532,
        2.80242431491628, 1.9085975101704165, 1.6365348987988568,
        1.2862624420120454]
    errs = [11.545616293310026, 7.465247997520784, 5.094975894132645,
            1.947743209648337, 1.1311476078307692, 0.8588403358374848,
            0.5093018609158833]
    return x, y, errs, None


# Inactivation curves.

def Inact_Xu():
    """Data points for inactivation curve of i_Kur.

    Extracted from figure 9C in Xu 1999. Reported as mean \pm SEM.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]
    y = [0.9989331754339306, 0.9945193518764771, 0.9138481456019831,
         0.8198019901346326, 0.7204106722600523, 0.6919165207268055,
         0.6674357571895244, 0.5881023469136856, 0.3482304060055299,
         0.11505045588434548]
    N = None
    errs = [1.0163275342101432, 1.023950863846891, 0.8763898576396381,
            0.785019294169598, 0.6816115775053567, 0.6397424767792536,
            0.6192720904442427, 0.538603894963444, 0.305426955635955,
            0.07759818950939135]
    return x, y, errs, N


def Inact_Brouillette():
    """Data points for inactivation curve of i_Kur.

    Extracted from figure 6B in Brouillette 2003. Data reported as mean
    \pm SEM for 11 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20]
    y = [0.9764089483299438, 0.9844645134604115, 0.9731239874530351,
         0.9438661197476819, 0.8489435041880666, 0.5704387990762124,
         0.24118093137085928, 0.08506428596049775, 0.05879838681879268]
    N = 11
    errs = [0.9629761125090482, 0.9993967805315225, 0.9910413291510116,
            0.9647685360725242, 0.8743235324532075, 0.6196856364827136,
            0.2904415566509255, 0.10146840853469374, 0.06626107338595699]
    return x, y, errs, N


# Recovery curves.

def Rec_Xu():
    """Data points for recovery curve in i_Kur.

    Extracted from figure 10C in Xu 1999.
    """
    x = [5, 41, 68, 217, 309, 419, 509, 616, 716, 814, 1081, 2087, 3092, 4073,
         5073, 6076, 7076, 8077, 9087]
    y = [0.086498, 0.161238, 0.211569, 0.276925, 0.328654, 0.423096, 0.450398,
         0.477669, 0.532435, 0.565829, 0.714924, 0.810642, 0.894146, 0.960905,
         0.948233, 0.987465, 0.982426, 0.983493, 1.0]
    return x, y


def Rec_Brouillette():
    """Data points for recovery curve in i_Kur.

    Extracted from figure 6D in Brouillette 2003. Data reported as mean \pm
    SEM for 
    """
    x = [50, 100, 150, 200, 250, 500, 750, 1000, 2000, 3000]
    y = [0.2854284840068727, 0.3561501165062254, 0.4293136724174452,
         0.48785510862146064, 0.5269199990585355, 0.6259797114411466,
         0.6799962341422081, 0.73032927719067, 0.8281827382493467,
         0.9077600207122177]
    N = 11
    errs = [0.3036987784498788, 0.3768623343610986, 0.4488078706427847,
            0.5146633087768022, 0.5537164309082777, 0.6430437545602183,
            0.6946065855438133, 0.7534775343046107, 0.8525372466872219,
            0.9175100619012874]
    return x, y, errs, N