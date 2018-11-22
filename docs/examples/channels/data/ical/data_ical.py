# Digitised data for HL-1 i_CaL channel.


# IV curves.

def IV_Rao():
    """IV curve for i_CaL in HL-1.

    Data from figure 3B in Rao 2009. Reported as mean \pm SEM for
    n = 8 cells.
    """
    x = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    x = [v + 0.0001 for v in x] # avoid nan errors
    y = [-0.12103620322693764, -0.17723158329658872, -2.697690729237914,
         -10.923204855336389, -18.501404450493553, -21.583348668289275,
         -19.82415724290223, -15.557318593257945, -8.481023426025908,
         -3.911906754209012, -0.46685392673247605]
    N = 8
    errs = [-0.3373285233652652, -0.6093474947311428, -3.604837281650479,
            -13.387312378260704, -22.218788735230703, -25.300732953026426,
            -23.066354643713993, -17.97819890074406, -10.729307357653996,
            -5.554853426755042, -0.8558988656762025]
    return x, y, errs, N


# Activation curves.

def Act_Rao():
    """Activation data for i_CaL in HL-1 cells.

    Data extracted from figure 3C in Rao 2009. Reported as mean \pm SEM
    for n = 8 control cells.
    """
    x = [-50, -40, -30, -20, -10, 0, 10]
    x = [v + 0.001 for v in x] # avoid nan errors
    y = [0.0024607480881480015, 0.005599519078786974, 0.0067007664681828505,
         0.07807109825890612, 0.4131862930544542, 0.7931071414381685,
         1.0019494238100912]
    N = 8
    errs = [0.01569990101729335, 0.02189279810119027, 0.028089150078857816,
            0.13306091495951722, 0.4915951071794419, 0.8440210472128508,
            1.0192610327713947]
    return x, y, errs, N


# Inactivation curves.

def Inact_Rao():
    """Inactivation dynamics of HL-1 cells i_CaL channel.

    Data from figure 3C in Rao 2009. Reported as mean \pm SEM
    for n = 8 cells.
    """
    x = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10]
    x = [v + 0.0001 for v in x]
    y = [1.0020366598778003, 1.0021213047751814, 0.9950785038237036,
         0.9809048021295965, 0.8547173983267949, 0.7733356481121596,
         0.705194778273555, 0.5840955692714837, 0.3173777701770113,
         0.018073413037732466, 0.03852292926623213, 0.019261896494837227]
    N = 8
    errs = [1.0173098815144181, 1.0153604577043267, 1.0134075790004646,
            1.0104372340811452, 0.963677838065674, 0.849707802359347,
            0.8049928397326602, 0.6563995861037263, 0.360145036440492,
            0.10055641064176357, 0.09657032695387158, 0.03555517551724052]
    return x, y, errs, N


# Recovery

def Rec_Rao():
    """Relative recovery of HL-1 i_CaL current.

    Data from figure 3D in Rao 2009. Reported as mean \pm SEM for
    n = 8 cells.
    """
    x = [0.7789678675754743, 15.189873417721529, 32.32716650438171,
         64.26484907497567, 96.59201557935737, 128.1402142161636,
         160.07789678675755, 191.6260954235638, 223.56377799415776,
         254.72249269717628, 288.6075949367089, 320.9347614410906,
         352.87244401168454]
    y = [0.004744958481613271, 0.1779359430604981, 0.29893238434163694,
         0.4151838671411624, 0.5527876631079478, 0.6417556346381968,
         0.7413997627520759, 0.7841043890865955, 0.8825622775800712,
         0.8979833926453142, 0.9525504151838671, 0.9383155397390273,
         0.9406880189798339]
    N = 8
    errs = [0.023724792408066353, 0.20996441281138778, 0.3428232502965598,
            0.44365361803084213, 0.6346381969157769, 0.6998813760379596,
            0.8208778173190985, 0.8481613285883749, 0.9727164887307236,
            0.9442467378410438, 1.0154211150652432, 0.9857651245551601,
            0.99644128113879]
    return x, y, errs, N