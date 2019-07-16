### NRVM DATA ###

# Lee 1999 data for rat ventricular myocytes

def Trace_Lee():
    """Current trace from rat ventricular myocytes.

    Data from Figure 1A in Lee 1999. Stimulated from HP of
    -80mV to step of -30mV, current in pA.
    """
    x = [0.3916111337244619, 0.6079178929377456, 0.7402166477353855,
         1.0495672799265208, 1.2590545156966257, 1.5786770550740457,
         2.206840408233722, 2.6077431426495146, 3.095893155116457,
         3.8193167263730157, 4.574152727489713, 5.651509565447181,
         8.334310087993165, 9.50181250146513, 10.552701064058772,
         13.062541423277166, 15.923928215991355, 18.931124944324985]
    y = [9.738971754998545, -180.36734370041245, -427.5864804823866,
         -704.6017137416694, -786.0123770231675, -761.381307521422,
         -535.4887337353222, -410.26658203744853, -279.5620437956205,
         -175.90447476991426, -107.55712472231039, -60.7743573468739,
         -7.684068549666733, -1.614566804189053, 1.674071723262557,
         19.34306569343073, 18.18073627419875, 19.81513805141236]
    return x, y, None, None

def IV_Lee():
    """IV curve for rat ventricular myocytes.

    Data from Figure 3A in Lee 1999.
    """
    x = [-80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20,
         -15, -10, -5, 0, 5, 10, 15, 20, 25]
    y = [1.7477685466693593E-4, 4.430390036906928E-4, 7.031719966832384E-4,
         -0.0017071227665144129, 0.0012478254507617415, -0.046697936820199404,
         -0.14552815126733532, -0.4666948477409075, -0.7932186580389223,
         -0.924199684588746, -0.9641179052790739, -0.9370640740078365,
         -0.8725023168094687, -0.7918773472938038, -0.6844749378119565,
         -0.5797429560862991, -0.4696457313801682, -0.35151283593737304,
         -0.23875331263108246, -0.12597753101273024, 0.008218576747362039,
         0.10491488773635516]
    return x, y, None, None

def Act_Lee():
    """Activation of i_Na in RVMs.

    Data from Figure 3B in Lee 1999.
    """
    x = [-80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20,
         -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
    y = [0.0021594972337868334, 0.0022447018865481283, 0.004316056376099109,
         0.00240923500912249, 0.002494439661884229, 0.006560758262647681,
         0.006645962915408976, 0.10035345240438653, 0.29963832093948417,
         0.5527078920075099, 0.682271849851773, 0.8018756775973461,
         0.917495453303443, 0.9912826825950398, 0.9993330532352795,
         0.9994211959795156, 0.9995005244493282, 0.9995798529191406,
         0.9996679956633769, 0.9997473241331896, 0.9998295906944765,
         0.9999177334387128, 0.9999970619085254, 1.0000822665612872]
    return x, y, None, None

def Inact_Lee():
    """Inactivation of i_Na in RVMs.

    Data from Figure 4B in Lee 1999.
    """
    x = [-160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50,
         -40, -30, -20, -10]
    y = [1.000270143773029, 1.0005340051327316, 0.9988283299146534,
         0.9853933890164568, 0.9876205045406141, 0.9722160270646365,
         0.9372481144907006, 0.7887758403513123, 0.5150636565530284,
         0.2746199925239279, 0.08505759402416846, 0.0070425853387319215,
         0.003398785609503907, 0.005613336307008776, 0.0019569717511283002,
         0.004184087275286075]
    return x, y, None, None

def Rec_Lee1():
    """Recovery of i_Na in RVMs for a pulse train.

    Data From Figure 5A in Lee 1999.
    """
    x = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
         2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800]
    y = [1.0012345679012347, 0.9296296296296296, 0.9148148148148147,
         0.9098765432098765, 0.9037037037037037, 0.9024691358024691,
         0.8987654320987655, 0.8950617283950617, 0.8913580246913579,
         0.8876543209876543, 0.8839506172839506, 0.8814814814814815,
         0.8802469135802469, 0.874074074074074, 0.8753086419753087,
         0.8703703703703703, 0.8691358024691358, 0.8666666666666667,
         0.8641975308641975, 0.865432098765432]
    return x, y, None, None

def Rec_Lee2():
    """Recovery of i_Na in RVMs for two-pulse protocol.

    Data from Figure 5B in Lee 1999.
    """
    x = [0, 1, 10, 15, 20, 25, 30, 35, 50, 75, 100, 250, 500, 1000, 2000]
    y = [0.0024049047968568438, 0.08937437934458803, 0.19465049004792556,
         0.2930529769871768, 0.3731617805794225, 0.444108630888131,
         0.5036181512024527, 0.5928802728725011, 0.6569794050343252,
         0.7645913388886492, 0.828725011873408, 0.9434825784724321,
         0.9692241267648201, 0.9886706100772851, 1.0028971115236822]
    return x, y, None, None
