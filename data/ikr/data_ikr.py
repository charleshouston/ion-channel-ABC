'''
    Author: Charles Houston
    Date: 13/6/2017

    Data points extracted from Toyoda for iKr in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
I-V curve

Returns x, y data from points in Figure 1E from Toyoda et al., 2010.
'''
def IV_Toyoda1E():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40]
    y = [0.079545,0.011364,0.181818,0.318182,1.102273,4.511364,9.420455,14.056818,16.272727,15.318182,11.50000,7.034091,4.136364]
    return x, y

'''
Activation curve

Returns x, y data from activation points in Figure 2B in Toyoda et al., 2010.
'''
def Activation_Toyoda2B():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50]
    y = [-0.001961,-0.001881,0.003102,0.008088,0.061110,0.242565,0.518136,0.772139,0.931044,0.986027,1.005718,1.000897,0.996078,0.998120]
    return x, y
