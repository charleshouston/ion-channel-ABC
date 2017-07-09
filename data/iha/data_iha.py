'''
    Author: Charles Houston
    Date: 29/6/2017

    Data points extracted from Sartiana et al, 2002 paper on I_f channel
    in HL-1 myocytes.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
    I-V curve

    Returns x, y data from points in figure 5B
'''
def IV_Sartiana5B():
    x = [-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0]
    y = [-4.638676,-3.479303,-3.489089,-2.615102,-3.094392,-2.008659,-1.954012,-1.328569,-1.172639,-0.602457,-0.188768,0.5010999937853455,0.868767219770886,1.6691152404035385,2.147237586228327]
    return x,y

'''
Activation curve

Returns x,y data from activation points in figure 4B
'''
def Inact_Sartiana4B():
    x = [-130,-120,-110,-100,-90,-80,-70,-60]
    y = [1.00625,0.9787500000000001,0.925,0.8737500000000001,0.66125,0.4437500000000001,0.1787500000000002,0.04875000000000007]
    return x,y
