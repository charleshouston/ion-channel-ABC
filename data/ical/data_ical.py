'''
    Author: Charles Houston
    Date: 27/6/2017

    Data points extracted from Dias 2014 and Rao 2009 for L-type calcium channel
    in HL-1 myocytes.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
    I-V curve

    Returns x, y data from points in Fig. 7 from Dias et al., 2014
'''
def IV_DiasFig7():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40]
    x = [v+0.0001 for v in x]
    y = [-0.035530,-0.001224,0.004548,0.010148,0.015920,0.021634,0.027292,-4.627351,-13.971229,-17.053434,-13.044918,-10.294271,-9.202124,-3.506631,7.478355]
    return x,y

'''
Activation curve

Returns x,y data from activation points in Fig 3C from Rao et al., 2009
'''
def Act_RaoFig3C():
    x = [-50,-40,-30,-20,-10,0,10]
    x = [v+0.0001 for v in x]
    y = [0.002883,0.006106,0.012382,0.081763,0.413742,0.793552,1.000000]
    return x,y

'''
Inactivation curve

Returns x,y data from inactivation points in Fig 3C from Rao et al., 2009
'''
def Inact_RaoFig3C():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10]
    x = [v+0.0001 for v in x]
    y = [1.003566,1.003732,0.997797,0.984733,0.857677,0.776422,0.704328,0.586433,0.320948,0.020864,0.039354,0.021205]
    return x,y

'''
Recovery curve

Returns x,y data from recovery points in Fig 3D from Rao et al., 2009
'''
def Recovery_RaoFig3D():
    x = [0.8,15.5,31.9,63.8,95.3,127.3,159.6,191.6,223.5,255.5,287.8,319.8,351.7]
    y = [0.001189,0.177983,0.299028,0.415374,0.553072,0.640949,0.740689,0.785861,0.882042,0.898744,0.954593,0.935708,0.941734]
    return x,y
