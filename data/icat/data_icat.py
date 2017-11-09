'''
    Author: Charles Houston
    Date: 25/5/2017

    Data points extracted from Deng et al, 2009 paper on T-type calcium channel
    in HL-1 myocytes.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
    I-V curve

    Returns x, y data from points in figure 1B
'''
def IV_DengFig1B():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30]
    y = [-0.106837,-0.492639,-2.350421,-5.561476,-11.289143,-17.396676,-20.109156,-18.643111,-14.826645,-9.751873,-5.341866,-1.697428]
    return x,y

'''
Activation curve

Returns x,y data from activation points in figure 3B
'''
def Act_DengFig3B():
    x = [-80,-70,-60,-50,-40,-30,-20,-10]
    y = [0.001373,0.027600,0.003571,0.054514,0.173841,0.497494,0.846686,1.000619]
    return x,y

'''
Inactivation curve

Returns x,y data from inactivation points in figure 3B
'''
def Inact_DengFig3B():
    x = [-100,-90,-80,-70,-60,-50,-40]
    y = [1.000205,0.990302,0.992363,0.980386,0.914755,0.534569,0.172016]
    return x,y

'''
Recovery curve

Returns x,y data from recovery points in figure 4B
'''
def Recovery_DengFig4B():
    # x = [0.3,32.1,64.5,96.3,128.3,160.1,192.5,224.3,256.3,288.4,320.7]
    x = [32.1,64.5,96.3,128.3,160.1,192.5,224.3,256.3,288.4,320.7]
    # y = [0.0,0.240493,0.443266,0.577305,0.656860,0.771620,0.815969,0.894686,0.933167,0.970810,0.978277]
    y = [0.240493,0.443266,0.577305,0.656860,0.771620,0.815969,0.894686,0.933167,0.970810,0.978277]
    return x,y
