'''
    Author: Charles Houston
    Date: 5/6/2017

    Data points extracted from publications for fast sodium channel in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
    I-V curve

    Returns x, y data from points in figure 6 from Dias et al, 2014.
'''
def IV_DiasFig6():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40]
    y  = [0.186916,0.373832,0.373832,0.373832,-4.299065,-47.289720,-81.495327,-80.934579,-76.261682,-58.504673,-39.813084,-25.607477,-12.149533,24.299065,49.719626]
    return x,y

'''
Activation curve

Returns x,y data from activation points in figure 5B from Fukuda et al, 2005.
'''
def Act_FukudaFig5B():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10]
    y = [0.009682,0.011749,0.015352,0.075660,0.318340,0.649146,0.817498,0.929147,0.986389,0.988456]
    return x,y

'''
Inactivation curve

Returns x,y data from inactivation points in figure 5C from from Fukuda et al, 2005.
'''
def Inact_FukudaFig5C():
    x = [-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30]
    y = [0.988466,0.991018,0.987681,0.989054,0.942135,0.733851,0.366558,0.112337,0.026550,0.018500,0.019875]
    return x,y

'''
Recovery curve

Returns x,y data from recovery points in figure 4F from Zhang et al, 2013
'''
def Recovery_ZhangFig4B():
    #x = [2,10,20,30,40,50,60,70,80]
    x = [10,20,30,40,50,60,70,80]
    #y = [0.0,0.706855,0.890526,0.930046,0.941333,0.952623,0.968956,0.982263,0.986496]
    y = [0.706855,0.890526,0.930046,0.941333,0.952623,0.968956,0.982263,0.986496]
    return x,y
