'''
    Author: Charles Houston
    Date: 13/6/2017

    Data points extracted from publications for fast sodium channel in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
I-V curve

Returns x, y data from points in Figure 2B from Maharani et al, 2015.
'''
def IV_MaharaniFig2B():
    x = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
    y = [0.0,0.0,0.0,0.647668,1.910622,2.331606,3.108808,3.821244,3.853627,4.922280,6.023316,6.994819,7.998705]
    return x, y

'''
Inactivation curve

Returns x, y data from inactivation points in Figure 9C in Xu et al., 1999.
'''
def Inact_XuFig9C():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10]
    y = [1.0,1.0,0.920000,0.823546,0.724417,0.698866,0.674651,0.595587,0.354649,0.124415]
    return x, y

'''
Recovery curve

Returns x,y data from recovery points in Figure 10C in Xu et al., 1999.
'''
def Recovery_XuFig10C():
    x = [5,41,68,217,309,419,509,616,716,814,1081,2087,3092,4073,5073,6076,7076,8077,9087]
    y = [0.086498,0.161238,0.211569,0.276925,0.328654,0.423096,0.450398,0.477669,0.532435,0.565829,0.714924,0.810642,0.894146,0.960905,0.948233,0.987465,0.982426,0.983493,1.0]
    return x, y
