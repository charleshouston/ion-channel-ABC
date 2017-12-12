'''
    Author: Charles Houston
    Date: 25/5/2017

    Data points extracted from Deng et al, 2009 paper on T-type calcium channel
    in HL-1 myocytes.

    Data is reported for N=19 cells.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

import math

# PLOT DATA ACCESSORS

# '''
#     I-V curve

#     Returns x, y data from points in figure 1B in Deng 2009
# '''
# def IV_DengFig1B():
#     x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30]
#     y = [-0.106837,-0.492639,-2.350421,-5.561476,-11.289143,-17.396676,-20.109156,-18.643111,-14.826645,-9.751873,-5.341866,-1.697428]
#     return x,y

'''
Activation curve

Returns x,y,sem,sd data from activation points in figure 3B in Deng 2009
'''
def Act():
    x = [-80,-70,-60,-50,-40,-30,-20,-10]
    y = [0.0016477857878474111,
         0.026364572605561243,
         0.0016477857878474111,
         0.055200823892893824,
         0.2949536560247167,
         0.49845520082389283,
         0.8486096807415036,
         1.0018537590113286]
    
    N = 19
    
    err_bars = [8.238928939237056E-4,
                0.017301750772399593,
                0.026364572605561243,
                0.06343975283213177,
                0.3559217301750772,
                0.5462409886714726,
                0.8634397528321318,
                1.0018537590113286]
    
    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))]
    return x,y,sem,sd

'''
Inactivation curve

Returns x, mean and SD data from inactivation points in figure 3B in Deng 2009

Data is reported for N=19 cells.
'''
def Inact():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20]
    y = [1.000205973223481,
         0.9868863714383797,
         0.9883968417439065,
         0.9767250257466529,
         0.910676278750429,
         0.5315482320631649,
         0.1730175077239956,
         6.875652927706977E-4,
         5.492619292823964E-4]
    
    N = 19
    
    err_bars = [0.9820803295571575,
                0.9720562993477514,
                0.9768623412289735,
                0.9602471678681771,
                0.8694816340542395,
                0.4639890147614142,
                0.23233779608650862,
                0.01139817291377998,
                0.013731548232063018]

    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))]
    return x,y,sem,sd

'''
Recovery curve

Returns x,y,sem,sd data from recovery points in figure 4B in Deng 2009
'''
def Rec():
    x = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    y = [0.24120603015075381,
         0.44472361809045224,
         0.5787269681742044,
         0.6582914572864322,
         0.7738693467336684,
         0.8165829145728644,
         0.8961474036850922,
         0.9338358458961474,
         0.9706867671691792,
         0.9807370184254607]
    
    N = 19
    
    err_bars = [0.24036850921273034,
                0.4061976549413736,
                0.5326633165829147,
                0.6072026800670017,
                0.7286432160804021,
                0.7654941373534339,
                0.8634840871021776,
                0.9053601340033501,
                0.9405360134003351,
                0.9514237855946399]
    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))]   
   
    return x,y,sem,sd
