'''
    Author: Charles Houston
    Date: 3/7/2017

    Data points extracted from Goldoni (2010) for ik1 (inward rectifier)
    in HL-1 cardiomyocytes.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
I-V curve

Returns x, y data from points in Figure 3D from Goldoni et al., 2010.
'''
def IV_GoldoniFig3D():
    x = [-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30]
    y = [-8.645093,-7.674650,-7.104772,-6.413508,-5.673686,-5.043152,-4.109075,-3.405682,-2.495895,-1.768191,-1.016262,-0.616305,-0.264961,0.049976,0.255667,0.400617,0.509223,0.520670,0.410678]
    return x, y
