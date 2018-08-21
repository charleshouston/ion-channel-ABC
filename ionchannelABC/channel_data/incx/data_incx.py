'''
    Author: Charles Houston
    Date: 2/7/2017

    Data points extracted from Lu et al (2016) for incx in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
I-V curve

Returns x, y data from points in Figure 2 from Lu et al., 2016.
'''
def IV_Lu2016Fig2():
    x = [-100,-80,-60,-40,-20,0,20,40,60,80,100]
    y = [-1.355072,-0.826087,-0.340580,0.0,0.268116,0.637681,0.978261,1.485507,1.862319,2.311594,2.782609]
    return x, y
