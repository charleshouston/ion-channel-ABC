'''
    Author: Charles Houston
    Date: 2/7/2017

    Data points extracted from Kao (2016) for ito in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

# PLOT DATA ACCESSORS

'''
I-V curve

Returns x, y data from points in Figure 6 from Kao et al., 2016.
'''
def IV_KaoFig6():
    x = [-115,-105,-95,-85,-75,-65,-55,-45,-35,-25,-15,-5,5,15,25,35,45]
    y = [-1.682923,-1.191313,-0.781899,-0.427523,0.036762,0.418573,0.937841,1.319653,1.921395,2.440718,2.877512,3.231832,3.696174,4.105476,4.459796,4.869099,5.168437]
    return x, y
