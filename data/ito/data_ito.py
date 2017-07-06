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
    x = [-40,-30,-20,-10,0,10,20,30,40,50,60]
    y = [0.056990,0.296790,0.609235,0.812738,1.107084,1.727961,2.312640,3.024312,3.863076,4.592847,5.431610]
    return x, y

'''
I-V curve

Returns x, y data from points in Figure 6 from Lu et al., 2011.
'''
def IV_LuFig6():
    x = [-40,-30,-20,-10,0,10,20,30,40,50,60]
    y = [-0.0062509,0.670831,1.326172,1.959688,2.723971,3.618953,4.492178,5.496153,6.260419,7.656664,8.464530]
    return x, y
