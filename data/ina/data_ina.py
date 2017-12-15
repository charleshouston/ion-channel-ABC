'''
    Author: Charles Houston
    Date: 5/6/2017

    Data points extracted from publications for fast sodium channel in HL-1.

    Data was digitised from graphs in the publication using
    [http://plotdigitizer.sourceforge.net]
'''

import math

# PLOT DATA ACCESSORS

'''
    I-V curve

    Returns x, y data from points in figure 6 from Dias et al, 2014.
'''
def IV_Dias():
    x = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40]
    y  = [0.0,0.0,0.0,0.,-4.299065,-47.289720,-81.495327,-80.934579,-76.261682,-58.504673,-39.813084,-25.607477,-12.149533,24.299065,49.719626]
    return x,y


'''
I-V curve

Returns x,y,sem,sd data from points in figure 4A from Nakajima 2009.

Data recorded as mean \pm SEM for N=17 cells.
'''
def IV_Nakajima():
    x = [-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
    y  = [-0.7432320780058248,
          -0.9827759506897493,
          -1.4767730657267109,
          -4.2606008724466164,
          -20.02204101816085,
          -81.07566132372837,
          -140.0936556904715,
          -160.1805524666931,
          -159.40228336996478,
          -144.88353918617202,
          -124.00346394355311,
          -99.30659006563847,
          -74.60946769822938,
          -49.40343884611417,
          -24.19790697298795,
          -3.5720364832275777]
    
    N = 17
    
    err_bars = [-2.7791065063246805,
                -3.2728551318671606,
                -3.512399004551085,
                -5.278413841858807,
                -26.891781582704127,
                -90.23597804843804,
                -154.59773899408964,
                -175.95665349258195,
                -174.1605714264415,
                -158.11510778853045,
                -135.45361135994568,
                -108.97556478555977,
                -81.47945675226715,
                -54.74720542502263,
                -26.996644149376976,
                -3.5720364832275777]
    
    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))]
    
    return x,y,sem,sd

'''
Activation curve

Returns x,y data from activation points in figure 5B from Fukuda et al, 2005.
'''
def Act_Fukuda():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10]
    y = [0.009682,0.011749,0.015352,0.075660,0.318340,0.649146,0.817498,0.929147,0.986389,0.988456]
    return x,y

'''
Activation curve

Returns x,y,sem,sd data from activation points in figure 4B from Nakajima 2009.

Data reported as mean \pm SEM for N=17 cells.
'''
def Act_Nakajima():
    x = [-80,-70,-60,-50,-40,-30,-20,-10,0,10]
    y = [0.004354136429608246,
         0.007256894049346929,
         0.020319303338171446,
         0.08272859216255446,
         0.3367198838896952,
         0.6357039187227868,
         0.8055152394775037,
         0.9332365747460087,
         0.9738751814223513,
         0.9869375907111757]
    
    N = 17
    
    err_bars = [0.010159651669085612,
                0.018867924528301883,
                0.029027576197387495,
                0.158200290275762,
                0.4049346879535559,
                0.679245283018868,
                0.8476052249637156,
                0.941944847605225,
                0.988388969521045,
                1.0101596516690856]

    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))] 
    
    return x,y,sem,sd


'''
Inactivation curve

Returns x,y data from inactivation points in figure 5C from from Fukuda et al, 2005.
'''
def Inact_Fukuda():
    x = [-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30]
    y = [0.988466,0.991018,0.987681,0.989054,0.942135,0.733851,0.366558,0.112337,0.026550,0.018500,0.019875]
    return x,y

'''
Inactivation curve

Returns x,y,sem,sd data from inactivation points in figure 4D from Nakajima 2009.

Data reported as mean \pm SEM for N=19 cells.
'''
def Inact_Nakajima():
    x = [-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30]
    y = [0.979968115790937,
         0.9841111082151793,
         0.9839927370030581,
         0.9881346434528775,
         0.9439821813316653,
         0.7293751737559078,
         0.34573407527105926,
         0.10129643626633311,
         0.02589506011954401,
         0.01583133514039492,
         0.01997432756463713]
    
    N = 19
    
    err_bars = [0.9813885703363914,
                0.9798486586043926,
                0.9768893783013624,
                0.9810323707256047,
                0.9368799086043926,
                0.7577831786905755,
                0.38124435293299974,
                0.11692143626633311,
                0.04009851959966637,
                0.028615426049485748,
                0.02420226300687589]
    
    sem = [abs(err_bars[i] - y[i]) for i in range(len(y))]
    sd = [sem[i] * math.sqrt(N) for i in range(len(y))] 
    
    return x,y,sem,sd


'''
Recovery curve

Returns x,y data from recovery points in figure 4F from Zhang et al, 2013
'''
def Recovery_Zhang():
    x = [10,20,30,40,50,60,70,80]
    y = [0.706855,0.890526,0.930046,0.941333,0.952623,0.968956,0.982263,0.986496]
    return x,y
