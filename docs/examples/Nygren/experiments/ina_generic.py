"""
@author: brm18
"""
from ionchannelABC.experiment import Experiment

import data.ina.Sakakibara1992.data_Sakakibara1992 as dataSaka
import data.ina.Schneider1994.data_Schneider1994 as dataSch
from ionchannelABC.protocol import recovery_tpreList ,manual_steptrain_linear, availability_linear,varying_test_duration_double_pulse,availability
import numpy as np
import pandas as pd
import myokit
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so

"""
NB : Nao in the nygren_Na.mmt file corresponds to Nao in the Sakakibara paper according to equation (3) in Sakakibara paper.
"""

def temperature_adjust(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


# All experiments use conditions as defined in the
# Sakakibara and Schneider paper. Data is adjusted to these conditions.
# Experimental conditions are included in experiment setup below
# for reference.
Q10 = 2.79 # Ten Tusscher paper, 2004

#######################################################################################################################
### IV curve - Sakakibara 1992
sakakibara_iv_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper figure 1B

    page 3 of the paper : The holding potential (Vh) was -140 mV. Depolarizing pulses in 10-mVsteps were applied from
     -70 to +20mV. Testpulses were applied at 0.1 Hz.
    
    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
vsteps, peaks, _ = dataSaka.IV_Sakakibara()
sakakibara_iv_max_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / sakakibara_iv_max_peak for p in peaks]
variances = [0 for p in peaks] # no error reported
sakakibara_iv_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_iv_saka - tstep # before the first pulse occurs
tpreMeasuring_iv_saka = tperiod_iv_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 20+dV # need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


sakakibara_iv_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 

# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_saka, adjust=True)
        current = d['ina.i_Na'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        index = np.argmax(np.abs(current))
        output = output+[current[index]/sakakibara_iv_max_peak]
    return output

# Experiment
sakakibara_iv = Experiment(
    dataset=sakakibara_iv_dataset,
    protocol=sakakibara_iv_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_desc)


#######################################################################################################################
### activation curve - Sakakibara 1992
sakakibara_act_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    
    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B :
    single test pulse at a frequency of 0.1Hz
    """


# DATA
vsteps_act, act, sd_act = dataSaka.Act_Sakakibara()
variances_act = [(sd_)**2 for sd_ in sd_act]
sakakibara_act_dataset = np.asarray([vsteps_act, act, variances_act])

# PROTOCOL
tperiod_act_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_act_saka - tstep # before the first pulse occurs
tpreMeasuring_act_saka = tperiod_act_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 20 + dV# need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


sakakibara_act_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 

# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_act_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_act_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_act_saka, adjust=True)
        act_gate = d['ina.m_infinity_cube']
        index = np.argmax(np.abs(act_gate))
        output = output+[np.abs(act_gate[index])]
    return output

# Experiment
sakakibara_act = Experiment(
    dataset=sakakibara_act_dataset,
    protocol=sakakibara_act_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_act_sum_stats,
    description=sakakibara_act_desc)

#######################################################################################################################
### Inactivation curve - Sakakibara 1992
sakakibara_inact_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    page 7 of the paper : 
    The voltage dependence of h, was studied using a double-pulse protocol consisting of a
    1-second conditioning pulse from holding a potential of-140 mV 0.1 Hz (inset at lower left).
    Current amplitude elicited during the test pulse was normalized to that in absence of a conditioning pulse.

    TODO : The corresponding block in summary statistics function need to be changed to normalize the conditionning pulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
vsteps_inact, inact, sd_inact = dataSaka.Inact_Sakakibara()
variances_inact = [(sd_)**2 for sd_ in sd_inact]
sakakibara_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

# PROTOCOL
tperiod_inact_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
twait = 2 
ttest = 30
tpre = tperiod_inact_saka - tstep - twait - ttest
tpreMeasuring_inact_saka = tperiod_inact_saka - tstep - twait - ttest # tperiod - ttest
tMeasuring_inact_saka = ttest 

Vhold = -140 # mV
Vtest = -20
Vlower = -140 
dV = 10
Vupper = -40 + dV # check why the +dv is required
    
sakakibara_inact_protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)


# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_saka, adjust=True):
        d = d.trim(tpreMeasuring_inact_saka,tpreMeasuring_inact_saka  + tMeasuring_inact_saka, adjust = True)
        inact_gate = d['ina.hss']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

# Experiment
sakakibara_inact = Experiment(
    dataset=sakakibara_inact_dataset,
    protocol=sakakibara_inact_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_sum_stats,
    description=sakakibara_inact_desc)

#######################################################################################################################
### IV curve : potential reversal as function of Nao - Sakakibara 1992
# TODO : need to change a bit the experiment class, to have multiple conditions for a fused protocol.
sakakibara_rever_desc ="""
    describes the protocol used to measure the reversal potential curve in the Sakakibara Paper (figure 3B)

    page 5 of the paper :

    The reversal potential was determined by the intersecting point on the current-voltage relation curve

    the IV curve was plotted every 1mV and the protocol used for the IV curve is the same than the fig 1B
    (single pulse protocol with a frequence of 0.1Hz)
       
    """

# DATA
Nao, rever, sd_rever = dataSaka.Reversal_potential_Sakakibara()
max_rever = np.max(np.abs(rever)) # normalising
rever = [rever_ / max_rever for rever_ in rever]
variances_rever = [(sd_/max_rever)**2 for sd_ in sd_rever]
sakakibara_rever_dataset = np.asarray([Nao, rever, variances_rever])

# PROTOCOL
conditions_list, tmp_protocol = [], []
tpreMeasuring_list, tMeasuring_list = [], []

tperiod_rever = 3000 # ms
tstep = 500 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_rever - tstep # before the first pulse occurs
tpreMeasuring_rever = tperiod_rever - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
Vupper = 40
dV = 1 # needs to be small to ensure a smooth curve for the detection of V such that Ina(V) = 0

protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 

for Nao in [2000,5000,10000,20000]:
    sakakibara_conditions = {'membrane.Nao': Nao,
                            'membrane.Nai': 5000,
                            'membrane.T': 290.15}

    tmp_protocol.append(protocol)
    conditions_list.append(sakakibara_conditions)
    tpreMeasuring_list.append(tpreMeasuring_rever)

sakakibara_rever_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 

# SUMMARY STATISTICS
def sakakibara_rever_sum_stats(data):
    output = []
    for i in range(4):
        nomalized_peak = []
        for d in data[i].split_periodic(tperiod_rever, adjust = True):

            d = d.trim_left(tpreMeasuring_rever, adjust = True)
            current = d['ina.i_Na']
            current = current[:-1]           
            index = np.argmax(np.abs(current))
            nomalized_peak.append(current[index])
        nomalized_peak_cut_50 = nomalized_peak[50:] 
        # the reversal potential is after -50 mV and since the protocol starts at -100mV with dV = 1mV
        # the cut was choosen to be at index 50
        index = np.argmin(np.abs(nomalized_peak_cut_50))
        output = output+ [(index - 50)/ max_rever] # shifting back the index
    return output

# Experiment
sakakibara_rever = Experiment(
    dataset=sakakibara_rever_dataset,
    protocol=sakakibara_rever_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rever_sum_stats,
    description=sakakibara_rever_desc)

#######################################################################################################################
###  Inactivation kinetics part 1 - Sakakibara 1992
sakakibara_inact_kin_1_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

# DATA
tmp_dataset = []
# 5 : Fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = dataSaka.TauF_Inactivation_Sakakibara()
th1 = [temperature_adjust(th1_,290.15,306.15,Q10) for th1_ in th1]
max_th1 = np.max(np.abs(th1)) # normalising
th1 = [th1_ / max_th1 for th1_ in th1]
variances_th1 = [(sd_/max_th1)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = dataSaka.TauS_Inactivation_Sakakibara()
th2 = [temperature_adjust(th2_,290.15,306.15,Q10) for th2_ in th2]
max_th2 = np.max(np.abs(th2)) # normalising
th2 = [th2_ / max_th2 for th2_ in th2]
variances_th2 = [(sd_/max_th2)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])

sakakibara_inact_kin_1_dataset = [dataset1,dataset2]

# PROTOCOL
tperiod_kin_1_saka = 1000 # ms
tstep = 100 # ms 
tpre = tperiod_kin_1_saka - tstep # before the first pulse occurs
tpreMeasuring_kin_1_saka = tperiod_kin_1_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -50 
Vupper = -20
dV = 10

sakakibara_inact_kin_1_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 


# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}


# SUMMARY STATISTICS
def sakakibara_inact_kin_1_sum_stats(data):

    def double_exp(t, tauh,taus,Ah,As):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)

    output = []
    ss_list =  []
    for d in data.split_periodic(tperiod_kin_1_saka, adjust = True):
        d = d.trim_left(tpreMeasuring_kin_1_saka, adjust = True)
        
        current = d['ina.i_Na'][:-1] # sometimes, the last value is nan and crashes the following,
                                                # so getting rid of the last value is perhaps the solution
        time = d['environment.time'][:-1]
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]


                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(double_exp, time, current,p0=[5,1,1,1], bounds=([0.01,0.01,0.01,0.01], [100.0,10,10,10]))

                tauh = popt[0]
                taus = popt[1]
                output = output+[tauh/max_th1]
                ss_list.append(taus/max_th2)
            except:
                output = output+[float('inf')]
                ss_list.append(float('inf'))
    for value in ss_list:
        output = output+[value]
    return output

# Experiment
sakakibara_inact_kin_1 = Experiment(
    dataset=sakakibara_inact_kin_1_dataset,
    protocol=sakakibara_inact_kin_1_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_1_sum_stats,
    description=sakakibara_inact_kin_1_desc)

#######################################################################################################################
###  Inactivation kinetics part 2 - Sakakibara 1992
sakakibara_inact_kin_2_desc = """
    describes the protocol used to measure the inactivation kinetics in the Sakakibara Paper (figure 6)
    
    the Voltage used here is -100mV and -80mV
    
    page 7 of the paper : 
    Inactivation was induced by increasing
    the conditioning pulse duration. Conditioning
    pulses were applied to -80 and -100 mV Pulse
    frequency was 0.1 Hz. Current magnitude during
    the test pulse was normalized to levels obtained in
    the absence of a conditioning pulse.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
sakakibara_inact_kin_2_dataset = []
for Vcond in [-100,-80]:
    timecourse, inact_timecourse,_ = dataSaka.Time_course_Inactivation_Sakakibara_all(Vcond)
    variances_time_course = [0 for t in timecourse]
    sakakibara_inact_kin_2_dataset.append(np.asarray([timecourse, inact_timecourse, variances_time_course]))



# PROTOCOL
conditions_list, tmp_protocol = [], []
tpreMeasuring_list, tMeasuring_list = [], []



# This is the main part of the protocol

tperiod_kin_2_saka = 10000 # ms


for Vcond in [-100,-80]:
    tstepList,_,_ = dataSaka.Time_course_Inactivation_Sakakibara_all(Vcond)
    
    twait = 2 
    ttest = 30

    tpreList = []
    for tstep in tstepList :
        tpre = tperiod_kin_2_saka - tstep - twait - ttest
        tpreList.append(tpre)
    
    Vhold = -140 # mV
    Vtest = -20        
        
    protocol = varying_test_duration_double_pulse(Vcond,Vhold, Vtest, tpreList, tstepList, twait, ttest)
    
    tmp_protocol.append(protocol)

## This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse

twait = 0
tstep = 0
ttest = 30 
tpre = tperiod_kin_2_saka - twait - ttest - tstep

tpreMeasuring_kin_2_saka = tperiod_kin_2_saka - ttest

Vhold = -140 # mV
Vtest = -20        
protocol = availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)

tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_inact_kin_2_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_inact_kin_2_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}


# SUMMARY STATISTICS
def sakakibara_inact_kin_2_sum_stats(data):
    output = []

    # retrieving the current max in absence of conditionning pulse
    dtwoProtocol,d_split_init = data.split(tperiod_kin_2_saka*23)

    d_init = d_split_init.trim_left(tpreMeasuring_kin_2_saka, adjust = True)
    current = d_init['ina.i_Na']

    current = current[:-1]
    index = np.argmax(np.abs(current))
    normalizing_peak = current[index] 

    # 6 : Fast inactivation kinetics : tau h1 part 2 
    loop = 0
    # tperiod_kin_2_saka*12 corresponds to the split between the vCond = -100 and the vcond = -80 (there is 12 data points for vCond = -100,
    # but 11 for vcond = -80 )
    for dOneProtocol in dtwoProtocol.split(tperiod_kin_2_saka*12): 
        D_split = dOneProtocol.split_periodic(tperiod_kin_2_saka, adjust = True)
        if loop == 1 :
            D_split = D_split[12:] # specific to split_periodic function
        for d in D_split:

                d = d.trim_left(tpreMeasuring_kin_2_saka, adjust = True)
            
                current = d['ina.i_Na']
                current = current[:-1]

                index = np.argmax(np.abs(current))
                try :
                    output = output+ [current[index] / normalizing_peak]
                except : 
                    output = output+ [float('inf')]
        loop += 1
    return output

# Experiment
sakakibara_inact_kin_2 = Experiment(
    dataset=sakakibara_inact_kin_2_dataset,
    protocol=sakakibara_inact_kin_2_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_2_sum_stats,
    description=sakakibara_inact_kin_2_desc)

#######################################################################################################################
###  recovery curves - Sakakibara 1992
sakakibara_recov_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)
    
    the Vhold used here is -140mV ,-120mV and -100mV
    
    page 8 of the paper : 
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse. 


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
sakakibara_recov_dataset = []
for Vhold in [-140,-120,-100]:
    time,recov,_ = dataSaka.Recovery_Sakakibara_all(Vhold)
    variances_time = [0 for t in time]
    sakakibara_recov_dataset.append(np.asarray([time, recov, variances_time]))


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov = []

tperiod_recov = 10000 # ms
tstep1 = 1000
tstep2 = 1000

tMeasuring1_recov = tstep1 
tpreMeasuring2_recov = tperiod_recov - tstep2 

Vstep1 = -20
Vstep2 = -20  

Split_list_recov = [] # for the summary statistics function

for Vhold in [-140,-120,-100] :
    twaitList,_,_ = dataSaka.Recovery_Sakakibara_all(Vhold)

    Split_list_recov.append(len(twaitList))


    tpreList = []
    for twait in twaitList:
        tpre = tperiod_recov - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov.append(tpre)

    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_recov_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_recov_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}


# SUMMARY STATISTICS
def sakakibara_recov_sum_stats(data):
    output = []
    loop = 0
    sub_loop = 0

    # spliting back the protocols
    dProtocolOne,dProtocoltwothree = data.split(tperiod_recov*Split_list_recov[0])
    dProtocolTwo,dProtocolThree  = dProtocoltwothree.split(tperiod_recov*(Split_list_recov[1]+Split_list_recov[0]))
    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree]
   
    for dOneProtocol in dProtocols:
        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)
        if loop == 1 :
            d_split = d_split[Split_list_recov[0]:]    # specific to split_periodic function
        if loop == 2:
            d_split = d_split[Split_list_recov[0]+Split_list_recov[1]:]  


        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov[sub_loop], tpreMeasuringList1_recov[sub_loop]+tMeasuring1_recov, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov, adjust = True)
            
            current_cond = dcond['ina.i_Na'][:-1]
            current_test = dtest['ina.i_Na'][:-1]



            index_cond = np.argmax(np.abs(current_cond))
            index_test = np.argmax(np.abs(current_test))
            try :
                output = output + [current_test[index_test] / current_cond[index_cond]]  
                sub_loop += 1
            except :
                output = output + [float('inf')]  
                sub_loop += 1
        loop += 1
    return output

# Experiment
sakakibara_recov = Experiment(
    dataset=sakakibara_recov_dataset,
    protocol=sakakibara_recov_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_recov_sum_stats,
    description=sakakibara_recov_desc)

#######################################################################################################################
###  kinetics of recovery curves - Sakakibara 1992
sakakibara_recov_kin_desc = """
    describes the protocol used to measure the fast time constant from the Recovery of I_na in the Sakakibara Paper (figure 9)
    
    the Vhold used here is -140mV ,-120mV -110mV, -100mV and -90mV
    
    The protocol is not decribed but was assumed to be the same as for the recovery protocol (fig 8A)
    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
vsteps_th_depol, th_depol, _ = dataSaka.TauF_Inactivation_Sakakibara_Depol()
th_depol = [temperature_adjust(th_, 290.15, 306.15, Q10) for th_ in th_depol]
max_th_depol = np.max(np.abs(th_depol))
th_depol = [th_ / max_th_depol for th_ in th_depol]
variances = [0.]*len(th_depol)
sakakibara_recov_kin_dataset = np.asarray([vsteps_th_depol, th_depol, variances])


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov_kin = []

tperiod_recov_kin = 10000 # ms
tstep1 = 1000
tstep2 = 1000

tMeasuring1_recov_kin = tstep1 
tpreMeasuring2_recov_kin = tperiod_recov_kin - tstep2 

Vstep1 = -20
Vstep2 = -20  

twaitList = [2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
Split_list_recov_kin = [len(twaitList)]

for Vhold in [-140, -120, -110, -100, -90]:

    tpreList = []
    for twait in twaitList:
        tpre = tperiod_recov_kin - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov_kin.append(tpre)

    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_recov_kin_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_recov_kin_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Nao': 5000,
                        'membrane.Nai': 5000,
                        'membrane.T': 290.15}


# SUMMARY STATISTICS
def sakakibara_recov_kin_sum_stats(data):
    def simple_exp(t, tauh):
        return np.exp(-t/tauh)
    output = []
    loop = 0
    sub_loop = 0
    

    # spliting back the 5 protocols
    dProtocolOne,dProtocoltwofive = data.split(tperiod_recov_kin*Split_list_recov_kin[0])
    dProtocolTwo,dProtocolThreefive  = dProtocoltwofive.split(tperiod_recov_kin*2*Split_list_recov_kin[0])
    dProtocolThree,dProtocolfourfive = dProtocolThreefive.split(tperiod_recov_kin*3*Split_list_recov_kin[0])
    dProtocolFour,dProtocolFive  = dProtocolfourfive.split(tperiod_recov_kin*4*Split_list_recov_kin[0])
    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive]
   
    for dOneProtocol in dProtocols:
        rec = []

        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)

        d_split = d_split[loop*Split_list_recov_kin[0]:]    # specific to split_periodic function 
        #( if the time begins at t0 >0 it will create empty arrays from 0 to t0 : here we are getting rid of them)
 
        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov_kin[sub_loop], tpreMeasuringList1_recov_kin[sub_loop]+tMeasuring1_recov_kin, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov_kin, adjust = True)
            
            current_cond = dcond['ina.i_Na'][:-1]
            current_test = dtest['ina.i_Na'][:-1]

            index_cond = np.argmax(np.abs(current_cond))
            index_test = np.argmax(np.abs(current_test))
            try :
                rec.append(current_test[index_test] / current_cond[index_cond])
                sub_loop += 1
            except :
                rec.append(float('inf'))
                sub_loop += 1

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                # Fit simple exponential to recovery curve
                popt, _ = so.curve_fit(simple_exp, twaitList, 1.-np.asarray(rec),p0=[5], bounds=([0.1], [50.0]))
                tauh = popt[0]

                output = output + [tauh/max_th_depol]
            except:
                output = output + [float('inf')]
        loop += 1

    return output

# Experiment
sakakibara_recov_kin = Experiment(
    dataset=sakakibara_recov_kin_dataset,
    protocol=sakakibara_recov_kin_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_recov_kin_sum_stats,
    description=sakakibara_recov_kin_desc)



#######################################################################################################################
### IV curve - Schneider 1994
schneider_iv_desc = """
    describes the protocol used to measure the IV peak-current curve in the Schneider Paper figure 1B
    
    page 2 of the paper :

    depolarizing the membrane from -135 mV to test
    potentials varying from -85 to +55 mV in 10-mV.

    simple pulse protocol at the frequency 1Hz (the frequency of the pulses is not given)
    """

# DATA
vsteps, peaks, sd = dataSch.IV_Schneider()
cm_mean = 89 # pF
cm_sd = 26.7
# convert nA to pA/pF
peaks = np.array(peaks)
peaks = peaks*1000/cm_mean
sd = [(cm_sd/cm_mean)*p for p in peaks]
schneider_iv_max_peak = np.max(np.abs(peaks)) # normalising
peaks = [p / schneider_iv_max_peak for p in peaks]
variances = [(sd_ / schneider_iv_max_peak)**2 for sd_ in sd]
schneider_iv_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_sch = 1000 # ms (not precised in the paper, 1s seems enough)
tstep = 12 # ms 
tpre = tperiod_iv_sch - tstep # before the first pulse occurs
tpreMeasuring_iv_sch = tperiod_iv_sch - tstep # before the measurement

Vhold = -135 # mV
Vlower = -85.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
Vupper = 55
dV = 10

schneider_iv_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 

# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}

# SUMMARY STATISTICS
def schneider_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_sch, adjust=True):
        d = d.trim_left(tperiod_iv_sch, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        output = output+[current[index]/schneider_iv_max_peak]
    return output

schneider_iv = Experiment(
    dataset=schneider_iv_dataset,
    protocol=schneider_iv_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_iv_sum_stats,
    description=schneider_iv_desc
)



#######################################################################################################################
### Activation kinetics Schneider 1994
schneider_taum_desc =     """
    describes the protocol used to measure the activation curve in the Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)

    Temperature adjusted from 297 to 310 using Q10 of 3."""

# DATA
vsteps_tm, tm, sd_tm = dataSch.TauM_Activation_Schneider()
tm = [temperature_adjust(tm_, 297, 310, Q10) for tm_ in tm]
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
schneider_taum_dataset = np.asarray([vsteps_tm, tm, variances_tm])

# PROTOCOL
tperiod_act_kin_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_act_kin_sch - tstep # before the first pulse occurs
tpreMeasuring_act_kin_sch = tperiod_act_kin_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_taum_protocol = manual_steptrain_linear(vsteps_tm,Vhold,tpre,tstep) 

# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_taum_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_act_kin_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_act_kin_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]


        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[0]/max_tm]
            except:
                output = output+[float('inf')]
    return output

schneider_taum = Experiment(
    dataset=schneider_taum_dataset,
    protocol=schneider_taum_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taum_sum_stats,
    description=schneider_taum_desc
)

#######################################################################################################################
### Fast Inactivation kinetics Schneider 1994
schneider_tauf_desc =     """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)

    Temperature adjusted from 297 to 310 using Q10 of 3."""

# DATA
vsteps_tf, tf, sd_tf = dataSch.TauF_Inactivation_Schneider()
tf = [temperature_adjust(tf_, 297, 310, Q10) for tf_ in tf]
max_tf = np.max(np.abs(tf)) # normalising
tf = [tf_ / max_tf for tf_ in tf]
variances_tf = [(sd_/max_tf)**2 for sd_ in sd_tf]
schneider_tauf_dataset = np.asarray([vsteps_tf, tf, variances_tf])

# PROTOCOL
tperiod_inact_kin_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_inact_kin_sch - tstep # before the first pulse occurs
tpreMeasuring_inact_kin_sch = tperiod_inact_kin_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_tauf_protocol = manual_steptrain_linear(vsteps_tf,Vhold,tpre,tstep) 


# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_tauf_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_inact_kin_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_kin_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[1]/max_tf]
            except:
                output = output+[float('inf')]
    return output

schneider_tauf = Experiment(
    dataset=schneider_tauf_dataset,
    protocol=schneider_tauf_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_tauf_sum_stats,
    description=schneider_tauf_desc
)
    

#######################################################################################################################
### Slow Inactivation kinetics Schneider 1994
schneider_taus_desc =     """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 5B)
    
    
    page 4 of the paper :
    Slow inactivation was investigated in the potential
    range from -115 to -55 mV with prepulses of variable
    duration up to 1024 ms. The fraction of available Na
    channels was again determined with test pulses to
    -20 mV

    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    
    Temperature adjusted from 297 to 310 using Q10 of 3."""

# DATA
vsteps_ts, ts, sd_ts = dataSch.TauS_Inactivation_Schneider()
ts = [temperature_adjust(ts_, 297, 310, Q10) for ts_ in ts]
max_ts = np.max(np.abs(ts)) # normalising
ts = [ts_ / max_ts for ts_ in ts]
variances_ts = [(sd_/max_ts)**2 for sd_ in sd_ts]
schneider_taus_dataset = np.asarray([vsteps_ts, ts, variances_ts])

# PROTOCOL
tperiod_inact_kin_slow_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_inact_kin_slow_sch - tstep # before the first pulse occurs
tpreMeasuring_inact_kin_slow_sch = tperiod_inact_kin_slow_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_taus_protocol = manual_steptrain_linear(vsteps_ts,Vhold,tpre,tstep) 


# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_taus_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_inact_kin_slow_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_kin_slow_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]


        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[1]/max_ts]
            except:
                output = output+[float('inf')]
    return output

schneider_taus = Experiment(
    dataset=schneider_taus_dataset,
    protocol=schneider_taus_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taus_sum_stats,
    description=schneider_taus_desc
)

#######################################################################################################################
### Inactivation Schneider 1994
schneider_inact_desc = """
    describes the protocol used to measure the activation curve in the Schneider Paper (figure 4)
    

    page 4 of the paper :

    Starting from a conditioning
    potential of -135 mV, the cells were depolarized
    with prepulses varying from -135 mV to +5 mV in
    10-mV steps with a subsequent test pulse to -20 mV.
    To test for fast and intermediate inactivation, h infinity curves
    were determined with five different prepulse durations
    (increasing from 32 ms to 512ms by factors of 2,
    Fig. 4). 

    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
    """

# DATA
schneider_inact_dataset = []
for tprepulse in [32,64,128,256,512]:
    t_prepulse,inact,_ = dataSch.Inact_Schneider_all(tprepulse)
    variances_inact = [0 for t in t_prepulse]
    schneider_inact_dataset.append(np.asarray([t_prepulse,inact, variances_inact]))

# PROTOCOL
tmp_protocol = []
    
tperiod_inact_sch = 3000 # ms
tstep = 500 # ms (not precised in the paper, 0.5s seems enough to mesure the peak current)
twait = 2 
ttest = 30
tpreMeasuring_inact_sch = tperiod_inact_sch - ttest -twait -tstep
tMeasuring_inact_sch = tstep

Vhold = -135 # mV
Vtest = -20
Vlower = -135.001 
Vupper = 5
dV = 10

for tstep in [32,64,128,256,512]:

    tpre = tperiod_inact_sch - tstep - twait - ttest
    protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)

    tmp_protocol.append(protocol)

 
# Fuse all the protocols into one
schneider_inact_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_inact_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_inact_sum_stats(data):

    output = []
    for d in data.split_periodic(tperiod_inact_sch, adjust=True):
        d = d.trim(tpreMeasuring_inact_sch,tpreMeasuring_inact_sch  + tMeasuring_inact_sch, adjust = True)
        inact_gate = d['ina.h_inf']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

schneider_inact = Experiment(
    dataset=schneider_inact_dataset,
    protocol=schneider_inact_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_inact_sum_stats,
    description=schneider_inact_desc
)
    
#######################################################################################################################
### Reduction Schneider 1994
schneider_reduc_desc ="""
    describes the protocol used to measure the Relative_peak IV curve in the Schneider Paper (figure 5A)
    
    page 5 of the paper :

    Reduction of the Na + current peak
    amplitude elicited by pulses to -20 mV following prepotentials
    varying in time (abscissa) and magnitude. Symbols represent
    different inactivation potential values: -105 mV to -65 mV

    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
    """

# DATA
schneider_reduc_dataset = []
for Vinact in [-105,-95,-85,-75,-65]:
    tstepList,reduc,_ = dataSch.Reduction_Schneider_all(Vinact)
    variances_reduc = [0 for t in tstepList]
    schneider_reduc_dataset.append(np.asarray([tstepList,reduc, variances_reduc]))

# PROTOCOL
tmp_protocol = []

## This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse
tperiod_reduc_sch = 3000 # ms seems enough for a double pulse protocol
twait = 0
tstep = 0
ttest = 30
tpre = tperiod_reduc_sch - twait - ttest - tstep

tpreMeasuring_reduc_sch = tperiod_reduc_sch - ttest
tMeasuring_reduc_sch = ttest

Vhold = -135 # mV
Vtest = -20   
       
protocol = availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)
tmp_protocol.append(protocol)

## This part is the main protocol
twait = 2 
tstep = 1000
ttest = 30

for Vinact in [-105,-95,-85,-75,-65]:
    tstepList,_,_ = dataSch.Reduction_Schneider_all(Vinact)

    tpreList = []
    for tstep in tstepList :
        tpre = tperiod_reduc_sch - tstep - twait - ttest
        tpreList.append(tpre)
                      
    protocol = varying_test_duration_double_pulse(Vinact,Vhold, Vtest, tpreList, tstepList, twait, ttest)
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
schneider_reduc_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_reduc_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_reduc_sum_stats(data):

    output = []
    d_split,d_init = data.split(tperiod_reduc_sch)

    d_init = d_init.trim_left(tpreMeasuring_reduc_sch, adjust = True)
    current = d_init['ina.i_Na']

    current = current[:-1]
    index = np.argmax(np.abs(current))
    normalizing_peak = current[index] 

    for d in d_split.split_periodic(tperiod_reduc_sch, adjust=True):
        d = d.trim(tpreMeasuring_reduc_sch,tpreMeasuring_inact_sch  + tMeasuring_reduc_sch, adjust = True)
        current = d['ina.i_Na'][:-1]
        index = np.argmax(np.abs(current))
        try :
            output = output+[np.abs(current[index])/normalizing_peak]
        except :
            output = output + [float('inf')]
    return output

schneider_reduc = Experiment(
    dataset=schneider_reduc_dataset,
    protocol=schneider_reduc_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_reduc_sum_stats,
    description=schneider_reduc_desc
)
    

#######################################################################################################################
###  recovery curves - Schneider 1992
schneider_recov_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)
    
    the Vhold used here is -140mV ,-120mV and -100mV
    
    page 8 of the paper : 
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse. 


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
schneider_recov_dataset = []
for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
    time,recov,_ = dataSch.Recovery_Schneider_all(Vhold)
    variances_time = [0 for t in time]
    schneider_recov_dataset.append(np.asarray([time, recov, variances_time]))


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov_sch = []

tperiod_recov_sch = 5000 # ms
tstep1 = 200
tstep2 = 12

tMeasuring1_recov_sch = tstep1 
tpreMeasuring2_recov_sch = tperiod_recov_sch - tstep2 

Vstep1 = -20
Vstep2 = -20  

Split_list_recov_sch = [] # for the summary statistics function

for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
    twaitList,_,_ = dataSch.Recovery_Schneider_all(Vhold)

    Split_list_recov_sch.append(len(twaitList))


    tpreList = []
    for twait in twaitList:
        tpre = tperiod_recov_sch - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov_sch.append(tpre)

    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
schneider_recov_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_recov_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Nao': 120000,
            'membrane.Nai': 70000,
            'membrane.T': 297.15}

# SUMMARY STATISTICS
def schneider_recov_sum_stats(data):
    output = []
    loop = 0
    sub_loop = 0

    # spliting back the protocols
    Cumulated_len = Split_list_recov_sch[0]
    dProtocolOne,dProtocoltwoseven = data.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[1]
    dProtocolTwo,dProtocolThreeseven  = dProtocoltwoseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[2]
    dProtocolThree,dProtocolfourseven = dProtocolThreeseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[3]
    dProtocolFour,dProtocolfiveseven  = dProtocolfourseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[4]
    dProtocolFive,dProtocolsixseven = dProtocolfiveseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[5]
    dProtocolSix,dProtocolSeven  = dProtocolsixseven.split(tperiod_recov*Cumulated_len)

    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive,dProtocolSix,dProtocolSeven]
    Cumulated_len = 0
    for dOneProtocol in dProtocols:
        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)

        if loop > 0 :
            Cumulated_len += Split_list_recov_sch[loop-1]

        d_split = d_split[Cumulated_len:]  


        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov_sch[sub_loop],
             tpreMeasuringList1_recov_sch[sub_loop]+tMeasuring1_recov_sch, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov_sch, adjust = True)
            
            current_cond = dcond['ina.i_Na'][:-1]
            current_test = dtest['ina.i_Na'][:-1]



            index_cond = np.argmax(np.abs(current_cond))
            index_test = np.argmax(np.abs(current_test))
            try :
                output = output + [current_test[index_test] / current_cond[index_cond]]  
                sub_loop += 1
            except :
                output = output + [float('inf')]  
                sub_loop += 1
        loop += 1
    return output

# Experiment
schneider_recov = Experiment(
    dataset=schneider_recov_dataset,
    protocol=schneider_recov_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_recov_sum_stats,
    description=schneider_recov_desc)