from ionchannelABC.protocol import recovery_manual , availability, manual_steptrain_linear,availability_tstep_list,manual_availability
import data.ina.Schneider1994.data_Schneider1994 as dataSc
import myokit

# TODO : need to discuss what they call prepulse duration in the paper :
#  is is the duration of the conditionning pulse or the time between the conditioning pulse and the test pulse ?

def Protocol_IV_Schneider():
    """
    describes the protocol used to measure the IV peak-current curve in the Schneider Paper figure 1B
    
    page 2 of the paper :

    depolarizing the membrane from -135 mV to test
    potentials varying from -85 to +55 mV in 10-mV.

    simple pulse protocol at the frequency 1Hz (the frequency of the pulses is not given)
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
    
    tperiod = 1000 # ms (not precised in the paper, 1s seems enough)
    tstep = 12 # ms 
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -135 # mV
    Vlower = -85.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
    Vupper = 55
    dV = 10
    
    protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring

def Protocol_TauM_Schneider():
    """
    describes the protocol used to measure the activation curve in the Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
      
    tperiod = 1000 # ms
    tstep = 12 # ms (not precised in the paper, 1s seems enough)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    
    Vlist, _ , _ = dataSc.TauM_Activation_Schneider()
    Vhold = -135 # mV

    protocol = manual_steptrain_linear(Vlist,Vhold,tpre,tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring


def Protocol_TauF_Schneider():
    """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 3B)
    
    
    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
    
    
    tperiod = 1000 # ms
    tstep = 12 # ms (not precised in the paper, 1s seems enough)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    
    Vlist, _ , _ = dataSc.TauF_Inactivation_Schneider()
    Vhold = -135 # mV

    
    protocol = manual_steptrain_linear(Vlist,Vhold,tpre,tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring

def Protocol_TauS_Schneider():
    """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 5B)
    
    
    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
    
    
    tperiod = 1000 # ms
    tstep = 12 # ms (not precised in the paper, 1s seems enough)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    
    Vlist, _ , _ = dataSc.TauS_Inactivation_Schneider()
    Vhold = -135 # mV

    
    protocol = manual_steptrain_linear(Vlist,Vhold,tpre,tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring

def Protocol_Inact_Schneider():
    """
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

    conditions_list, protocol_list = [], []
    tpreMeasuring_list, tMeasuring_list = [], []

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
    
    for tstep in [32,64,128,256,512]:
        tperiod = 3000 # ms
        #tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
        twait = 2 
        ttest = 30
        tpre = tperiod - tstep - twait - ttest
        tpreMeasuring = tperiod - ttest
        tMeasuring = ttest # the sum tMeasuring and tpreMeasuring should be equal to tperiod
        Vhold = -135 # mV
        Vtest = -20
        Vlower = -135.001 
        Vupper = 5
        dV = 10
    
        protocol = availability(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)
    
        protocol_list.append(protocol)
        conditions_list.append(conditions)
        tpreMeasuring_list.append(tpreMeasuring)
        tMeasuring_list.append(tMeasuring)
        
    return protocol_list,conditions_list,tpreMeasuring_list,tMeasuring_list


def Protocol_Relative_Peak_Schneider():
    """
    describes the protocol used to measure the Relative_peak IV curve in the Schneider Paper (figure 5A)
    
    page 5 of the paper :

    Reduction of the Na + current peak
    amplitude elicited by pulses to -20 mV following prepotentials
    varying in time (abscissa) and magnitude. Symbols represent
    different inactivation potential values: -105 mV to -65 mV

    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
    """

    conditions_list, protocol_list = [], []
    tpreMeasuring_list, tMeasuring_list = [], []

    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}
    
    for Vinact in [-105,-95,-85,-75,-65]:
        tstepList,_,_ = dataSc.Reduction_Schneider_all(Vinact)

        tperiod = 3000 # ms seems enough for a double pulse protocol
        twait = 2 
        ttest = 30

        tpreList = []
        for tstep in tstepList :
            tpre = tperiod - tstep - twait - ttest
            tpreList.append(tpre)
        tpreMeasuring = tperiod - ttest
        tMeasuring = ttest # the sum tMeasuring and tpreMeasuring should be equal to tperiod
        
        Vhold = -135 # mV
        Vtest = -20        
            
            
        protocol = availability_tstep_list(Vinact,Vhold, Vtest, tpreList, tstepList, twait, ttest)
        
        protocol_list.append(protocol)
        conditions_list.append(conditions)
        tpreMeasuring_list.append(tpreMeasuring)
        tMeasuring_list.append(tMeasuring)

    return protocol_list, conditions_list , tpreMeasuring_list , tMeasuring_list


def Protocol_Recovery_Schneider():
    """
    describes the protocol used to measure the Recovery of I_na in the Schneider Paper (figure 6)
    
    the Vhold used here is -75 mV, -85 mV, -95 mV, -105 mV,-115 mV -125 mV -135 mV 
    
    page 5 of the paper : 

    For the study of the time course of recovery from inactivation
    the cells were depolarized to -20 mV for 200 ms
    to render all Na + channels inactivated. Hyperpolarizing
    prepulses from -135 to -75 mV were then applied (recovery
    potential). For each prepulse potential, the prepulse
    duration was increased logarithmically from 2 to
    1024ms. Following each prepulse, a 12-ms step to
    -20 mV tested the fraction of Na + channels that had
    recovered from inactivation.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  
    conditions_list, protocol_list = [], []
    tpreMeasuring_list1, tMeasuring_list1 = [], []
    tpreMeasuring_list2, tMeasuring_list2 = [], []
    
    conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                'intracellular_ion_concentrations.Na_i': 70,
                'membrane.T': 297.15}   
    
    for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
        twaitList,_,_ = dataSc.Recovery_Schneider_all(Vhold)
        
        tperiod = 5000 # ms
        tstep1 = 200
        tstep2 = 12
        
        tpreList = []
        for twait in twaitList:
            tpre = tperiod - tstep1 - twait - tstep2
            tpreList.append(tpre)
            tpreMeasuring_list1.append(tpre)

            
        tMeasuring1 = tstep1 # the sum tMeasuring, tpreMeasuring and tpostMeasuring should be equal to tperiod


        tpreMeasuring2 = tperiod -tstep2 
        tMeasuring2 = tstep2

        Vstep1 = -20
        Vstep2 = -20       
        
        protocol = recovery_manual(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
        
        protocol_list.append(protocol)
        conditions_list.append(conditions)

        tMeasuring_list1.append(tMeasuring1)

        tpreMeasuring_list2.append(tpreMeasuring2)
        tMeasuring_list2.append(tMeasuring2)

    return protocol_list, conditions_list , tpreMeasuring_list1 , tMeasuring_list1,\
            tpreMeasuring_list2, tMeasuring_list2