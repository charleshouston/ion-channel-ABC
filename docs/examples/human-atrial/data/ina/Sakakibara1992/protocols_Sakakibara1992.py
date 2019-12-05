from ionchannelABC.protocol import recovery_manual , availability,availability_tstep_list,manual_availability
import data.ina.Sakakibara1992.data_Sakakibara1992 as data
import myokit

"""
NB : Na_c in the nygren_Na.mmt file corresponds to Na_o in the Sakakibara paper according to equation (3) in Sakakibara paper.
"""

def Protocol_IV_Sakakibara_Fig1B():
    """
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper figure 1B

    page 3 of the paper : The holding potential (Vh) was -140 mV. Depolarizing pulses in 10-mVsteps were applied from
     -70 to +20mV. Testpulses were applied at 0.1 Hz.
    
    
    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15}
    
    tperiod = 10000 # ms
    tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -140 # mV
    Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
    Vupper = 20.002
    dV = 10
    
    protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring
    

def Protocol_IV_Sakakibara_Fig3A(Na_c : int):
    """
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper in figure 3A
    takes the Na_c concentration in mM as input (by default, this value should be set to 5mM)
    
    page 5 of the paper :
    Graph showing current-voltage relations generated in 2, 5, and 20 mM [Na+],.Vh,holdingpotential. 
    Test pulses were applied at 0.1 Hz.

    
    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': Na_c,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15}
    

    
    tperiod = 10000 # ms
    tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -140 # mV
    Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
    Vupper = 40
    dV = 10
    
    protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring
        
def Protocol_Act_Sakakibara():
    """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    
    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B :
    single test pulse at a frequency of 0.1Hz
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15}
    

    
    tperiod = 10000 # ms
    tstep = 1000 # ms (not precised in the paper, 1s seems enough)
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -140 # mV
    Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
    Vupper = 20
    dV = 10
    
    protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring



def Protocol_Inact_Sakakibara():
    """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    page 7 of the paper : 
    The voltage dependence of h, was studied using a double-pulse protocol consisting of a
    1-second conditioning pulse from holding a potential of-140 mV 0.1 Hz (inset at lower left).
    Current amplitude elicited during the test pulse was normalized to that in absence of a conditioning pulse.

    TODO : The corresponding block in summary statistics function need to be changed to normalize the conditionning pulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15} # not precised but by hypothesis, 
                                       # the same values as figure 1B were
                                       # used for the concentrations
    

    
    tperiod = 10000 # ms
    tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
    twait = 2 
    ttest = 30
    tpre = tperiod - tstep - twait - ttest
    tpreMeasuring = tperiod - tstep - twait - ttest # tperiod - ttest
    tMeasuring = ttest # ttest # the sum tMeasuring, tpreMeasuring and tpostMeasuring should be equal to tperiod
    tpostMeasuring = tstep + twait
    Vhold = -140 # mV
    Vtest = -20
    Vlower = -140 
    dV = 10
    Vupper = -40 + dV
    
    protocol = availability(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)
    
    return protocol, conditions , tpreMeasuring , tMeasuring , tpostMeasuring


def Protocol_Reversal_Sakakibara():
    """
    describes the protocol used to measure the reversal potential curve in the Sakakibara Paper (figure 3B)

    page 5 of the paper :

    The reversal potential was determined by the intersecting point on the current-voltage relation curve

    the IV curve was plotted every 1mV and the protocol used for the IV curve is the same than the fig 1B
    (single pulse protocol with a frequence of 0.1Hz)
       
    """
    conditions_list, protocol_list = [], []
    tpreMeasuring_list, tMeasuring_list = [], []
    
    for Na_c in [2,5,10,20]:
        conditions = {'cleft_space_ion_concentrations.Na_c': Na_c,
                     'intracellular_ion_concentrations.Na_i': 5,
                     'membrane.T': 290.15}
        
    
        
        tperiod = 3000 # ms
        tstep = 500 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
        tpre = tperiod - tstep # before the first pulse occurs
        tpreMeasuring = tperiod - tstep # before the measurement
        tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
        
        Vhold = -140 # mV
        Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
        Vupper = 40
        dV = 1 # needs to be small to ensure a smooth curve in the detection of Ina(V) = 0
        
        protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
        
        protocol_list.append(protocol)
        conditions_list.append(conditions)
        tpreMeasuring_list.append(tpreMeasuring)
        tMeasuring_list.append(tMeasuring)
        
    return protocol_list,conditions_list,tpreMeasuring_list,tMeasuring_list

def Protocol_Inactivation_Kinetics_Sakakibara_1():
    """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15}
    

    
    tperiod = 1000 # ms
    tstep = 100 # ms 
    tpre = tperiod - tstep # before the first pulse occurs
    tpreMeasuring = tperiod - tstep # before the measurement
    tMeasuring = tstep # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -140 # mV
    Vlower = -50 
    Vupper = -20
    dV = 10
    
    protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
    
    return protocol, conditions , tpreMeasuring , tMeasuring

def Protocol_Inactivation_Kinetics_Sakakibara_2():
    """
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
    
    conditions_list, protocol_list = [], []
    tpreMeasuring_list, tMeasuring_list = [], []
    
    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15} # not precised but by hypothesis, 
                                       # the same values as figure 1B were
                                       # used for the concentrations


    ## This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse
    tperiod = 1000 # ms
    twait = 2 
    ttest = 30    
    tstep = 1
    tpre = tperiod - twait - ttest - tstep

    tpreMeasuring = tperiod - ttest
    tMeasuring = ttest # the sum tMeasuring and tpreMeasuring should be equal to tperiod
    
    Vhold = -140 # mV
    Vtest = -20        
    protocol = manual_availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)

    protocol_list.append(protocol)
    conditions_list.append(conditions)
    tpreMeasuring_list.append(tpreMeasuring)
    tMeasuring_list.append(tMeasuring)

    # Here is the rest of the protocol
    for Vcond in [-100,-80]:
        tstepList,_,_ = data.Time_course_Inactivation_Sakakibara_all(Vcond)
        
        tperiod = 10000 # ms
        twait = 2 
        ttest = 30

        tpreList = []
        for tstep in tstepList :
            tpre = tperiod - tstep - twait - ttest
            tpreList.append(tpre)
        tpreMeasuring = tperiod - ttest
        tMeasuring = ttest # the sum tMeasuring and tpreMeasuring should be equal to tperiod
        
        Vhold = -140 # mV
        Vtest = -20        
            
            
        protocol = availability_tstep_list(Vcond,Vhold, Vtest, tpreList, tstepList, twait, ttest)
        
        protocol_list.append(protocol)
        conditions_list.append(conditions)
        tpreMeasuring_list.append(tpreMeasuring)
        tMeasuring_list.append(tMeasuring)

    return protocol_list, conditions_list , tpreMeasuring_list , tMeasuring_list

def Protocol_Recovery_Sakakibara():
    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)
    
    the Vhold used here is -140mV ,-120mV and -100mV
    
    page 8 of the paper : 
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse. 


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  
    conditions_list, protocol_list = [], []
    tpreMeasuring_list1, tMeasuring_list1 = [], []
    tpreMeasuring_list2, tMeasuring_list2 = [], []
    
    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15} # not precised but by hypothesis, 
                                       # the same values as figure 1B were
                                       # used for the concentrations    
    
    for Vhold in [-140,-120,-100] :
        twaitList,_,_ = data.Recovery_Sakakibara_all(Vhold)
        
        tperiod = 10000 # ms
        tstep1 = 1000
        tstep2 = 1000
        
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


def Protocol_Recovery_Fast_Time_Constant_Sakakibara():
    """
    describes the protocol used to measure the fast time constant from the Recovery of I_na in the Sakakibara Paper (figure 9)
    
    the Vhold used here is -140mV ,-120mV -110mV, -100mV and -90mV
    
    The protocol is not decribed but was assumed to be the same as for the recovery protocol (fig 8A)
    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  
    conditions_list, protocol_list = [], []
    tpreMeasuring_list1, tMeasuring_list1 = [], []
    tpreMeasuring_list2, tMeasuring_list2 = [], []
    
    conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                 'intracellular_ion_concentrations.Na_i': 5,
                 'membrane.T': 290.15} # not precised but by hypothesis, 
                                       # the same values as figure 1B were
                                       # used for the concentrations  
    
    twaitList = [2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
    
    for Vhold in [-140, -120, -110, -100, -90]:
          
        tperiod = 10000 # ms
        tstep1 = 1000
        tstep2 = 1000
        
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
            tpreMeasuring_list2, tMeasuring_list2, twaitList
    
    
    
    
    
    
    
    
    