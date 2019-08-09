import myokit
from typing import List


def recovery_tpreList(twait: List[float],
             vhold: float,
             vstep1: float,
             vstep2: float,
             tpre: List[float],
             tstep1: float,
             tstep2: float,
             tpost: float=0.) -> myokit.Protocol:
    """Standard double-pulse recovery protocol."""

    # Check time arguments
    for t in tpre:
        if t < 0:
            raise ValueError('Time tpre can not be negative.')
    if tstep1 < 0:
        raise ValueError('Time tstep can not be negative.')
    if tstep2 < 0:
        raise ValueError('Time tstep can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    for t in twait:
        if t < 0:
            raise ValueError('Time twait can not be negative.')

    assert(len(tpre) == len(twait))


    # Create protocol
    p = myokit.Protocol()
    time = 0
    for i in range(len(twait)):
        twait_i = twait[i]
        tpre_i = tpre[i]
        if tpre_i > 0:
            p.schedule(vhold, time, tpre_i)
            time += tpre_i
        if tstep1 > 0:
            p.schedule(vstep1, time, tstep1)
            time += tstep1
        if twait_i > 0:
            p.schedule(vhold, time, twait_i)
            time += twait_i
        if tstep2 > 0:
            p.schedule(vstep2, time, tstep2)
            time += tstep2
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p


def manual_steptrain_linear(vlist: List[float],
                 vhold: float,
                 tpre: float,
                 tstep: float):

    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')


    # Create protocol
    p = myokit.Protocol()
    time = 0
    for vstep in vlist:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
    return p


def varying_test_duration(tstep: List[float],
                          vhold: float,
                          vstep: float,
                          tpre: float,
                          tpost: float=0.) -> myokit.Protocol:
    """Varying duration of test pulses to same potential."""

    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    for ts in tstep:
        if ts < 0:
            raise ValueError('Time tstep can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')

    # Create protocol
    p = myokit.Protocol()
    time = 0.
    for i in range(len(tstep)):
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep[i] > 0:
            p.schedule(vstep, time, tstep[i])
            time += tstep[i]
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p


def varying_test_duration_double_pulse(vstep : float,
                            vhold: float,
                            vtest: float,
                            tpreList: List[float],
                            tstepList: List[float],
                            twait: float,
                            ttest: float,
                            tpost: float=0.) -> myokit.Protocol:
    """Varying duration of conditioning pulses to same potential with a test pulse."""


    # Check time arguments
    for tpre in tpreList:
        if tpre < 0:
            raise ValueError('Time tpre can not be negative.')
    for tstep in tstepList :
        if tstep < 0:
            raise ValueError('Time tstep can not be negative.')
    assert(len(tpreList) == len(tstepList))

    if twait < 0:
        raise ValueError('Time tstep can not be negative.')
    if ttest < 0:
        raise ValueError('Time tpost can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')

    # Create protocol
    p = myokit.Protocol()
    time = 0.
    for i in range(len(tstepList)):
        tstep = tstepList[i]
        tpre = tpreList[i]
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
        if twait > 0:
            p.schedule(vhold, time, twait)
            time += twait
        if ttest > 0:
            p.schedule(vtest, time, ttest)
            time += ttest
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p
