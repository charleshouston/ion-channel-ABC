import myokit
from typing import List


def recovery(twait: List[float],
             vhold: float,
             vstep1: float,
             vstep2: float,
             tpre: float,
             tstep1: float,
             tstep2: float,
             vwait: float=None,
             tpost: float=0.) -> myokit.Protocol:
    """Standard double-pulse recovery protocol."""

    # Check time arguments
    if tpre < 0:
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

    if vwait is None:
        vwait = vhold

    # Create protocol
    p = myokit.Protocol()
    time = 0
    for t in twait:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep1 > 0:
            p.schedule(vstep1, time, tstep1)
            time += tstep1
        if t > 0:
            p.schedule(vwait, time, t)
            time += t
        if tstep2 > 0:
            p.schedule(vstep2, time, tstep2)
            time += tstep2
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p


def availability(vsteps: List[float],
                 vhold: float,
                 vtest: float,
                 tpre: float,
                 tstep: float,
                 twait: float,
                 ttest: float,
                 tpost: float=0.) -> myokit.Protocol:
    """Standard availability (inactivation) protocol."""
    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if twait < 0:
        raise ValueError('Time tstep can not be negative.')
    if ttest < 0:
        raise ValueError('Time tpost can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')

    # Create protocol
    p = myokit.Protocol()
    time = 0.
    for vstep in vsteps:
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


def availability_linear(vstart: float,
                        vend: float,
                        dv: float,
                        vhold: float,
                        vtest: float,
                        tpre: float,
                        tstep: float,
                        twait: float,
                        ttest: float,
                        tpost: float=0.) -> myokit.Protocol:
    """Standard availability (inactivation) protocol with linear steps."""

    # Check v arguments
    if vend > vstart:
        if dv <= 0:
            raise ValueError('vend > vstart so dv must be strictly positive.')
    else:
        if dv >= 0:
            raise ValueError('vend < vstart so dv must be negative.')

    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if twait < 0:
        raise ValueError('Time tstep can not be negative.')
    if ttest < 0:
        raise ValueError('Time tpost can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')

    # Create protocol
    p = myokit.Protocol()
    time = 0.
    for i in range(int(abs((vend - vstart) / dv))):
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstart + i * dv, time, tstep)
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

