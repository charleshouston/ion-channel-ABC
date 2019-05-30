import myokit
from typing import List


def recovery(twait: List[float],
             vhold: float,
             vstep1: float,
             vstep2: float,
             tpre: float,
             tstep1: float,
             tstep2: float,
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
            p.schedule(vhold, time, t)
            time += t
        if tstep2 > 0:
            p.schedule(vstep2, time, tstep2)
            time += tstep2
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p