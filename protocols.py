'''
Author: Charles Houston
Date: 26/5/17

Practical functions for generating a selection of experimental protocols
in myokit. Extends basic protocols included in myokit.
'''

import myokit

def steptrain(vsteps, vhold, tpre, tstep, tpost=0):
    """
    Creates a series of increasing or decreasing steps away from some fixed
    holding value. This type of protocol is commonly used to measure activation
    or inactivation in ion channel models.

      1. For the first ``tpre`` time units, the pacing variable is held at the
         value given by ``vhold``.
      2. For the next ``tstep`` time units, the pacing variable is held at a
         value from ``vsteps``
      3. For the next ``tpost`` time units, the pacing variable is held at
         ``vhold`` again.

    These three steps are repeated for each value in the ``vsteps``.
    """
    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    # Create protocol
    p = myokit.Protocol()
    time = 0
    for vstep in vsteps:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p

def steptrain_double(vsteps, vhold, vpost, tpre, tstep, tbetween, tpost):
    """
    Creates a series of double steps.

      1. For the first ``tpre`` time units, the pacing variable is held at the
         value given by ``vhold``.
      2. For the next ``tstep`` time units, the pacing variable is held at a
         value from ``vsteps``
      3. For the next ``tbetween`` time units, the pacing variable is returned
         to ``vhold``.
      4. For the next ``tpost`` time units, the pacing variable is held at
         ``vpost``.

    These four steps are repeated for each value in the ``vsteps``.
    """
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if tbetween < 0:
        raise ValueError('Time tbetween can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    # Create protocol
    p = myokit.Protocol()
    time = 0
    for vstep in vsteps:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
        if tbetween > 0:
            p.schedule(vhold, time, tbetween)
            time += tbetween
        if tpost > 0:
            p.schedule(vpost, time, tpost)
            time += tpost
    return p

def intervaltrain(vstep, vhold, vpost, tpre, tstep, tintervals, tpost):
    """
    Creates a series of two steps separated by differing time intervals.

      1. For the first ``tpre`` time units, the pacing variable is held at the
         value given by ``vhold``.
      2. For the next ``tstep`` time units, the pacing variable is held at ``vstep``.
      3. For the next value from ``tintervals`` time units, the pacing variable is returned
         to ``vhold``.
      4. For the next ``tpost`` time units, the pacing variable is held at
         ``vpost``.

    These four steps are repeated for each value in the ``tintervals``.
    """
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    for t in tintervals:
        if t < 0:
            raise ValueError('Time intervals can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    p = myokit.Protocol()
    time = 0
    for tinterval in tintervals:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
        if tinterval > 0:
            p.schedule(vhold, time, tinterval)
            time += tinterval
        if tpost > 0:
            p.schedule(vpost, time, tpost)
            time += tpost
    return p
