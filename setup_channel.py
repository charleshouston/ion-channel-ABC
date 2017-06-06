'''
Author: Charles Houston

Specific channel settings for use with approximate Bayesian computation procedure.
'''

import distributions as Dist

# class AbstractChannel(object):



class Setup_icat():
    self.name = 'icat'
    self.prior_intervals = [(0,100),
                            (1,10),
                            (0,10),
                            (0,100),
                            (1,100),
                            (0,10),
                            (0,100),
                            (1,100),
                            (0,100),
                            (1,10),
                            (0,0.1),
                            (0,100),
                            (1,100),
                            (0,0.1),
                            (0,100),
                            (1,100)]
