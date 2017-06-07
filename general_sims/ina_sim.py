import matplotlib.pyplot as plt
import numpy as np
import myokit
import simulations

import protocols

m,p,x =  myokit.load('bondarenko2004.mmt')

v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')
s = myokit.Simulation(m)

steps = np.linspace(-80,40,12)

res = simulations.activation_sim(s,steps,m.get('ina.E_Na').value())
# p = protocols.steptrain(
#     vsteps = np.array([-20]),
#     vhold = -100,
#     tpre = 5000,
#     tstep = 300
# )
# s.set_protocol(p)
# t = p.characteristic_time()

# d = s.run(t, log=['environment.time','membrane.V','ina.i_Na'],log_interval=.1)
# d.trim_left(5000,adjust=True)
# d.trim_right(50)

plt.figure()
plt.plot(steps,res[0],'bo')
plt.show()
