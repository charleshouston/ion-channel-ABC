import matplotlib.pyplot as plt
import ast
import numpy as np

import math

import myokit

import protocols

f = open('ABCPredCalciumTType.txt')
pool = f.readline()
pool = ast.literal_eval(pool)

m,p,x = myokit.load('Takeuchi2013_iCaT.mmt')

V = np.linspace(-100,40,1000)

params_exp = []
params_exp.append(m.get('icat_d_gate.dssk1').value())
params_exp.append(m.get('icat_d_gate.dssk2').value())
params_exp.append(m.get('icat_d_gate.dtauk1').value())
params_exp.append(m.get('icat_d_gate.dtauk2').value())
params_exp.append(m.get('icat_d_gate.dtauk3').value())
params_exp.append(m.get('icat_d_gate.dtauk4').value())
params_exp.append(m.get('icat_d_gate.dtauk5').value())
params_exp.append(m.get('icat_d_gate.dtauk6').value())

params_exp.append(m.get('icat_f_gate.fssk1').value())
params_exp.append(m.get('icat_f_gate.fssk2').value())
params_exp.append(m.get('icat_f_gate.ftauk1').value())
params_exp.append(m.get('icat_f_gate.ftauk2').value())
params_exp.append(m.get('icat_f_gate.ftauk3').value())
params_exp.append(m.get('icat_f_gate.ftauk4').value())
params_exp.append(m.get('icat_f_gate.ftauk5').value())
params_exp.append(m.get('icat_f_gate.ftauk6').value())

dssk1 = params_exp[0]
dssk2 = params_exp[1]
dtauk1 = params_exp[2]
dtauk2 = params_exp[3]
dtauk3 = params_exp[4]
dtauk4 = params_exp[5]
dtauk5 = params_exp[6]
dtauk6 = params_exp[7]

fssk1 = params_exp[8]
fssk2 = params_exp[9]
ftauk1 = params_exp[10]
ftauk2 = params_exp[11]
ftauk3 = params_exp[12]
ftauk4 = params_exp[13]
ftauk5 = params_exp[14]
ftauk6 = params_exp[14]

dss = pow((1+np.exp(-V+dssk1)/dssk2), -1)
tau_d = pow((dtauk1*np.exp((V+dtauk2)/dtauk3) + dtauk4*np.exp(-(V+dtauk2)/dtauk5)), -1)
fss = pow((1 + np.exp((V+fssk1)/fssk2)), -1)
tau_f = pow((ftauk1*np.exp(-(V+ftauk2)/ftauk3) + ftauk4*np.exp((V+ftauk2)/ftauk5)), -1)

alpha_d = dss/tau_d
beta_d = (1-dss)/tau_d
alpha_f = fss/tau_f
beta_f = (1-fss)/tau_f

plt.subplot(1,4,1)
plt.subplot(1,4,1)
plt.plot(V,alpha_d,'b-')
plt.subplot(1,4,2)
plt.plot(V,beta_d,'b-')
plt.subplot(1,4,3)
plt.plot(V,alpha_f,'b-')
plt.subplot(1,4,4)
plt.plot(V,beta_f, 'b-')

for params in pool:
    dssk1 = params[0]
    dssk2 = params[1]
    dtauk1 = params[2]
    dtauk2 = params[3]
    dtauk3 = params[4]
    dtauk4 = params[5]
    dtauk5 = params[6]

    fssk1 = params[7]
    fssk2 = params[8]
    ftauk1 = params[9]
    ftauk2 = params[10]
    ftauk3 = params[11]
    ftauk4 = params[12]
    ftauk5 = params[13]

    dss = pow((1+np.exp(-V+dssk1)/dssk2), -1)
    tau_d = pow((dtauk1*np.exp((V+dtauk2)/dtauk3) + dtauk4*np.exp(-(V+dtauk2)/dtauk5)), -1)
    fss = pow((1 + np.exp((V+fssk1)/fssk2)), -1)
    tau_f = pow((ftauk1*np.exp(-(V+ftauk2)/ftauk3) + ftauk4*np.exp((V+ftauk2)/ftauk5)), -1)

    alpha_d = dss/tau_d
    beta_d = (1-dss)/tau_d
    alpha_f = fss/tau_f
    beta_f = (1-fss)/tau_f

    plt.subplot(1,4,1)
    plt.plot(V,alpha_d,'r--',linewidth=0.1, alpha = 0.7)
    plt.subplot(1,4,2)
    plt.plot(V,beta_d,'r--',linewidth=0.1,alpha=0.7)
    plt.subplot(1,4,3)
    plt.plot(V,alpha_f,'r--',linewidth=0.1,alpha=0.7)
    plt.subplot(1,4,4)
    plt.plot(V,beta_f, 'r--', linewidth=0.1,alpha=0.7)

# plt.show()



m,p,x = myokit.load('Takeuchi2013_iCaT.mmt')
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')
s = myokit.Simulation(m)
# Create protocol for step experiment
p = protocols.steptrain(
    vsteps = np.array([-20]), # Use voltage steps from experiments
    vhold = -80, # Holding potential at -80mV
    tpre = 5000, # 5000ms pre-conditioning before each step
    tstep = 300, # 300ms at step potential
    tpost = 1000
)
t = p.characteristic_time()

plt.figure()
for params in pool:
    s.reset()

    # Set d gate and f gate to current parameter values
    s.set_constant('icat_d_gate.dssk1',params[0])
    s.set_constant('icat_d_gate.dssk2',params[1])
    s.set_constant('icat_d_gate.dtauk1',params[2])
    s.set_constant('icat_d_gate.dtauk2',params[3])
    s.set_constant('icat_d_gate.dtauk3',params[4])
    s.set_constant('icat_d_gate.dtauk4',params[5])
    s.set_constant('icat_d_gate.dtauk5',params[6])
    s.set_constant('icat_d_gate.dtauk6',params[7])

    s.set_constant('icat_f_gate.fssk1',params[8])
    s.set_constant('icat_f_gate.fssk2',params[9])
    s.set_constant('icat_f_gate.ftauk1',params[10])
    s.set_constant('icat_f_gate.ftauk2',params[11])
    s.set_constant('icat_f_gate.ftauk3',params[12])
    s.set_constant('icat_f_gate.ftauk4',params[13])
    s.set_constant('icat_f_gate.ftauk5',params[14])
    s.set_constant('icat_f_gate.ftauk6',params[14])

    s.set_protocol(p)

    d = s.run(t, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)

    d.trim_left(5000,adjust=True)
    d.trim_right(300)

    plt.plot(d['environment.time'],d['icat.i_CaT'],'r--',alpha=0.7,linewidth=0.1)

s.reset()

# Set d gate and f gate to current parameter values
s.set_constant('icat_d_gate.dssk1',params_exp[0])
s.set_constant('icat_d_gate.dssk2',params_exp[1])
s.set_constant('icat_d_gate.dtauk1',params_exp[2])
s.set_constant('icat_d_gate.dtauk2',params_exp[3])
s.set_constant('icat_d_gate.dtauk3',params_exp[4])
s.set_constant('icat_d_gate.dtauk4',params_exp[5])
s.set_constant('icat_d_gate.dtauk5',params_exp[6])
s.set_constant('icat_d_gate.dtauk6',params_exp[7])

s.set_constant('icat_f_gate.fssk1',params_exp[8])
s.set_constant('icat_f_gate.fssk2',params_exp[9])
s.set_constant('icat_f_gate.ftauk1',params_exp[10])
s.set_constant('icat_f_gate.ftauk2',params_exp[11])
s.set_constant('icat_f_gate.ftauk3',params_exp[12])
s.set_constant('icat_f_gate.ftauk4',params_exp[13])
s.set_constant('icat_f_gate.ftauk5',params_exp[14])
s.set_constant('icat_f_gate.ftauk6',params_exp[14])

d = s.run(t,log=['environment.time','membrane.V','icat.i_CaT'],log_interval=.1)
d.trim_left(5000,adjust=True)
d.trim_right(300)

plt.plot(d['environment.time'],d['icat.i_CaT'],'b-')

plt.show()
