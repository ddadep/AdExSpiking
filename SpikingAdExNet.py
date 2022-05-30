'''
A comprehensive neural simulation of slow-wave sleep and highly responsive wakefulness dynamics 
Jennifer S. Goldman, Lionel Kusch, Bahar Hazal Yalçinkaya, Damien Depannemaecker, Trang-Anh E. Nghiem, Viktor Jirsa, Alain Destexhe 
bioRxiv 2021.08.31.458365; doi: https://doi.org/10.1101/2021.08.31.458365
'''
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

##########################################################
b_val = 60 ##### 60 for Up/Down, 1 for AI state ##########
##########################################################

#########################################################
#Define conditions for simulation
#start Brian scope:
start_scope()
#set dt value for integration (ms):
DT=0.1
seed(4)
defaultclock.dt = DT*ms

#total duration of the simulation (ms):
TotTime=3000
duration = TotTime*ms


#######################################################
#set the number of neuron of each population:
#inhibitory Fast Spiking (FS, population 1):
N1 = 2000
#Excitatory Regular Spiking (RS, population 2):
N2 = 8000


########################################################
# define equations of the model 
# define units of parameter
eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Pvar:1
Is:ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''

########################################################
#Create populations:

# Population 1 - FS

b1 = 0.0*pA #no adaptation for FS
#generate the population
G1 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
#set values:
# initial values of variables:
G1.v = -65 *mV
G1.w = 0.0 *pA
G1.GsynI =0.0 *nS
G1.GsynE =0.0 *nS
# parameters values:
#soma:
G1.Cm = 200.*pF
G1.gl = 10.*nS
G1.El = -65.*mV
G1.Vt = -50.*mV
G1.Dt = 0.5*mV
G1.tau_w = 1.0 *ms #(no adapation, just to do not have error due to zero division)
G1.a = 0.0 *nS
G1.Is = 0.0  
#synapses:
G1.Ee =0.*mV
G1.Ei =-80.*mV
G1.Tsyn =5.*ms


# Population 2 - RS
b2 = b_val*pA 
#generate the population
G2 = NeuronGroup(N2, eqs, threshold='v > -40.0*mV', reset='v = -55*mV; w += b2', refractory='5*ms',  method='heun')
#set values:
# initial values of variables:
G2.v = -65.*mV
G2.w = 1.35 *pA
G2.GsynI =0.0 *nS
G2.GsynE =0.0 *nS
# parameters values:
#soma:
G2.Cm = 200.*pF
G2.gl = 10.*nS
G2.El = -63.*mV
G2.Vt = -50.*mV
G2.Dt = 2.*mV
G2.tau_w = 500*ms
G2.a = 0.*nS
G2.Is = 0.*nA  
#synpases:
G2.Ee =0.*mV
G2.Ei =-80.*mV
G2.Tsyn =5.*ms




#######################################################
# external drive---------------------------------------

P_ed=PoissonGroup(8000, rates= .35*Hz) 


#######################################################
# connections-------------------------------------------
#quantal increment when spike:
Qi=5.*nS
Qe=1.5*nS

#probability of connection
prbC= 0.05 

#synapses from FS to RS:
S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV')
S_12.connect('i!=j', p=prbC)
#synapses from FS to FS:
S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
S_11.connect('i!=j',p=prbC)
#synapses from RS to FS:
S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
S_21.connect('i!=j',p=prbC)
#synapses from RS to RS:
S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
S_22.connect('i!=j', p=prbC)



#synapses from external drive to both populations:
S_ed_in = Synapses(P_ed, G1, on_pre='GsynE_post+=Qe')
S_ed_in.connect(p=prbC)

S_ed_ex = Synapses(P_ed, G2, on_pre='GsynE_post+=Qe')
S_ed_ex.connect(p=prbC)


######################################################
#set recording during simulation
#number of neuron record of each population:
Nrecord=1

M1G1 = SpikeMonitor(G1)
M2G1 = StateMonitor(G1, 'v', record=range(Nrecord))
M3G1 = StateMonitor(G1, 'w', record=range(Nrecord))
FRG1 = PopulationRateMonitor(G1)


M1G2 = SpikeMonitor(G2)
M2G2 = StateMonitor(G2, 'v', record=range(Nrecord))
M3G2 = StateMonitor(G2, 'w', record=range(Nrecord))
FRG2 = PopulationRateMonitor(G2)


#######################################################
# Useful trick to record global variables ------------------------------------------------------

Gw_inh = NeuronGroup(1, 'Wtot : ampere', method='rk4')
Gw_exc = NeuronGroup(1, 'Wtot : ampere', method='rk4')

SwInh1=Synapses(G1, Gw_inh, 'Wtot_post = w_pre : ampere (summed)')
SwInh1.connect(p=1)
SwExc1=Synapses(G2, Gw_exc, 'Wtot_post = w_pre : ampere (summed)')
SwExc1.connect(p=1)

MWinh = StateMonitor(Gw_inh, 'Wtot', record=0)
#MWexc 
P2mon = StateMonitor(Gw_exc, 'Wtot', record=0)



#GV_inh = NeuronGroup(1, 'Vtot : volt', method='rk4')
#GV_exc = NeuronGroup(1, 'Vtot : volt', method='rk4')

#SvInh1=Synapses(G_inh, GV_inh, 'Vtot_post = v_pre : volt (summed)')
#SvInh1.connect(p=1)
#SvExc1=Synapses(G_exc, GV_exc, 'Vtot_post = v_pre : volt (summed)')
#SvExc1.connect(p=1)

#MVinh = StateMonitor(GV_inh, 'Vtot', record=0)
#MVexc = StateMonitor(GV_exc, 'Vtot', record=0)

#######################################################
#Run the simulation

print('--##Start simulation##--')
run(duration)
print('--##End simulation##--')


#######################################################
#Prepare recorded data

#organize arrays for raster plots:
RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
RasG2 = np.array([M1G2.t/ms, M1G2.i])


#organize time series of single neuron variables
LVG1=[]
LwG1=[]
LVG2=[]
LwG2=[]
for a in range(Nrecord):
    LVG1.append(array(M2G1[a].v/mV))
    LwG1.append(array(M3G1[a].w/mamp))
    LVG2.append(array(M2G2[a].v/mV))
    LwG2.append(array(M3G2[a].w/mamp))

Ltime=array(M2G1.t/ms)

#Calculate population firing rate :

#function for binning:
def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)


BIN=5
time_array = np.arange(int(TotTime/DT))*DT

LfrG2=np.array(FRG2.rate/Hz)
TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)

LfrG1=np.array(FRG1.rate/Hz)
TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)

TimBinned,Pw=bin_array(time_array, BIN, time_array),bin_array(P2mon[0].Wtot, BIN, time_array)
                

##############################################################################
# prepare figure and plot

fig=plt.figure(figsize=(12,5))
fig.suptitle('AdEx. b='+str(b2)+' Tau_w='+str(G2.tau_w[0]), fontsize=12)
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)

ax1.set_title('Network activity')
ax1.plot(RasG1[0], RasG1[1], linestyle='None', marker=',', color='r')
ax1.plot(RasG2[0], RasG2[1], linestyle='None', marker=',', color='SteelBlue')

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Neuron index')

ax3.plot(TimBinned/1000,popRateG1, 'r')
ax3.plot(TimBinned/1000,popRateG2, 'SteelBlue')
ax3.plot(TimBinned/1000,Pw, 'orange')

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('population Firing Rate')

for a in range(Nrecord):
    ax2.plot(Ltime, LVG1[a],'r')
    ax4.plot(Ltime, LwG1[a],'r')
    ax2.plot(Ltime, LVG2[a],'SteelBlue')
    ax4.plot(Ltime, LwG2[a],'SteelBlue')


ax2.set_title('Single neuron variables')
ax2.set_ylabel('$V_m$')

ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('$w$')

fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(111)
ax1.set_title('Network activity')
ax1.plot(RasG1[0], RasG1[1], linestyle='None', marker=',', color='r')
ax1.plot(RasG2[0], RasG2[1], linestyle='None', marker=',', color='SteelBlue')

fig=plt.figure(figsize=(8,5))
ax3=fig.add_subplot(111)
ax2 = ax3.twinx()
ax3.plot(TimBinned/1000,popRateG1, 'r')
ax3.plot(TimBinned/1000,popRateG2, 'SteelBlue')
ax2.plot(TimBinned/1000,(Pw/8000)*1e9, 'orange')
ax2.set_ylabel('mean w (nA)')
ax2.set_ylim(0.0, 0.045)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('population Firing Rate')



plt.tight_layout()
plt.show()

	
	
	

