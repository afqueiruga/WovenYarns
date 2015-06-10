"""
Verify the RK routines on silly ODES
"""
import numpy as np

from dolfin import *
from src.RKnew import exRK
import matplotlib
from matplotlib import pylab as plt

def problem1():
    """
    2nd order oscillator
    """
    M_1 = None #np.array([[1.0]],dtype=np.double)
    x = np.array([0.1],dtype=np.double)
    u = np.array([0.0],dtype=np.double)
    def sys_1(time):
        return np.array([-x[0]],np.double)
    def bcapp(K,R,time,hold):
        pass
    def update():
        pass

    odef = exRK.RK_field(2, [u,x],M_1,sys_1,bcapp,update)

    Tmax = 10.0
    NT = 500
    h = Tmax / NT
    RKER = exRK.exRK(h, exRK.exRK_table[1], [odef])

    us = np.zeros(NT+1,dtype=np.double)
    us[0] = x[0]
    for t in xrange(NT):
        RKER.march()
        us[t+1] = x[0]

    plt.plot(us)
    plt.show()

def problem2(NT,type,plotit=False):
    """
    That DAE of mine
    """
    # Define Variables
    nd = lambda x : np.array([x],dtype=np.double)
    
    y = nd(1.0)
    v = nd(0)
    T = nd(1.0)
    I = nd(0)
    
    # Define Parameters
    K = 1.0
    K_alpha = -0.1
    R = 1.0
    R_alpha = 0.1
    Vapp = 1.0
    B = 0.1
    h_conv = 1.0
    h_conv_alpha = 1.0
    Tinf = 0.0
    
    # The same trivials for each
    def bcapp(K,R,time,hold):
        pass
    def update():
        pass
    
    # Build Fields
    def sys_mech(time):
        return nd( -(K+K_alpha*T[0])*y[0] + I[0]*B )
    field_mech = exRK.RK_field(2, [v,y],None,sys_mech,bcapp,update)
    def sys_temp(time):
        return nd( I[0]**2.0/R - (h_conv+h_conv_alpha*v[0])*(T[0]-Tinf) )
    field_temp = exRK.RK_field(1, [T],None,sys_temp,bcapp,update)
    def sys_em(time):
        return nd( -Vapp + (R+R_alpha*T[0])*I[0] + v[0]*B), nd([-R])
    field_em = exRK.RK_field(0, [I],None,sys_em,bcapp,update)
    
    # Create the timestepper and march
    Tmax = 10.0
    h = Tmax / NT
    RKER = exRK.exRK(h, exRK.exRK_table[type], [field_mech,field_temp,field_em])

    if plotit:
        us = np.zeros([NT+1,4],dtype=np.double)
        us[0,:] = [y[0],v[0],T[0],I[0]]
    for t in xrange(NT):
        RKER.march()
        if plotit:
            us[t+1,:] = [y[0],v[0],T[0],I[0]]

    if plotit:
        times = np.linspace(0.0,Tmax,NT+1)
        plt.plot(times,us[:,0],label='y')
        plt.plot(times,us[:,1],label='v')
        plt.plot(times,us[:,2],label='T')
        plt.plot(times,us[:,3],label='I')
        plt.legend()
        plt.show()
    return [y[0],v[0],T[0],I[0]]


from IPython import embed

# problem2(1000,'RK4',True)
# exit()

tests = {
    # '1':range(5000,100000,10000),
    'RK2-trap':range(100,2000,200),
    'RK2-mid':range(100,2000,200),
    'RK3-1':range(100,3000,300),
    'RK4':range(100,1000,100)+[10000]
    }

results = {}
# embed()
for method,NTS in tests.iteritems():
    R = []
    for nt in NTS:
        res=problem2(nt,method,False)
        R.append([nt]+res)
    results[method]=np.array(R)

from collections import defaultdict
# colorgen =
embed()
Tmax = 1.0
labels = [ 'y(0)','z(0.5)','T(0.5)','V(0.5)' ]
def make_error_plots(sd, sdconv, labels):
    font = {'family' : 'normal',
            'size'   : 16}
    matplotlib.rc('font', **font)
    hs = { k:Tmax / dat[:,0] for k,dat in sd.iteritems() }
    # hsconv = [ Tmax / dat[:,0] for dat in sdconv]
    colors = ["b","g","r"]
    c = 'r'
    for i in xrange(4):
        label = labels[i]
        plt.figure()
        plt.xlabel("Logarithm of time step size log(h)")
        plt.ylabel("Logarithm of error in "+label)
        best = sd['RK4'][-1,i+1]
        for k in sd.iterkeys():
            plt.loglog( hs[k], [np.abs(y-best) for y in sd[k][:,i+1]],'+-',label=k)
            # plt.loglog( hsconv[o], [np.abs(y-best) for y in datconv[:,i+1]],'+',color=c)
        plt.legend()
    plt.show()
make_error_plots(results, results, labels)
embed()
