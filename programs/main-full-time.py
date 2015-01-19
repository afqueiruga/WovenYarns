#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from src import Warp
from src.DIRK import DIRK_Monolithic

endpts = [ [ [-10.0, 0.0,-1.0],  [10.0, 0.0, -1.0] ] ]

warp = Warp(endpts)


zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
extend = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))

left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -10.0)

right = CompiledSubDomain("near(x[0], side) && near(x[2], -1.0) && on_boundary", side = 10.0)


def apply_BCs(K,R,hold=False):
    bcleft = MultiMeshDirichletBC(warp.mmfs, zero, left)
    bcleft.apply(K,R)
    if not hold:
        bcright = MultiMeshDirichletBC(warp.mmfs, extend, right)
    else:
        bcright = MultiMeshDirichletBC(warp.mmfs, zero, right)
    bcright.apply(K,R)


# Assemble once to get M
warp.assemble_mass()

Tmax = 40.0

probes = [ (np.array([0.0,0.0,-1.0],dtype=np.double),1,'x'),
            (np.array([5.0,0.0,-1.0],dtype=np.double),2,'x'),
            (np.array([5.0,0.0,-1.0],dtype=np.double),9,'v'),
            (np.array([5.0,0.0,-1.0],dtype=np.double),10,'v')]
weval = np.zeros(11)

def solve(order,NT):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    
    for i,fib in enumerate(warp.fibrils):
        fib.wx.interpolate(Expression(("0.0","0.0","0.0", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0")))
        fib.wv.interpolate(Expression(("0.0","0.0","0.0", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "x[0]/100.0")))
        warp.wx.vector()[ warp.mdof.part(i).dofs() ] = fib.wx.vector()[:]
        warp.wv.vector()[ warp.mdof.part(i).dofs() ] = fib.wv.vector()[:]
        time_series[0] = 0.0
    
    dirk = DIRK_Monolithic(order, h, warp, warp.assemble_system, warp.update, apply_BCs)

    for t in xrange(NT):
        dirk.march()
        for g,p in enumerate(probes):
            if p[2]=='x':
                warp.fibrils[0].wx.eval(weval,p[0])
            else:
                warp.fibrils[0].wv.eval(weval,p[0])
            time_series[t+1,g] = weval[p[1]]
        times[t+1] = (t+1)*h
        # warp.output_states("post/fibril_time_{0}_"+str(t)+".pvd",1)
    return (times,time_series,h,order)


NTS = [50] #[3000,4000,5000]#[50,100,250,500,1000,2000]
orders = [1] #,2,3]
all_series = [ ]

for order in orders:
    results = []
    for NT in NTS:
        results.append( solve(order,NT) )
    all_series.append( (order, results) )
embed()

def make_plots(all_series):
    for g in xrange(len(probes)):
        plt.figure()
        for series in all_series:
            for ts,ys,o,h in series[1]:
                plt.plot(ts,ys[:,g])
        plt.figure()
        exact = all_series[-1][1][-1][1][-1,g]
        for series in all_series:
            plt.loglog([ x[2] for x in series[1]],
                        [ np.abs(x[1][-1,g]-exact) for x in series[1] ],'-+')

def compute_convergence(all_series):
    import scipy.stats
    for g in xrange(len(probes)):
        exact = all_series[-1][1][-1][1][-1,g]
        for ix,series in enumerate(all_series):
        
            print (scipy.stats.linregress([ np.log(x[2]) for x in series[1][:(-1 if ix==len(all_series)-1 else -2)] ],
                                                [ np.log(np.abs(x[1][-1,g]-exact)) for x in series[1][:(-1 if ix==len(all_series)-1 else -2)] ]))[0],
        print ""
        
#for g in xrange(len(probes)):
#    plt.figure()
#   plt.plot([ x[0][1]-x[0][0] for x in all_series],
#             [ x[1][-1,g] for x in all_series ])

plt.show()
import cPickle
cPickle.dump(all_series,open("convergence1.p","wb"))
embed()
