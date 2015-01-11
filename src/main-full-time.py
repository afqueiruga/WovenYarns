#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp
from DIRK import DIRK_Monolithic

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
warp.assemble_system()

Tmax = 100.0
weval = np.zeros(11)
all_series = []
hs = [2.0,3.0,4.0]

for h in hs:
    NT = int(Tmax/h)
    time_series = np.zeros(NT+1)
    times = np.zeros(NT+1)
    for i,fib in enumerate(warp.fibrils):
        fib.wx.interpolate(Expression(("0.0","0.0","0.0", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0")))
        fib.wv.interpolate(Expression(("0.0","0.0","0.0", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "x[0]/100.0")))
        warp.wx.vector()[ warp.mdof.part(i).dofs() ] = fib.wx.vector()[:]
        warp.wv.vector()[ warp.mdof.part(i).dofs() ] = fib.wv.vector()[:]
        time_series[0] = 0.0
    dirk = DIRK_Monolithic(warp, warp.assemble_system, warp.update, apply_BCs, h)
    # embed()
    for t in xrange(NT):
        dirk.march()

        warp.fibrils[0].wx.eval(weval,np.array([0.0,0.0,-1.0],dtype=np.double))
        time_series[t+1] = weval[1]
        times[t+1] = (t+1)*h
        # warp.output_states("../post/fibril_time_{0}_"+str(t)+".pvd",1)
    all_series.append((times,time_series))

for ts,ys in all_series:
    plt.plot(ts,ys)
plt.show()
embed()
