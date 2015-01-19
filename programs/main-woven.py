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

pattern = [ [-0.5,sqrt(3.0)/2.0], [0.5,sqrt(3.0)/2.0],
    [-1.0,0.0], [0.0, 0.0], [1.0,0.0],
    [-0.5,-sqrt(3.0)/2.0], [ 0.5,-sqrt(3.0)/2.0] ]

endpts = []

scale = 0.15
for l in pattern:
    endpts.append([ [ -2.5+scale*l[0], -5.0, scale*l[1] ], [ -2.5+scale*l[0], 5.0, scale*l[1] ] ])
for l in pattern:
    endpts.append([ [ 2.5+scale*l[0], -5.0, scale*l[1] ], [ 2.5+scale*l[0], 5.0, scale*l[1] ] ])
for l in pattern:
    endpts.append([ [  -5.0, scale*l[0]+2.5,scale*l[1] ], [  5.0, scale*l[0]+2.5, scale*l[1] ] ])
for l in pattern:
    endpts.append([ [  -5.0, scale*l[0]-2.5,scale*l[1] ], [  5.0, scale*l[0]-2.5, scale*l[1] ] ])

# endpts = [
#     [ [ 1.0,-10.0, -0.1],  [ 1.0,10.0,  -0.1] ],
#     [ [ -3.0,-10.0, -0.1],  [ -3.0,10.0,  -0.1] ],
#     [ [-10.0, 0.05,-0.05],  [10.0, 0.05, -0.05] ] ,
#     [ [-10.0, -0.06,-0.05],  [10.0, -0.06, -0.05] ] ]

warp = Warp(endpts, cutoff=0.5)


zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
extend = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))

left = CompiledSubDomain("on_boundary", side = -10.0)

right = CompiledSubDomain("near(x[0], side) && near(x[2], -1.0) && on_boundary", side = 10.0)


def apply_BCs(K,R,hold=False):
    # embed()
    bcleft = MultiMeshDirichletBC(warp.mmfs, zero, left)
    bcleft.apply(K,R)
    if not hold:
        bcright = MultiMeshDirichletBC(warp.mmfs, extend, right)
    else:
        bcright = MultiMeshDirichletBC(warp.mmfs, zero, right)
    bcright.apply(K,R)

# Assemble once to get M
warp.assemble_mass()
probes = [ (np.array([0.0,0.05,-0.05],dtype=np.double),1,'x'),
            (np.array([5.0,0.05,-0.05],dtype=np.double),2,'x'),
            (np.array([5.0,0.05,-0.05],dtype=np.double),9,'v'),
            (np.array([5.0,0.05,-0.05],dtype=np.double),10,'v')]
weval = np.zeros(11)
Tmax=5.0

def solve(order,NT):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    for i,fib in enumerate(warp.fibrils):
        if i<7:
            fib.wx.interpolate(Expression(("0.0","0.0","(1.5*scale)*sin(x[1]*PI/5.0)", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0"),PI=np.pi,scale=-scale))
        elif i<14:
            fib.wx.interpolate(Expression(("0.0","0.0","(1.5*scale)*sin(x[1]*PI/5.0)", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0"),PI=np.pi,scale=scale))
        elif i<21:
            fib.wx.interpolate(Expression(("0.0","0.0","-(1.5*scale)*sin(x[0]*PI/5.0)", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0"),PI=np.pi,scale=scale))
        else:
            fib.wx.interpolate(Expression(("0.0","0.0","(1.5*scale)*sin(x[0]*PI/5.0)", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0"),PI=np.pi,scale=scale))
        fib.wv.interpolate(Expression(("0.0","0.0","0.0", "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                            "0.0", "0.0")))# "x[0]/100.0")))
        warp.wx.vector()[ warp.mdof.part(i).dofs() ] = fib.wx.vector()[:]
        warp.wv.vector()[ warp.mdof.part(i).dofs() ] = fib.wv.vector()[:]
        time_series[0] = 0.0
    warp.create_contacts()
    # warp.output_contacts("../post/contact_{2}_{0}_{1}.pvd")
    dirk = DIRK_Monolithic(order, h, warp, warp.assemble_system, warp.update, apply_BCs)
    warp.output_states("../post/fibril2_time_{0}_"+str(0)+".pvd",1)
    warp.output_surfaces("../post/fibril2mesh_time_{0}_"+str(0)+".pvd",1)
    for t in xrange(NT):
        dirk.march()
        # for g,p in enumerate(probes):
        #     if p[2]=='x':
        #         warp.fibrils[2].wx.eval(weval,p[0])
        #     else:
        #         warp.fibrils[2].wv.eval(weval,p[0])
        #     time_series[t+1,g] = weval[p[1]]
        times[t+1] = (t+1)*h
        warp.output_states("../post/fibril2_time_{0}_"+str(t+1)+".pvd",1)
        warp.output_surfaces("../post/fibril2mesh_time_{0}_"+str(t+1)+".pvd",1)

    return (times,time_series,h,order)

solve(1,100)

embed()
