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

endpts = []

restL = 1.75
width = 1.5
scale = 0.15
for x in np.linspace(-width+scale,width-scale, width/scale):
    endpts.append([ [ -restL, x, 0.0 ], [ restL, x, 0.0 ] ])
NW = len(endpts)
for x in np.linspace(-width+scale,width-scale, width/scale):
    endpts.append([ [ x,-restL, 0.0 ], [ x, restL, 0.0 ] ])

warp = Warp(endpts, cutoff=0.8)



zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("on_boundary")
subs = MultiMeshSubSpace(warp.mmfs,0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
    bcall.apply(K,R)



def initialize():
    for i,fib in enumerate(warp.fibrils):
        if i<NW:
            fib.wx.interpolate(Expression(("(squeeze)*x[0]","0.0","(0.49*scale)*sin(x[0]*period)",
                                            "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                                            "0.0", "0.0"),squeeze = -(restL-width)/restL,period=np.pi/restL*NW/2.0,scale=(1.0 if i%2 else -1.0)*scale))
        else:
            fib.wx.interpolate(Expression(("0.0","(squeeze)*x[1]","(0.49*scale)*sin(x[1]*period)",
                                            "0.0"," 0.0","0.0", "0.0","0.0","0.0",
                                            "0.0", "0.0"),squeeze = -(restL-width)/restL,period=np.pi/restL*NW/2.0,scale=(1.0 if (i+1)%2 else -1.0)*scale))
        warp.wx.vector()[ warp.mdof.part(i).dofs() ] = fib.wx.vector()[:]
        warp.wv.vector()[ warp.mdof.part(i).dofs() ] = fib.wv.vector()[:]
warp.output_states("post/weave_{0}_"+str(0)+".pvd",1)
warp.output_surfaces("post/weavemesh_time_{0}_"+str(0)+".pvd",1)
initialize()                             
warp.output_states("post/weave_{0}_"+str(1)+".pvd",1)
warp.output_surfaces("post/weavemesh_time_{0}_"+str(1)+".pvd",1)


Tmax=2.0
NT = 100
h = Tmax/NT
warp.create_contacts()
warp.assemble_mass()
dirk = DIRK_Monolithic(1, h, warp, warp.assemble_system, warp.update, apply_BCs)
for t in xrange(NT):
    dirk.march()

    warp.output_states("post/weavemesh_time_{0}_"+str(t+2)+".pvd",1)
    warp.output_surfaces("post/weavemesh_time_{0}_"+str(t+2)+".pvd",1)

embed()
