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


warp = Warp(endpts, cutoff=1.5)

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("on_boundary")
subs = MultiMeshSubSpace(warp.mmfs,0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
    bcall.apply(K,R)


def initialize():
    for i,fib in enumerate(warp.fibrils):
        fib.wx.interpolate(Expression((

            "x[0]*sq + o*sin(x[0]*p)",
            "A1*cos((x[0]*p)/2.)",
            "A2*cos(x[0]*p)",
            "(-4*o*pow(p,2)*R*sin(x[0]*p))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2) + 16*(pow(A2,2)*pow(cos(x[0]*p),2) + pow(o,2)*pow(sin(x[0]*p),2))))",
            "R*(-1 - (A1*pow(p,2)*cos((x[0]*p)/2.))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2) + 16*(pow(A2,2)*pow(cos(x[0]*p),2) + pow(o,2)*pow(sin(x[0]*p),2)))))",
            "(-4*A2*pow(p,2)*R*cos(x[0]*p))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2) + 16*(pow(A2,2)*pow(cos(x[0]*p),2) + pow(o,2)*pow(sin(x[0]*p),2))))",
            "(-2*A1*A2*pow(p,3)*R*pow(sin((x[0]*p)/2.),3))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2)*pow(1 + 2*o*p + sq - o*p*cos(x[0]*p),2) + 16*pow(A2,2)*pow(o*p + (1 + sq)*cos(x[0]*p),2) + 4*pow(A1,2)*pow(A2,2)*pow(p,2)*pow(sin((x[0]*p)/2.),6)))",
            "(4*A2*pow(p,2)*R*(o*p + (1 + sq)*cos(x[0]*p)))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2)*pow(1 + 2*o*p + sq - o*p*cos(x[0]*p),2) + 16*pow(A2,2)*pow(o*p + (1 + sq)*cos(x[0]*p),2) + 4*pow(A1,2)*pow(A2,2)*pow(p,2)*pow(sin((x[0]*p)/2.),6)))",
            "R*(-1 + (A1*pow(p,2)*cos((x[0]*p)/2.)*(-1 - 2*o*p - sq + o*p*cos(x[0]*p)))/sqrt(pow(p,4)*(pow(A1,2)*pow(cos((x[0]*p)/2.),2)*pow(1 + 2*o*p + sq - o*p*cos(x[0]*p),2) + 16*pow(A2,2)*pow(o*p + (1 + sq)*cos(x[0]*p),2) + 4*pow(A1,2)*pow(A2,2)*pow(p,2)*pow(sin((x[0]*p)/2.),6))))",

                                        "0.0", "0.0"),
                                        R = 1.0,#warp.fibrils[0].radius,
                                        sq = -(restL-width)/restL,
                                        p=np.pi/width *4.0,
                                        o= width/8.0,
                                        A1=2.1*scale,
                                        A2=1.25*scale))
                # fib.wx.interpolate(Expression(("o*sin(p*x[0])+(sq)*x[0]",
                #                        "A1*cos(x[0]/2.0*p)",
                #                        "A2*cos(x[0]*p)",
                                       
                #                         "0.0"," 0.0","0.0",

                #                         "0.0","0.0","0.0",
                                        
                #                         "0.0", "0.0"),
                #                         sq = -(restL-width)/restL,
                #                         p=np.pi/width *4.0,
                #                         o= width/8.0,
                #                         A1=2.5*scale,
                #                         A2=1.25*scale))

        warp.wx.vector()[ warp.mdof.part(i).dofs() ] = fib.wx.vector()[:]
        warp.wv.vector()[ warp.mdof.part(i).dofs() ] = fib.wv.vector()[:]

warp.output_states("post/stockinette_{0}_"+str(0)+".pvd",1)
warp.output_surfaces("post/stockinettemesh_time_{0}_"+str(0)+".pvd",1)
initialize()                             
warp.output_states("post/stockinette_{0}_"+str(1)+".pvd",1)
warp.output_surfaces("post/stockinettemesh_time_{0}_"+str(1)+".pvd",1)


Tmax=4.0
NT = 100
h = Tmax/NT
warp.create_contacts()
warp.assemble_mass()
dirk = DIRK_Monolithic(1, h, warp, warp.assemble_system, warp.update, apply_BCs)
for t in xrange(NT):
    dirk.march()

    warp.output_states("post/stockinette_time_{0}_"+str(t+2)+".pvd",1)
    warp.output_surfaces("post/stockinettemesh_time_{0}_"+str(t+2)+".pvd",1)

embed()
