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

#
# Some parameters
#
Rate = 0.01*np.pi
Tmax = 200.0
NT = 100

#
# Define the end points
#
pattern = [ [-0.5,sqrt(3.0)/2.0], [0.5,sqrt(3.0)/2.0],
        [-1.0,0.0], [0.0, 0.0], [1.0,0.0],
        [-0.5,-sqrt(3.0)/2.0], [ 0.5,-sqrt(3.0)/2.0] ]

endpts = []
scale = 0.16
for l in pattern:
    endpts.append([ [ -5.0,  scale*l[0], scale*l[1] ], [ 5.0, scale*l[0], scale*l[1] ] ])

warp = Warp(endpts, monolithic=True, cutoff=0.5)


#
# Set up the BC applying routines
#
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
# rotate = Expression((" 0.0",
#                     " 0.0", " rate",
#                     "0.0","0.0","0.0", "0.0","0.0","0.0",  "0.0","0.0"),
#                     rate=Rate,
#                     theta=0.0,
#                     old_theta=0.0)
rotate = Expression((" 0.0",
                    "-rate*(x[1]*sin(theta)+x[2]*cos(theta))",
                    "-rate*(x[2]*sin(theta)-x[1]*cos(theta))",
                    "0.0","0.0","0.0", "0.0","0.0","0.0", "0.0","0.0"),
                    rate=Rate,
                    theta=0.0,
                    old_theta=0.0)

left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -5.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 5.0)
bcleft = MultiMeshDirichletBC(warp.mmfs, zero, left)
bcright =  MultiMeshDirichletBC(warp.mmfs, rotate, right)
def apply_BCs(K,R,t,hold=False):
    rotate.theta = Rate*t
    rotate.old_theta=Rate*(t-1.0)
    bcleft.apply(K,R)
    if not hold:
        bcright = MultiMeshDirichletBC(warp.mmfs, rotate, right)
    else:
        bcright = MultiMeshDirichletBC(warp.mmfs, zero, right)
    bcright.apply(K,R)
    # embed() 

#
# Set up initial conditions
#
# Nope fuck it, they're zero anyways

#
# Create the timestepper
#
dirk = DIRK_Monolithic(1, Tmax/NT, warp, warp.assemble_system, warp.update, apply_BCs)


#
# Loop away
#
warp.assemble_mass()
warp.create_contacts()
warp.output_states("post/yarn_time_{0}_"+str(0)+".pvd",1)
warp.output_surfaces("post/yarnmesh_time_{0}_"+str(0)+".pvd",1)
for t in xrange(NT):
    dirk.march(t*Tmax/NT)
    warp.output_states("post/yarn_time_{0}_"+str(t+1)+".pvd",1)
    warp.output_surfaces("post/yarnmesh_time_{0}_"+str(t+1)+".pvd",1)

# Give me a terminal at the end so I can play around
embed()
