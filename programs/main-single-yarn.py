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
Rate = 0.1
Tmax = 1.0
NT = 100

#
# Define the end points
#
pattern = [ [-0.5,sqrt(3.0)/2.0], [0.5,sqrt(3.0)/2.0],
    [-1.0,0.0], [0.0, 0.0], [1.0,0.0],
    [-0.5,-sqrt(3.0)/2.0], [ 0.5,-sqrt(3.0)/2.0] ]

endpts = []
scale = 0.15
for l in pattern:
    endpts.append([ [ scale*l[0], -5.0, scale*l[1] ], [ scale*l[0], 5.0, scale*l[1] ] ])

warp = Warp(endpts, cutoff=0.5)


#
# Set up the BC applying routines
#
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
rotate = Expression((" 0.0",
                    "-(x[1]*cos(theta)-x[2]*sin(theta)-x[1]-(x[1]*cos(old_theta)-x[2]*sin(old_theta)-x[1]))",
                    "-(x[2]*cos(theta)+x[1]*sin(theta)-x[2]-(x[2]*cos(old_theta)+x[1]*sin(old_theta)-x[2]))",
                    "0.0","0.0","0.0", "0.0","0.0","0.0"),
                    theta=0.0,
                    old_theta=0.0)

left = CompiledSubDomain("near(x[1], side) && on_boundary", side = -10.0)
right = CompiledSubDomain("near(x[1], side) && on_boundary", side = 10.0)
bcleft = MultiMeshDirichletBC(warp.mmfs, zero, left)
bcright =  MultiMeshDirichletBC(warp.mmfs, rotate, right)
def apply_BCs(K,R,t,hold=False):
    rotate.theta = Rate*t*np.pi,
    rotate.old_theta=Rate*(t-1.0)*np.pi
    bcleft.apply(K,R)
    if not hold:
        bcright = MultiMeshDirichletBC(warp.mmfs, extend, right)
    else:
        bcright = MultiMeshDirichletBC(warp.mmfs, zero, right)
    bcright.apply(K,R)


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
warp.create_contacts()
warp.output_states("post/yarn_time_{0}_"+str(0)+".pvd",1)
warp.output_surfaces("post/yarnmesh_time_{0}_"+str(0)+".pvd",1)
for t in xrange(NT):
    dirk.march()
    warp.output_states("post/yarn_time_{0}_"+str(t+1)+".pvd",1)
    warp.output_surfaces("post/yarnmesh_time_{0}_"+str(t+1)+".pvd",1)

# Give me a terminal at the end so I can play around
embed()
