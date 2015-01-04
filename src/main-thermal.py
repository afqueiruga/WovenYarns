#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [0.0, -1.0,-0.1],  [0.0, 1.0, -0.1] ],
           [ [-1.0, 0.0,0.0],  [1.0, 0.0, 0.0] ] ]

warp = Warp(endpts)
warp.create_contacts()

DelT = MultiMeshFunction(warp.Tmmfs)

warp.assemble_thermal_system()
warp.apply_thermal_bcs(Constant(1.0))

solve(warp.AT,DelT.vector(),warp.RT)
for i,fib in enumerate(warp.fibrils):
    fib.T.vector()[:] =  DelT.vector()[ warp.Tmdof.part(i).dofs() ]
warp.output_states("../post/fibril_{0}_temp.pvd",1)
embed()
