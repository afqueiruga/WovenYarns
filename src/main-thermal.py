#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [-0.9, -1.0,-0.1],  [-0.9, 1.0, -0.1] ],
           [ [0.9, -1.0,-0.1],  [0.9, 1.0, -0.1] ],
           # [ [-1.0, 0.0,0.0],   [1.0, 0.0, 0.0]  ],
           [ [-1.0, 0.1,0.0],   [1.0, 0.1, 0.0]  ] ]

warp = Warp(endpts,cutoff=0.3)
warp.create_contacts([(0,2), (1,2)])

DelT = MultiMeshFunction(warp.Tmmfs)

warp.assemble_thermal_system()
warp.apply_thermal_bcs(Constant(1.0))

solve(warp.AT,DelT.vector(),warp.RT)
for i,fib in enumerate(warp.fibrils):
    fib.T.vector()[:] =  DelT.vector()[ warp.Tmdof.part(i).dofs() ]
warp.output_states("../post/fibril_{0}_temp.pvd",1)
warp.output_contacts("../post/contact_{2}_{0}_{1}.pvd")
embed()
