#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [0.0,-1.0,0.0],[0.0, 1.0,0.0] ],
           [ [-0.71,-0.71,0.1],[0.71, 0.71,0.1] ],
           [ [0.0,-1.0,0.2],[0.0, 1.0,0.2] ] ]

warp = Warp(endpts)
warp.create_contacts()

warp.assemble_system()
warp.apply_bcs()

w = MultiMeshFunction(warp.mmfs)
solve(warp.AX,w.vector(),warp.R)
# embed()
# for i,fib in enumerate(warp.fibrils):
#     fib.wx.vector()[:] = w.part(i).vector()[:]

print "here"
for i,fib in enumerate(warp.fibrils):
    pass
    # assign(fib.wx, w.part(i))
print "there"
embed()
warp.output_states("../post/fibril_{0}_.pvd",1)
warp.output_contacts("../post/contact_{2}_{0}_{1}.pvd")

        
embed()
