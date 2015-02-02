"""
Make a warp and output it.

"""
from src import *
from src.Forms import MultiphysicsProblem

endpts = [ [[0.0,0.0,0.0],[1.0,0.0,0.0]] ,
           [[0.0,0.1,0.0],[1.0,0.1,0.0]] ,
           [[0.0,0.4,0.0],[1.0,0.4,0.0]] ]

defaults = { 'mu' : 10.0,
             'radius':0.2 }

props = [ { 'radius' : 0.1 },
            { 'mu' : 1.0 },
            {} ]

warp = Warp(endpts, props, defaults, MultiphysicsProblem)

warp.output_states("src/unit_tests/warp_{0}_.pvd",0)
warp.output_surfaces("src/unit_tests/warp_mesh_{0}_.pvd",0)

warp.create_contacts()
M = warp.assemble_form('AX','W')
from matplotlib import pylab as plt
plt.spy(M.array())
plt.show()
