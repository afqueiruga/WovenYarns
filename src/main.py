#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [-1.0, 0.0,0.0],  [1.0, 0.0, 0.0] ],
           [ [-1.0, 0.0,0.1],  [1.0, 0.0, 0.1] ] ,
           [ [-1.0, 0.0,0.2],  [1.0, 0.0, 0.2] ] ]

warp = Warp(endpts)
warp.create_contacts()

Delw = MultiMeshFunction(warp.mmfs)

maxiter = 10
tol = 1.0e-9
for t in xrange(1,2):
    it=0
    eps=1.0
    print "Simulation step ",t,":"
    while eps>tol and it < maxiter:
        
        warp.assemble_system()
        warp.apply_bcs(stheta=t*np.pi/4.0,hold=(it!=0))
        solve(warp.AX,Delw.vector(),warp.R)
        
        eps = np.linalg.norm(Delw.vector().array(), ord=np.Inf)
        for i,fib in enumerate(warp.fibrils):
            fib.wx.vector()[:] = fib.wx.vector()[:] - Delw.vector()[ warp.mdof.part(i).dofs() ]
        print " Newton iteration ",it," infNorm = ",eps, "  ",(it!=0)

        it+=1
        warp.output_states("../post/fibril_{0}_"+str(it)+".pvd",1)
        warp.output_contacts("../post/contact_{2}_{0}_{1}.pvd")

embed()
