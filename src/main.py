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
for t in xrange(1,10):
    it=0
    eps=1.0
    print "Simulation step ",t,":"
    extend = Expression(("0.0",
                         "-(x[1]*cos(theta)-x[2]*sin(theta)-x[1]-(x[1]*cos(old_theta)-x[2]*sin(old_theta)-x[1]))",
                         "-(x[2]*cos(theta)+x[1]*sin(theta)-x[2]-(x[2]*cos(old_theta)+x[1]*sin(old_theta)-x[2]))",
                         "0.0","0.0","0.0", "0.0","0.0","0.0"),theta=t*np.pi/8.0,old_theta=(t-1)*np.pi/8.0)
    while eps>tol and it < maxiter:
        
        warp.assemble_system()
        warp.apply_bcs(uend=(extend if it==0 else None))
        solve(warp.AX,Delw.vector(),warp.R)
        
        eps = np.linalg.norm(Delw.vector().array(), ord=np.Inf)
        for i,fib in enumerate(warp.fibrils):
            fib.wx.vector()[:] = fib.wx.vector()[:] - Delw.vector()[ warp.mdof.part(i).dofs() ]
        print " Newton iteration ",it," infNorm = ",eps, "  ",(it!=0)
        it+=1

    warp.output_states("../post/fibril_{0}_"+str(t)+".pvd",1)
    warp.output_contacts("../post/contact_{2}_{0}_{1}.pvd")

embed()
