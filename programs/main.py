#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from src import Warp

endpts = [ [ [-10.0, 0.0,-0.149],  [10.0, 0.0, -0.149] ],
           # [ [-1.0, 0.0,0.0],  [1.0, 0.0, 0.0] ] ,
           [ [-10.0, 0.0,0.0],  [10.0, 0.0, 0.0] ] ]

warp = Warp(endpts,monolithic=False,cutoff=1.5)

Delw = MultiMeshFunction(warp.mmfs)
embed()
maxiter = 10
tol = 1.0e-12
Nsteps = 64
for t in xrange(1,Nsteps):
    warp.create_contacts()
    # embed()
    it=0
    eps=1.0
    print "Simulation step ",t,":"
    extend = Expression((" 0.0",
                         "-(x[1]*cos(theta)-x[2]*sin(theta)-x[1]-(x[1]*cos(old_theta)-x[2]*sin(old_theta)-x[1]))",
                         "-(x[2]*cos(theta)+x[1]*sin(theta)-x[2]-(x[2]*cos(old_theta)+x[1]*sin(old_theta)-x[2]))",
                         "0.0","0.0","0.0", "0.0","0.0","0.0"),theta=t*np.pi/float(Nsteps),old_theta=(t-1)*np.pi/float(Nsteps))
    while eps>tol and it < maxiter:
        
        warp.assemble_system()
        warp.apply_bcs(uend=(extend if it==0 else None))
        solve(warp.AX,Delw.vector(),warp.R)
        
        eps = np.linalg.norm(Delw.vector().array(), ord=np.Inf)
        for i,fib in enumerate(warp.fibrils):
            fib.wx.vector()[:] = fib.wx.vector()[:] - Delw.vector()[ warp.mdof.part(i).dofs() ]
        print " Newton iteration ",it," infNorm = ",eps, "  ",(it!=0)
        it+=1

    warp.output_states("post/fibril_{0}_"+str(t)+".pvd",1)
    warp.output_contacts("post/contact_{2}_{0}_{1}.pvd")
    warp.output_surfaces("post/fibrilmesh_{0}_"+str(t+1)+".pvd",1)

embed()
