#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [-10.0, 0.0,-1.0],  [10.0, 0.0, -1.0] ] ]

warp = Warp(endpts)

Delw = MultiMeshFunction(warp.mmfs)

warp.assemble_system()

embed()

maxiter = 10
tol = 1.0e-9
it=0
eps=1.0
warp.create_contacts()
while eps>tol and it < maxiter:        
    warp.assemble_system()
    warp.AX.ident_zeros()
    warp.apply_multi_bcs(uend=None)
    solve(warp.AX,Delw.vector(),warp.R)
        
    eps = np.linalg.norm(Delw.vector().array(), ord=np.Inf)
    for i,fib in enumerate(warp.fibrils):
        fib.wx.vector()[:] = fib.wx.vector()[:] - Delw.vector()[ warp.mdof.part(i).dofs() ]
    print " Newton iteration ",it," infNorm = ",eps, "  ",(it!=0)
    it+=1

embed()
