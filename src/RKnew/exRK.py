import numpy as np
from dolfin import *

from RKbase import *

from IPython import embed

"""
Tableaus are from Butcher's Giant Book of Tableaus
"""
exRK_table = {
    '1' : {
        'a':np.array([ [0.0]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 0.0 ], dtype=np.double)
        },
    'RK2-trap': {
        'a':np.array([ [0.0,0.0],
                       [1.0,0.0] ], dtype=np.double),
        'b':np.array([ 0.5,0.5 ], dtype=np.double),
        'c':np.array([ 0.0,1.0 ], dtype=np.double)
        },
    'RK2-mid': {
        'a':np.array([ [0.0,0.0],
                       [0.5,0.0] ], dtype=np.double),
        'b':np.array([ 0.0,1.0 ], dtype=np.double),
        'c':np.array([ 0.0,0.5 ], dtype=np.double)
        },
    'RK3-1': {
        'a':np.array([ [0.0,    0.0,    0.0],
                       [2.0/3.0,0.0,    0.0],
                       [1.0/3.0,1.0/3.0,0.0] ], dtype=np.double),
        'b':np.array([ 0.25,0.0,0.75 ], dtype=np.double),
        'c':np.array([ 0.0,2.0/3.0,2.0/3.0 ], dtype=np.double)
        },
    'RK4' : {
        'a': np.array([ [ 0.0, 0.0, 0.0, 0.0 ],
                        [ 0.5, 0.0, 0.0, 0.0 ],
                        [ 0.0, 0.5, 0.0, 0.0 ],
                        [ 0.0, 0.0, 1.0, 0.0  ] ], dtype=np.double),
        'b':np.array( [ 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 ], dtype=np.double),
        'c': np.array( [ 0.0, 0.5, 0.5, 1.0 ], dtype=np.double)
        }
    }
            
class exRK(RKbase):
    """
    Explicit Runge-Kuttas
    Works on up to SemiExplicit Index-1 DAEs.
    """
    def march(self,time=0.0):
        h = self.h
        RK_a = self.RK_a
        RK_b = self.RK_b
        RK_c = self.RK_c
        
        for f in self.ex_fields:
            f.save_u0()
            f.ks = []
            if f.order == 2:
                f.vs = []
        
        for i in xrange(len(RK_c)):
            self.DPRINT( " Stage ",i," at ",RK_c[i]," with ai_=",RK_a[i,:] )
            # from IPython import embed
            # embed()
            # Step 1: Calculate values of explicit fields at this step
            for f in self.ex_fields:
                for s,v in zip(f.u,f.u0):
                    s[:] = v[:]
                for j in xrange(i):
                    if f.M!=None:
                        f.DU[0][:] += h*RK_a[i,j]*f.ks[j][:] # Need to solve matrix
                    else:
                        f.u[0][:] += h*RK_a[i,j]*f.ks[j][:] 
                    if f.order == 2:
                        f.u[1][:] += h*RK_a[i,j]*f.vs[j][:]
                if f.M!=None:
                    solve(f.M,f.u[0],f.DU[0])
                    f.u[0][:] += f.u0[0][:]
                f.update()
            # Step 2: Solve Implicit fields
            for f in self.im_fields:
                self.DPRINT( " Solving Implicit field... ")
                eps = 1.0
                tol = 1.0e-10
                maxiter = 10
                itcnt = 0
                while eps>tol and itcnt < maxiter:
                    self.DPRINT("  Solving...")
                    F,K = f.sys(time)
                    f.bcapp(K,F, time+h*RK_c[i],itcnt!=0)
                    self.DPRINT( "   Solving Matrix... ")
                    if K is Matrix:
                        solve(K,f.DU[0],F)
                        eps = np.linalg.norm(f.DU[0].array(), ord=np.Inf)
                    else:

                        f.DU[0][:] = 1.0/K[0,0]*F
                        eps = np.abs(f.DU[0])
                    # embed()
                    self.DPRINT( "  ",itcnt," Norm:", eps)
                    f.u[0][:] = f.u[0][:] + f.DU[0][:]
                    f.update()
                    itcnt += 1
            
            # Step 3: Compute k for each field
            for f in self.ex_fields:
                F = f.sys(time)
                f.bcapp(None,F,time+h*RK_c[i],False)
                f.ks.append(F)
                if f.order == 2:
                    f.vs.append(f.u[0].copy())

        # Do the final Mv=sum bk
        for f in self.ex_fields:
            for s,v in zip(f.u,f.u0):
                s[:] = v[:]
            for j in xrange(len(RK_b)):
                if f.M!=None:
                    f.DU[0][:] += h*RK_b[j]*f.ks[j][:] # Need to solve matrix
                else:
                    # embed()
                    f.u[0][:] += h*RK_b[j]*f.ks[j][:]
                if f.order == 2:
                    f.u[1][:] += h*RK_b[j]*f.vs[j][:]
            if f.M!=None:
                solve(K,f.u[0],f.DU[0])
                f.u[0][:] += f.u0[0][:]
            
            f.update()

        # Solve the implicit equation here
        # Step 2: Solve Implicit fields
        for f in self.im_fields:
            self.DPRINT( " Solving Implicit field... ")
            eps = 1.0
            tol = 1.0e-10
            maxiter = 10
            itcnt = 0
            while eps>tol and itcnt < maxiter:
                self.DPRINT("  Solving...")
                F,K = f.sys(time)
                f.bcapp(K,F,time+h*RK_c[i],itcnt!=0)
                self.DPRINT( "   Solving Matrix... ")
                if K is Matrix:
                    solve(K,f.DU[0],F)
                    eps = np.linalg.norm(f.DU[0].array(), ord=np.Inf)
                else:
                    f.DU[0][:] = 1.0/K[0,0]*F
                    eps = np.abs(f.DU[0])
                self.DPRINT( "  ",itcnt," Norm:", eps)
                f.u[0][:] = f.u[0][:] + f.DU[0][:]
                f.update()
                itcnt += 1
