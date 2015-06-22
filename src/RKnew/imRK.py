import numpy as np
from dolfin import *

from RKbase import *

from IPython import embed

"""
What paper did I get this one from?
"""

alpha_2 = 1.0-0.5*sqrt(2.0)

alpha = 0.43586652150845899942
tau = (1.0+alpha)/2.0
b1 = -(6.0*alpha*alpha - 16.0*alpha+1.0)/4.0
b2 = (6.0*alpha*alpha - 20.0*alpha+5.0)/4.0
                
LDIRK = {
    'BWEuler' : {
        'a':np.array([ [1.0]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 1.0 ], dtype=np.double)
    },
    'LSDIRK2' : {
        'a':np.array([ [alpha_2, 0.0],
                       [1.0-alpha_2, alpha_2] ], dtype=np.double),
        'b':np.array([ alpha_2, 1.0 ], dtype=np.double),
        'c':np.array([ alpha_2, 1.0 ], dtype=np.double)
    },
    'LSDIRK3' : {
        'a': np.array([ [ alpha, 0.0, 0.0 ],
                        [ tau-alpha, alpha, 0.0 ],
                        [ b1, b2, alpha ] ], dtype=np.double),
        'b':np.array( [ b1, b2, alpha ], dtype=np.double),
        'c': np.array( [ alpha, tau, 1.0 ], dtype=np.double)
    }
}

class DIRK(RKbase):
    """
    Diagonally Implicit Runge-Kuttas
    When doing an LDIRK, it can handle lots of DAEs.
    When doing one with asj != bj, it can only gauruntee do Semi-Explicit.
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
            if f.M==None:
                f.M = np.array([[1.0]],dtype=np.double)
        for i in xrange(len(RK_c)):
            self.DPRINT( " Stage ",i," at ",RK_c[i]," with aii=",RK_a[i,:] )
            aii = float(RK_a[i,i])
            
            # One time set up
            for f in self.ex_fields:
                f.Rhat[:] = f.M*f.u0[0][:]
                if f.order == 2:
                    f.uhat[1][:] = f.u0[1][:]
                for j in xrange(i):
                    f.Rhat[:] += h*RK_a[i,j]*f.ks[j][:]
                    if f.order == 2:
                        f.uhat[1][:] += h*RK_a[i,j]*f.vs[j][:]
            # The outer field iteration
            alldone = False
            itout = 0
            maxout = 100 if (len(self.im_fields)+len(self.ex_fields))>1 else 1
            
            while not alldone and itout < maxout:
                alldone = True
                # Iterate over each of the fileds
                for f in  self.im_fields + self.ex_fields :
                    self.DPRINT( " Solving field of order ",f.order)
                    eps = 1.0
                    tol = 1.0e-10
                    maxiter = 10
                    itcnt = 0

                    # Newton solve this field
                    while eps>tol and itcnt < maxiter:
                        self.DPRINT("  Solving...")
                        # Assemble
                        if f.order==0:
                            R,K = f.sys(time,True)
                        elif f.order==1:
                            F,AU = f.sys(time,True)
                            K = f.M - h*aii*AU
                            R = f.Rhat - f.M*f.u[0] + h*aii*F
                        elif f.order==2:
                            F,AX,AV = f.sys(time,True)
                            K = f.M - h*h*aii*aii*AX - h*aii*AV
                            R = f.Rhat - f.M*f.u[0] + h*aii*F
                        else:
                            print "Unknown order type ",f.order
                            raise
                        # Apply BCs to matrix
                        f.bcapp(K,R, time+h*RK_c[i],itcnt!=0)
                        self.DPRINT( "   Solving Matrix... ")
                        # Solve the Matrix
                        # embed()
                        if type(K) is GenericMatrix or type(K) is Matrix:
                            solve(K,f.DU[0],R)
                            eps = np.linalg.norm(f.DU[0].array(), ord=np.Inf)
                        else:
                            f.DU[0][:] = 1.0/K[0,0]*R
                            eps = np.abs(f.DU[0])
                        # embed()

                        self.DPRINT( "  ",itcnt," Norm:", eps)
                        if np.isnan(eps):
                            print "Hit a Nan! Quitting"
                            raise

                        # Apply the Newton steps
                        if f.order==0:
                            f.u[0][:] = f.u[0][:] - f.DU[0][:]
                        elif f.order==1:
                            f.u[0][:] = f.u[0][:] + f.DU[0][:]
                        else: # 2
                            f.u[0][:] = f.u[0][:] + f.DU[0][:]
                            f.u[1][:] = f.uhat[1][:] + h*aii*f.u[0][:]
                        f.update()
                        itcnt += 1
                    # end newton iteration
                    if itcnt > 1:
                        alldone = False
                # end field loop
                itout += 1
            # end coupling iteration
            print "Took ", itout, " coupling iterations"
            for f in self.ex_fields:
                f.ks.append(f.sys(time,False))
                if f.order == 2:
                    f.vs.append(f.u[0].copy())
        # end stage loop
