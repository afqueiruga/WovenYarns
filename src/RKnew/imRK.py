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

gamma = 0.5 + np.sqrt(3.0)/3.0*np.cos(np.pi/18.0)

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
    },
    
    'ImTrap' : {
        'a':np.array([ [0.5]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 0.5 ], dtype=np.double)
    },
    'DIRK2' : {
        'a':np.array([ [0.5+1.0/(2.0*np.sqrt(3.0)), 0.0],
                       [-1.0/np.sqrt(3.),           0.5+1.0/(2.0*np.sqrt(3.0))] ], dtype=np.double),
        'b':np.array([  0.5,                        0.5 ], dtype=np.double),
        'c':np.array([ 0.5+1.0/(2.0*np.sqrt(3.0)),  0.5-1.0/(2.0*np.sqrt(3.0)) ], dtype=np.double)
    },
    'DIRK3' : {
        'a': np.array([ [ gamma, 0.0, 0.0 ],
                        [ 0.5-gamma, gamma, 0.0 ],
                        [ 2.0*gamma, 1.0-4.0*gamma, gamma ] ], dtype=np.double),
        'b':np.array( [ 1.0/(24.0*(0.5-gamma)**2.),
                        1.0-1.0/(12.0*(0.5-gamma)**2.),
                        1.0/(24.0*(0.5-gamma)**2.) ], dtype=np.double),
        'c': np.array( [ gamma,      0.5,         1.0-gamma ], dtype=np.double)
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
                f.Mbc = np.array([[1.0]],dtype=np.double)
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

        # Do the last recombination if we need to
        if self.LSTABLE:
            return
        # Do the final Mv=sum bk
        for f in self.ex_fields:
            for s,v in zip(f.u,f.u0):
                s[:] = v[:]
            f.DU[0][:]=0.0
            for j in xrange(len(RK_b)):
                f.DU[0][:] += h*RK_b[j]*f.ks[j][:] # Need to solve matrix
                if f.order == 2:
                    f.u[1][:] += h*RK_b[j]*f.vs[j][:]
            if f.M!=None:
                f.u[0][:]=0.0
                if type(f.Mbc) is Matrix or type(f.Mbc) is GenericMatrix:
                    solve(f.Mbc,f.u[0],f.DU[0])
                else:
                    f.u[0][:] = 1.0/f.Mbc[0,0] * f.DU[0][:]
                # f.bcapp(None,f.u[0],time+h,False)
                f.u[0][:] += f.u0[0][:]
            else:
                # f.bcapp(None,f.DU[0],time+h,False)
                f.u[0][:] += f.DU[0][:]
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
                if type(K) is Matrix:
                    solve(K,f.DU[0],F)
                    eps = np.linalg.norm(f.DU[0].array(), ord=np.Inf)
                else:
                    f.DU[0][:] = 1.0/K[0,0]*F
                    eps = np.abs(f.DU[0])
                self.DPRINT( "  ",itcnt," Norm:", eps)
                if np.isnan(eps):
                    print "Hit a Nan! Quitting"
                    raise
                f.u[0][:] = f.u[0][:] - f.DU[0][:]
                f.update()
                itcnt += 1
        
