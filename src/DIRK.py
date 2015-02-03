#!/usr/bin/python

from dolfin import *
import numpy as np

from RK import RK

alpha_2 = 1.0-0.5*sqrt(2.0)

alpha = 0.43586652150845899942
tau = (1.0+alpha)/2.0
b1 = -(6.0*alpha*alpha - 16.0*alpha+1.0)/4.0
b2 = (6.0*alpha*alpha - 20.0*alpha+5.0)/4.0
                
LDIRK = {
    1 : {
        'a':np.array([ [1.0]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 1.0 ], dtype=np.double)
    },
    2 : {
        'a':np.array([ [alpha_2, 0.0],
                       [1.0-alpha_2, alpha_2] ], dtype=np.double),
        'b':np.array([ alpha_2, 1.0 ], dtype=np.double),
        'c':np.array([ alpha_2, 1.0 ], dtype=np.double)
    },
    3 : {
        'a': np.array([ [ alpha, 0.0, 0.0 ],
                        [ tau-alpha, alpha, 0.0 ],
                        [ b1, b2, alpha ] ], dtype=np.double),
        'b':np.array( [ b1, b2, alpha ], dtype=np.double),
        'c': np.array( [ alpha, tau, 1.0 ], dtype=np.double)
    }
}

class DIRK_Monolithic(RK):
    def march(self,time=0.0):
        h = self.h
        x = self.x
        v = self.v
        X0 = self.X0
        V0 = self.V0
        Xhat = self.Xhat
        DX = self.DX
        RK_a = self.RK_a
        RK_b = self.RK_b
        RK_c = self.RK_c
        M = self.M
        
        V0[:] = v[:]
        X0[:] = x[:]
        ks = []
        vs = []
        for i in xrange(len(RK_c)):
            print " Stage ",i," at ",RK_c[i]," with aii=",RK_a[i,i]
            eps = 1.0
            tol = 1.0e-10
            maxiter = 10
            itcnt = 0
            Rhat = M*V0[:]
            Xhat[:] = X0[:]
            aii = float(RK_a[i,i])
            for j in xrange(i):
                Rhat[:] += h*RK_a[i,j]*ks[j][:]
                Xhat[:] += h*RK_a[i,j]*vs[j][:]
            while eps>tol and itcnt < maxiter:
                print "  Assembling..."
                F,AX,AV = self.sys(time)
                K = M - h*h*aii*aii*AX - h*aii*AV
                R = Rhat - M*v + h*aii*F
                self.bcapp(K,R,time+h*RK_c[i],itcnt!=0)

                print "  Solving..."
                solve(K,DX,R)
                eps=np.linalg.norm(DX.array(), ord=np.Inf)
                print "  ",itcnt," Norm:", eps
                v[:] = v[:] + DX[:]
                x[:] = Xhat[:] + h*aii*v[:]
                self.update()
                itcnt += 1
            ks.append(R)
            vs.append(v.copy()) # TODO: Get rid of this copy

