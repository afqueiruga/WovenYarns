#!/usr/bin/python

from dolfin import *
import numpy as np

from Warp import Warp

class DIRK_Monolithic():
    def __init__(self,order,h,warp,sysass,update,bcapp):
        self.warp = warp
        self.sysass = sysass
        self.update = update
        self.bcapp = bcapp
        self.h = h

        self.X0 = MultiMeshFunction(warp.mmfs)
        self.V0 = MultiMeshFunction(warp.mmfs)
        self.DelW = MultiMeshFunction(warp.mmfs)
        if order==1:
            self.RK_b = np.array([ 1.0 ], dtype.np.double)
            self.RK_a = np.array([ [1.0]], dtype=np.double)
            self.RK_c = np.array([ 1.0 ], dtype.np.double)
        elif order==3:
            alpha = 0.43586652150845899942
            tau = (1.0+alpha)/2.0
            b1 = -(6.0*alpha*alpha - 16.0*alpha+1.0)/4.0
            b2 = (6.0*alpha*alpha - 20.0*alpha+5.0)/4.0
            self.RK_a = np.array([ [ alpha, 0.0, 0.0 ],
                                   [ tau-alpha, alpha, 0.0 ],
                                   [ b1, b2, alpha ] ], dtype=np.double)
            self.RK_b = np.array( [ b1, b2, alpha ], dtype=np.double)
            self.RK_c = np.array( [ alpha, tau, 1.0 ], dtype=np.double)
        else:
            RK_alpha = 1.0-0.5*sqrt(2.0)
            self.RK_b = np.array([ RK_alpha, 1.0 ])
            self.RK_a = np.array([ [RK_alpha, 0.0],
                  [1.0-RK_alpha, RK_alpha] ], dtype=np.double)
            self.RK_c = np.array([ RK_alpha, 1.0 ])

        
    def march(self,time=0.0):
        self.V0.vector()[:] = self.warp.wv.vector()[:]
        self.X0.vector()[:] = self.warp.wx.vector()[:]
        h = self.h
        KS = []
        VS = []
        for i in xrange(len(self.RK_b)):
            print " Stage ",i," at ",self.RK_b[i]," with aii=",self.RK_a[i,i]
            eps = 1.0
            tol = 1.0e-10
            maxiter = 10
            itcnt = 0
            Rhat = self.warp.M*self.V0.vector()
            Xhat = self.X0.vector().copy()
            for j in xrange(i):
                Rhat[:] += h*self.RK_a[i,j]*KS[j][:]
                Xhat[:] += h*self.RK_a[i,j]*VS[j][:]
            while eps>tol and itcnt < maxiter:
                print "  Assembling..."
                self.sysass()
                # AV.ident_zeros()

                K = self.warp.M - h*h*self.RK_a[i,i]**2*self.warp.AX - h*self.RK_a[i,i]*self.warp.AV
                R = Rhat - self.warp.M*self.warp.wv.vector() + h*float(self.RK_a[i,i])*self.warp.R
                self.bcapp(K,R,time+h*self.RK_c[i],itcnt!=0)

                print "  Solving..."
                solve(K,self.DelW.vector(),R)
                eps=np.linalg.norm(self.DelW.vector().array(), ord=np.Inf)
                print "  ",itcnt," Norm:", eps
                self.warp.wv.vector()[:] = self.warp.wv.vector()[:] + self.DelW.vector()[:]
                self.warp.wx.vector()[:] = Xhat[:] + h*self.RK_a[i,i]*self.warp.wv.vector()[:]
                self.update()
                itcnt += 1
            KS.append(self.warp.R)
            VS.append(self.warp.wv.vector().copy())
