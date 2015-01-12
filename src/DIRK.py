#!/usr/bin/python

from dolfin import *
import numpy as np

from Warp import Warp

RK_alpha = 1.0-0.5*sqrt(2.0)
RK_c = np.array([ RK_alpha, 1.0 ])
RK_a = np.array([ [RK_alpha, 0.0],
                  [1.0-RK_alpha, RK_alpha] ], dtype=np.double)

class DIRK_Monolithic():
    def __init__(self,warp,sysass,update,bcapp,h):
        self.warp = warp
        self.sysass = sysass
        self.update = update
        self.bcapp = bcapp
        self.h = h

        self.X0 = MultiMeshFunction(warp.mmfs)
        self.V0 = MultiMeshFunction(warp.mmfs)
        self.DelW = MultiMeshFunction(warp.mmfs)
        
    def march(self):
        self.V0.vector()[:] = self.warp.wv.vector()[:]
        self.X0.vector()[:] = self.warp.wx.vector()[:]
        h = self.h
        KS = []
        VS = []
        for i in xrange(len(RK_c)):
            print " Stage ",i," at ",RK_c[i]," with aii=",RK_a[i,i]
            eps = 1.0
            tol = 1.0e-10
            maxiter = 10
            itcnt = 0
            Rhat = self.warp.M*self.V0.vector()
            Xhat = self.X0.vector().copy()
            for j in xrange(i):
                Rhat[:] += h*RK_a[i,j]*KS[j][:]
                Xhat[:] += h*RK_a[i,j]*VS[j][:]
            while eps>tol and itcnt < maxiter:
                print "  Assembling..."
                self.sysass()
                # AV.ident_zeros()

                K = self.warp.M - h*h*RK_a[i,i]**2*self.warp.AX - h*RK_a[i,i]*self.warp.AV
                R = Rhat - self.warp.M*self.warp.wv.vector() + h*float(RK_a[i,i])*self.warp.R
                self.bcapp(K,R,itcnt!=0)

                print "  Solving..."
                solve(K,self.DelW.vector(),R)
                eps=np.linalg.norm(self.DelW.vector().array(), ord=np.Inf)
                print "  ",itcnt," Norm:", eps
                self.warp.wv.vector()[:] = self.warp.wv.vector()[:] + self.DelW.vector()[:]
                self.warp.wx.vector()[:] = Xhat[:] + h*RK_a[i,i]*self.warp.wv.vector()[:]
                self.update()
                itcnt += 1
            KS.append(self.warp.R)
            VS.append(self.warp.wv.vector().copy())
