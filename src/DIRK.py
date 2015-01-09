#!/usr/bin/python

from dolfin import *
from Warp import Warp


RK_alpha = 1.0-0.5*sqrt(2.0)
RK_c = np.array([ RK_alpha, 1.0 ])
RK_a = np.array([ [RK_alpha, 0.0],
                  [1.0-RK_alpha, RK_alpha] ], dtype=np.double)

class DIRK_Monolithic():
    def __init__(self,warp,rout):
        self.warp = warp
        self.rout = rout

        self.X0 = MultiMeshFunction(warp.mmfs)
        self.V0 = MultiMeshFunction(warp.mmfs)
        self.DelW = MultiMeshFunction(warp.mmfs)
    def march(self):
        self.V0.vector()[:] = self.warp.wv.vector()[:]
        self.X0.vector()[:] = self.warp.wx.vector()[:]
        KS = []
        VS = []

