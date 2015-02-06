import numpy as np

class RK():
    def __init__(self,h, tableau, sys,update,bcapp, x,v, M):
        self.h = h
        
        self.sys = sys
        self.update = update
        self.bcapp = bcapp

        self.RK_a = tableau['a']
        self.RK_b = tableau['b']
        self.RK_c = tableau['c']

        self.x = x
        self.v = v

        self.X0 = x.copy()
        self.V0 = v.copy()
        self.Xhat = x.copy()
        self.DX = x.copy()

        self.M = M
        
    def march(self,time=0.0):
        pass
