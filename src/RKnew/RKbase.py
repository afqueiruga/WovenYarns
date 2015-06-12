import numpy as np

class RK_field():
    """
    order:
       0 : implicit
       1 : 1st order
       2 : 2nd order
    """
    def __init__(self, order, u,M,sys,bcapp, update):
        self.order = order
        self.u = u
        self.M = M
        self.sys = sys
        self.bcapp = bcapp
        self.update = update

        self.u0 = [ s.copy() for s in self.u ]
        # if order == 0:
        self.DU = [ s.copy() for s in self.u ]
        if order == 2:
            self.uhat = [ s.copy() for s in self.u ]
    def save_u0(self):
        for s,v in zip(self.u0,self.u):
            s[:] = v[:]

class RKbase():
    def __init__(self,h, tableau, fields):
        self.h = h
        
        self.RK_a = tableau['a']
        self.RK_b = tableau['b']
        self.RK_c = tableau['c']

        self.im_fields = []
        self.ex_fields = []
        for f in fields:
            if f.order==0:
                self.im_fields.append(f)
            else:
                self.ex_fields.append(f)
    def DPRINT(*args):
        for a in args[1:]:
            print a,
        print ""
        
    def march(self,time=0.0):
        pass
