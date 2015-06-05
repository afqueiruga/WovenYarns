import numpy as np
from dolfin import *

exRK_table = {
    1 : {
        'a':np.array([ [0.0]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 0.0 ], dtype=np.double)
        },
    4 : {
        'a': np.array([ [ 0.0, 0.0, 0.0, 0.0 ],
                        [ 0.5, 0.0, 0.0, 0.0 ],
                        [ 0.0, 0.5, 0.0, 0.0 ],
                        [ 0.0, 0.0, 1.0, 0.0  ] ], dtype=np.double),
        'b':np.array( [ 1.0/6.0, 2.0/3.0, 2.0/3.0, 1.0/6.0 ], dtype=np.double),
        'c': np.array( [ 0.0, 0.5, 0.5, 1.0 ], dtype=np.double)
        }
    }



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
            
class exRK():
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
            print " Stage ",i," at ",RK_c[i]," with ai_=",RK_a[i,:]
            # Step 1: Calculate values of explicit fields at this step
            for f in self.ex_fields:
                for s,v in zip(f.u,f.u0):
                    s[:] = v[:]
                for j in xrange(i):
                    f.DU[0] += h*RK_a[i,j]*f.ks[j][:] # Need to solve matrix
                    if order == 2:
                        f.u[1] += h*RK_a[i,j]*f.vs[j][:]
                solve(K,f.u[0],f.DU[0])
                f.u[0][:] += f.u0[0][:]
                f.update()
            # Step 2: Solve Implicit fields
            for f in self.im_fields:
                print " Solving Implicit field... "
                F,K = f.sys(time)
                f.bcapp(F,K,time+h*RK_c[i],itcnt!=0)
                print "   Solving Matrix... "
                solve(K,f.DU[0],F)
                eps = np.linalg.norm(f.DU[0].array(), ord=np.Inf)
                print "  ",itcnt," Norm:", eps
                f.u[0][:] = f.u[0][:] + f.DU[0][:]
                f.update()
                itcnt += 1
            
            # Step 3: Compute k for each field
            for f in self.ex_fields:
                F = f.sys(time)
                f.bcapp(None,F,time+h*RK_c[i],itcnt!=0)
                f.ks.append(F)
                if f.order == 2:
                    f.vs.append(f.u[0].copy())

        # Do the final Mv=sum bk
        for f in self.ex_fields:
            for s,v in zip(f.u,f.u0):
                s[:] = v[:]
            for j in xrange(len(RK_b)):
                f.DU[0] += h*RK_b[j]*f.ks[j][:] # Need to solve matrix
                if order == 2:
                    f.u[1] += h*RK_b[j]*f.vs[j][:]
            solve(K,f.u[0],f.DU[0])
            f.u[0][:] += f.u0[0][:]
            f.update()
