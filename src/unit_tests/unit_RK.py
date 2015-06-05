"""
Verify the RK routines on silly ODES
"""
import numpy as np

from dolfin import *
from src import exRK


M_1 = np.array([[1.0]],dtype=np.double)
u = np.array([1.0],dtype=np.double)
def sys_1(time):
    return np.array([1.0],np.double)
def bcapp(K,R,time,hold):
    pass
def update():
    pass

odef = exRK.RK_field(1, [u],M_1,sys_1,bcapp,update)

RKER = exRK.exRK(0.01, exRK.exRK_table[4], [odef])
