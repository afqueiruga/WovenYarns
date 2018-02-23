"""
Make a single resistor and check the voltage and currents
"""

from dolfin import *
import numpy as np
from matplotlib import pylab as plt

from src import Fibril
from src.Forms import DecoupledProblem

RAD = 1.0/np.sqrt(np.pi)#0.2
REXT = 2.0 #2.0879
VAPP = 3.0 #0.21234
SIGMA = 1.0 #0.543

B = 0.0
props = {'radius':RAD,
         'em_bc_J_1': VAPP/(REXT*np.pi*RAD**2),
         'em_bc_r_1': 1.0/(REXT*np.pi*RAD**2),
         'em_sig':SIGMA,
         'em_B':Constant((0.0,0.0,B))}

Nelem = 10
L = 1.0
fib = Fibril.Fibril([ [0.0,0.0,0.0],[L,0.0,0.0]],
                    Nelem, props,
                    DecoupledProblem,order=(2,1))

RFIB = L/(SIGMA*np.pi*RAD**2)
VEMF = B*3.0*L #*np.pi*RAD**2 # CHECK MY As! This is wrong!
print (VAPP+VEMF)*RFIB/(REXT+RFIB) - VEMF

zero = Constant(0.0)
applied = Constant(1.0)
topbound = CompiledSubDomain("near(x[0],1.0) && on_boundary")
botbound = CompiledSubDomain("near(x[0],0.0) && on_boundary")
bctop = DirichletBC(fib.problem.spaces['S'], applied, topbound)
bcbot = DirichletBC(fib.problem.spaces['S'], zero, botbound)


fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                 "0.0"," 0.0","0.0",
                                                 "0.0","0.0","0.0")))

R = assemble(fib.problem.forms['FE'])
K = assemble(fib.problem.forms['AE'])


bcbot.apply(K,R)
# bctop.apply(K,R)

DelV = Function(fib.problem.spaces['S'])

solve(K,DelV.vector(),R)
print DelV.vector().array()


from IPython import embed
# embed()
