"""
Make a single resistor and check the voltage and currents
"""

from dolfin import *
import numpy as np
from matplotlib import pylab as plt

from src import Fibril
from src.Forms import DecoupledProblem

props = {}
Nelem = 10
L = 1.0
fib = Fibril.Fibril([ [0.0,0.0,0.0],[L,0.0,0.0]],
                    Nelem, props,
                    DecoupledProblem,order=(2,1))

zero = Constant(0.0)
applied = Constant(1.0)
topbound = CompiledSubDomain("near(x[0],1.0) && on_boundary")
botbound = CompiledSubDomain("near(x[0],0.0) && on_boundary")

bctop = DirichletBC(fib.problem.spaces['S'], applied, topbound)
bcbot = DirichletBC(fib.problem.spaces['S'], zero, botbound)


R = assemble(fib.problem.forms['FE'])
K = assemble(fib.problem.forms['AE'])

bcbot.apply(K,R)
bctop.apply(K,R)

DelV = Function(fib.problem.spaces['S'])

solve(K,DelV.vector(),R)

from IPython import embed
embed()
