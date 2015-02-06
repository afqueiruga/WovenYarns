"""
Just build one of the new Fibril objects
"""
from dolfin import *
from src.Fibril import Fibril
from src.Forms import MultiphysicsProblem

props = { 'mu' : 10.0 }

fib = Fibril([ [0.0,0.0,0.0],[1.0,0.0,0.0]], 10, props,
             MultiphysicsProblem)

fib.WriteFile("src/unit_tests/unit_fibril_test.pvd")
fib.WriteSurface("src/unit_tests/unit_fibril_mesh_test.pvd")

M = assemble(fib.problem.forms['AX'])
from matplotlib import pylab as plt
plt.spy(M.array())
plt.show()
