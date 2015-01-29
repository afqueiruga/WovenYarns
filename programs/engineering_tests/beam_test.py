"""

Take a single fiber and check it against standard Euler-Bernoulli beam theory. Check convergences as we refine, too.

"""
from dolfin import *
import numpy as np

from src import Fibril
from src.Forms import MultiphysicsProblem

E = 10.0
nu = 0.0
props = { 'mu' :E/(2*(1 + nu)),
          'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
          'rho': 1.0,
          'f_dens_ext' : Constant((0.0,0.01,0.0))
          }
fib = Fibril.Fibril([ [0.0,0.0,0.0],[10.0,0.0,0.0]], 500, props,
                MultiphysicsProblem)


zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("on_boundary")
bcall = DirichletBC(fib.problem.spaces['W'], zero, bound)

#TODO:
# Vary poly order, nu, Nelems, to check conversion
# Do a torsion test

R = assemble(fib.problem.forms['F'])
K = assemble(fib.problem.forms['AX'])
K.ident_zeros()
bcall.apply(K,R)


DelW = Function(fib.problem.spaces['W'])

solve(K,DelW.vector(),R)

fib.problem.fields['wx'].vector()[:] -= DelW.vector()[:]


weval = np.zeros(11)
fib.problem.fields['wx'].eval(weval,np.array([5.0,0.0,0.0]))
print "numerical: ", weval[1], "analytical: ", 0.01*10.0**4/(96*E*0.15**2)

fib.WriteFile("post/beam_test.pvd")
fib.WriteSurface("post/beam_mesh_test.pvd")




# from IPython import embed
# embed()
