"""

Take a single fiber and check it against standard Euler-Bernoulli beam theory.
Check convergences as we refine, too.

TODO:
 Vary poly order, nu, Nelems, to check conversion
 Do a torsion test

"""
from dolfin import *
import numpy as np
from matplotlib import pylab as plt

from src import Fibril
from src.Forms import MultiphysicsProblem

E = 10.0
nu = 0.0
fmag = 0.001
L = 100.0
props = { 'mu' :E/(2*(1 + nu)),
          'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
          'rho': 1.0,
          'f_dens_ext' : Constant((0.0,fmag,0.0))
          }

truesol = fmag*L**4/(96*E*0.15**2)
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("on_boundary")

def solveit(Nelem,p):
    fib = Fibril.Fibril([ [0.0,0.0,0.0],[L,0.0,0.0]], Nelem, props,
                    MultiphysicsProblem,order=(p,1))

    bcall = DirichletBC(fib.problem.spaces['W'], zero, bound)

    R = assemble(fib.problem.forms['F'])
    K = assemble(fib.problem.forms['AX'])
    K.ident_zeros()
    bcall.apply(K,R)

    DelW = Function(fib.problem.spaces['W'])
    solve(K,DelW.vector(),R)
    fib.problem.fields['wx'].vector()[:] -= DelW.vector()[:]

    # fib.WriteFile("post/beam_test.pvd")
    # fib.WriteSurface("post/beam_mesh_test.pvd")
    
    weval = np.zeros(11)
    fib.problem.fields['wx'].eval(weval,np.array([L/2.0,0.0,0.0]))
    return weval[1]

Nelems = [10,100,200,300,400,600,800,1000]
ps = [ 1,2,3, 4]
for p in ps:
    points = map(lambda x:solveit(x,p),Nelems)
    plt.loglog([10.0/x for x in Nelems],[np.abs( (y - truesol)/truesol) for y in points])

# plt.axhline(truesol)
# plt.plot(Nelems,points)
# plt.figure()
plt.show()


# from IPython import embed
# embed()
