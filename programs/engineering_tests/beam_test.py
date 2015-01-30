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
fmag = 0.0001
L = 1.0
R = 0.01
props = { 'mu' :E/(2*(1 + nu)),
          'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
          'rho': 1.0,
          # 'f_dens_ext' : Expression(("0.0","{1}*x[0]*{0}".format(1.0/L,fmag),"0.0")),
          'radius':R,
          # 'f_dens_ext' : Expression(("0.0","{0}".format(fmag),"0.0"))
          'f_dens_ext' : Expression(("0.0","{1}*cos(x[0]*{0})".format(np.pi/(2.0*L),fmag),"0.0"))
          }

# truesol = fmag*np.pi*R**2 * 11.0 * L**4 / ( 120.0 * E * np.pi/4.0*R**4 )
truesol = 2.0*fmag*np.pi*R**2*L**4/(3.0*np.pi**4 *E *np.pi/4.0*R**4)*(np.pi**3-24.0)
#fmag*L**4/(96*E*0.15**2)
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("near(x[0],0.0) && on_boundary")
print truesol

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
    fib.problem.fields['wx'].eval(weval,np.array([L,0.0,0.0],dtype=np.double))
    # print weval
    # print fib.problem.fields['wx'].compute_vertex_values()

    return weval[1]
Nelems = [2,4,6,8,10,12,14,16,18,20,30,40,50] #,100 ,200] #,300,400,600] #,800,1000]
ps = [ 1,2,3, 4]
for p in ps:
    points = map(lambda x:solveit(x,p),Nelems)
    print points
    plt.loglog([L/x for x in Nelems],
               [np.abs( (y - truesol)) for y in points]
               ,'+-',label='p='+str(p))
plt.legend()
# plt.axhline(truesol)
# plt.plot(Nelems,points)
# plt.figure()
plt.show()


# from IPython import embed
# embed()
