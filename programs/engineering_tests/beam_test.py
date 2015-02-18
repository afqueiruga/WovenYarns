"""

Take a single fiber and check it against standard Euler-Bernoulli beam theory.
Check convergences as we refine, too.

TODO:
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
Nelems = [ [250,275,300,325,350,375,400],
	   [100,125,150,175,200],
	   [2,3,4,5,10,15,20,25],
	   [2,3,4,5,6,11] ]
ps = [ 1,2 ,3, 4]
points = []
for NS,p in zip(Nelems,ps):
    points.append( map(lambda x:solveit(x,p),NS) )
    # print points
bestsol = points[-1][-1]
def make_plots(Nelems,points):
    for p,NS,pts in zip(ps,Nelems,points):
        plt.loglog([L/x for x in NS],
                   [np.abs( (y - bestsol)) for y in pts]
                   ,'+-',label='p='+str(p))
        
        plt.legend()
    plt.figure()
    for p,NS,pts in zip(ps,Nelems,points):
        plt.plot([x for x in NS],
                 [y for y in pts]
                 ,'+-',label='p='+str(p))
        plt.axhline(truesol)
    plt.show()


def compute_convergence(Nelems,points):
    import scipy.stats
    best = points[-1][-1]
    for ix,(NS,pts) in enumerate(zip(Nelems,points)):
        hs = [L/x for x in NS]
        print (scipy.stats.linregress([ np.log(x) for x in hs[:(-1 if ix==len(points)-1 else -2)] ],
                                      [ np.log(np.abs((y-best))) for y in pts[:(-1 if ix==len(points)-1 else -2)] ]))[0],
        print ""

make_plots(Nelems,points)
compute_convergence(Nelems, points)

from IPython import embed
embed()
