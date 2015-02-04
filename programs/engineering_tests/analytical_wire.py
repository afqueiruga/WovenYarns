#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from IPython import embed

"""
Properties
"""

Tmax = 10.0
L = 1.0
Delta = 0.9
VBound = (L-Delta)/Tmax
endpts = [ [-L,0.0,0.0],[L,0.0,0.0] ]

props =  { 'radius':0.02,
           'em_B':Constant((0.01,0.01,0.0)) } 
        


def solveit(Nelem,p):
   
    
    fib = Fibril.Fibril( endpts, Nelem, props,
                        CurrentBeamProblem,order=(p,1))

    """
    Boundary conditions on the velocity updates
    """
    zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
    velocity = Expression(("x[0]*V","0.0","0.0"),V=-VBound/L)
    
    bound = CompiledSubDomain("on_boundary")
    bczero = DirichletBC(fib.problem.spaces['W'].sub(0), zero, bound)
    bcmove = DirichletBC(fib.problem.spaces['W'].sub(0), velocity, bound)
    fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",#"cos(x[0]*period)","sin(-x[0]*period)",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0"),period=np.pi/10.0))
    fib.problem.fields['wv'].interpolate(Expression(("x[0]*V","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0"),V=-VBound/L))
    """
    DIRK's assembling routine
    """
    
    
    def sys(time):
        R = assemble(fib.problem.forms['F'])
        K = assemble(fib.problem.forms['AX'])
        AV = assemble(fib.problem.forms['AV'])
        return R,K,AV
    def apply_BCs(K,R,time,hold):
        bczero.apply(K,R)
        # if hold:
        #     bczero.apply(K,R)
        # else:
        #     bcmove.apply(K,R)

    # Do a bit of dynamic relaxation first
    NT = 50
    h = Tmax/NT
    dirk = DIRK_Monolithic(h,LDIRK[1], sys, lambda : None,apply_BCs,
                       fib.problem.fields['wx'].vector(),fib.problem.fields['wv'].vector(),
                       assemble(fib.problem.forms['M']))
    for t in xrange(NT):
        dirk.march()
        fib.WriteFile("post/wire_test_"+str(t)+".pvd")
        fib.WriteSurface("post/wire_mesh_test"+str(t)+".pvd")


    # And the do a newton iteration at the end
    DelW = Function(fib.problem.spaces['W'])

    eps = 1.0
    tol = 1.0e-10
    maxiter = 10
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        R,AX,AV=sys(0.0)
        apply_BCs(AX,R,0.0,True)
        solve(AX,DelW.vector(),R)
        fib.problem.fields['wx'].vector()[:] -= DelW.vector()[:]
        eps=np.linalg.norm(DelW.vector().array(), ord=np.Inf)
        print "  ",itcnt," Norm:", eps

        fib.WriteFile("post/wire_test_"+str(t+itcnt)+".pvd")
        fib.WriteSurface("post/wire_mesh_test"+str(t+itcnt)+".pvd")
        itcnt += 1
    
    
    weval = np.zeros(9)
    fib.problem.fields['wx'].eval(weval,np.array([0.0,0.0,0.0],dtype=np.double))
    # print weval
    # print fib.problem.fields['wx'].compute_vertex_values()

    return weval[1]

solveit(20,1)
