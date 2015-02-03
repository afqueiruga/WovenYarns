#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from IPython import embed

"""
Set up the warp with a single fibril
"""
endpts = [ [[-10.0,0.0,0.0],[10.0,0.0,0.0]] ,
            ]
defaults = { 'radius':0.2,
             'em_B':Constant((0.0,1.0,0.0)) }
props = [ { 'em_B':Constant((1.0,1.0,0.0))} ]

warp = Warp(endpts, props, defaults, CurrentBeamProblem)



"""
Boundary conditions on the velocity updates
"""
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
bound = CompiledSubDomain("on_boundary")
# subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(warp.spaces['W'], zero, bound)
def apply_BCs(K,R,t,hold=False):
    bcall.apply(K,R)


"""
DIRK's assembling routine
"""
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')


"""
Initialize routine
"""
def initialize():
    for i,fib in enumerate(warp.fibrils):
        fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "0.0")))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "x[0]/100.0")))
        mdof = warp.spaces['W'].dofmap()
        warp.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]


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

solveit(20,1)
