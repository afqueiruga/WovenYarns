from src import *

from IPython import embed

"""
Compute the effective properties of a single yarn
"""

# endpts = Geometries.PackedYarn([[-1.0,0.0,0.0],[1.0,0.0,0.0]], [2,3,2], 0.1)
# endpts.append([[-3.0,0.0,0.0],[3.0,0.0,0.0]])


yarn = ([[-1.0,0.0,0.0],[1.0,0.0,0.0]], 0.1, 1.1, [4,5],[0.11,0.11],[3,3,3])

endpts = Geometries.CoiledYarn_endpts(*yarn)

defaults = { 'radius':0.05,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':2.0e1,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
props[-1] = { 'radius':0.05 }
Nelems = [ 40 for i in endpts ]


warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)


outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/Yarn/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/Yarn/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1

#
# Setting of fibril velocities
#
def init_stretch():
    for fib in warp.fibrils:
        fib.problem.fields['wv'].interpolate(Expression(("alpha*x[0]","0.0","0.0",
                                                         "0.0"," 0.0","0.0",
                                                         "0.0","0.0","0.0"),alpha=0.05))
    warp.pull_fibril_fields()
def init_freeze():
    for fib in warp.fibrils:
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                         "0.0"," 0.0","0.0",
                                                         "0.0","0.0","0.0")))
    warp.pull_fibril_fields()



#
# Boundary conditions
#
zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
# bound = CompiledSubDomain("(near(x[0],s) || near(x[1],s)) && on_boundary",s=-2.2)
bound = CompiledSubDomain("on_boundary")

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)         
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')



output()
warp.create_contacts(cutoff=0.25)
Geometries.CoiledYarn_initialize(warp,0, *yarn)
warp.pull_fibril_fields()
# init_stretch()
output()

Tmax=0.2
NT = 10
h = Tmax/NT
dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
# warp.CG.OutputFile("post/Yarn/gammaC.pvd" )

def dynamic_steps(NT):
    for t in xrange(NT):
        if t%1==0:
            warp.create_contacts(cutoff=0.3)
        dirk.march()
        output()


def integrate_f():
    tx = np.zeros(3)
    ty = np.zeros(3)
    I = 0.0
    for fib in warp.fibrils:
        tx[0] += assemble(fib.problem.forms['p_t0_1'])
        tx[1] += assemble(fib.problem.forms['p_t1_1'])
        tx[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils:
        ty[0] += assemble(fib.problem.forms['p_t0_1'])
        ty[1] += assemble(fib.problem.forms['p_t1_1'])
        ty[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils:
        I += assemble(fib.problem.forms['p_J_1'])
    return tx,ty,I


NITER = 5
probes = [np.zeros((NITER+1,3)),np.zeros((NITER+1,3)),np.zeros((NITER+1))]

init_freeze()
dynamic_steps(NT)
tx,ty,I= integrate_f()
probes[0][0,:] = tx[:]
probes[1][0,:] = ty[:]
probes[2][0] = I
for it in xrange(NITER):
    init_stretch()
    dynamic_steps(NT)
    init_freeze()
    dynamic_steps(NT)
    init_freeze()
    tx,ty,I= integrate_f()
    probes[0][it+1,:] = tx[:]
    probes[1][it+1,:] = ty[:]
    probes[2][it+1] = I
    
from matplotlib import pylab as plt
plt.plot(probes[0],'-+')
plt.figure()
plt.plot(probes[1],'-+')
plt.show()
embed()
