from src import *

from IPython import embed

"""
Compute the effective properties of a single yarn
"""


inner = [1,6,12]
inrad = 0.01
innum = np.sum(inner)
outer = [3,4]
outrad = 0.02



yarn = Geometries.CoiledYarn([[-1.0,0.0,0.0],[1.0,0.0,0.0]], 0.0, 1.1, 
                             inner+outer,[2.0*inrad for x in inner]+[2.0*outrad for x in outer],
                             [1 for x in inner]+ [-3 for x in outer])
endpts = yarn.endpts()

defaults = { 'radius':outrad,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':2.0e1,
             'em_sig':1.0e10,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
for i in xrange(innum):
    props[i]['radius'] = inrad
    props[i]['em_sig'] = 100.0
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

zeroS = Constant(0.0)
zeroW = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))

em_bc = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)

output()

# Geometries.CoiledYarn_initialize(warp,0, *yarn)
yarn.initialize(warp,0)
warp.create_contacts(cutoff=0.03)

warp.pull_fibril_fields()
# init_stretch()
output()
# exit()

Tmax=1.0
NT = 20
h = Tmax/NT
dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
# warp.CG.OutputFile("post/Yarn/gammaC.pvd" )

def dynamic_steps(NT):
    for t in xrange(NT):
        if t%5==0:
            warp.create_contacts(cutoff=0.03)
        dirk.march()
        # output()


def solve_em():
    print "Solving EM..."
    # Reset the potentials
    for fib in warp.fibrils:
        fib.problem.fields['Vol'].interpolate(Expression("A*x[0]+B",A=0.5/yarn.restL,B=0.5))
    warp.pull_fibril_fields()
    DelV = MultiMeshFunction(warp.spaces['S'])
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        warp.create_contacts(cutoff=0.5)
        R,AE = warp.assemble_forms(['FE','AE'],'S')
        em_bc.apply(AE,R)
        # bcground.apply(AE,R)
        solve(AE,DelV.vector(),R)
        warp.fields['Vol'].vector()[:] -= DelV.vector()[:]
        eps=np.linalg.norm(DelV.vector().array(), ord=np.Inf)
        warp.update()
        print "  ",itcnt," Norm:", eps
        itcnt += 1


temp_bc =  MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
def solve_temp():
    print "Solving Temperature"
    DelT = MultiMeshFunction(warp.spaces['S'])
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        warp.create_contacts(cutoff=0.5)
        R,AT = warp.assemble_forms(['FT','AT'],'S')
        temp_bc.apply(AT,R)
        solve(AT,DelT.vector(),R)
        warp.fields['T'].vector()[:] -= DelT.vector()[:]
        eps=np.linalg.norm(DelT.vector().array(), ord=np.Inf)
        warp.update()
        print "  ",itcnt," Norm:", eps
        itcnt += 1
        
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
sample_num = 0
def record_samples():
    global sample_num
    init_freeze()
    solve_em()
    solve_temp()
    output()
    tx,ty,I= integrate_f()
    probes[0][sample_num,:] = tx[:]
    probes[1][sample_num,:] = ty[:]
    probes[2][sample_num] = I
    sample_num += 1
solve_em()
init_freeze()
dynamic_steps(NT)

record_samples()

for it in xrange(NITER):
    init_stretch()
    dynamic_steps(NT)
    init_freeze()
    dynamic_steps(NT)
    record_samples()
    
from matplotlib import pylab as plt
plt.plot(probes[0],'-+')
plt.figure()
plt.plot(probes[1],'-+')
plt.show()
embed()
