from src import *

from IPython import embed

"""
Compute the effective properties of a textile square
"""

sheet = (2, 2.0,2.0, 2, 2.0,2.0, 
         0.0,0.35*1.5, [3,4], 0.35)
endpts = Geometries.PlainWeaveFibrils_endpts(*sheet)
defaults = { 'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':-2.0e1,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)



outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/RVE/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/RVE/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1


output()

Geometries.PlainWeaveFibrils_initialize(warp,0, *sheet)

def init_skew():
    for fib in warp.fibrils:
        fib.problem.fields['wv'].interpolate(Expression(("0.0","alpha*x[0]","0.0",
                                                         "0.0"," 0.0","0.0",
                                                         "0.0","0.0","0.0"),alpha=0.05))
    warp.pull_fibril_fields()
def init_freeze():
    for fib in warp.fibrils:
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                         "0.0"," 0.0","0.0",
                                                         "0.0","0.0","0.0")))
    warp.pull_fibril_fields()
output()

warp.create_contacts(cutoff=0.5)



Tmax=0.5
NT = 5
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
# bound = CompiledSubDomain("(near(x[0],s) || near(x[1],s)) && on_boundary",s=-2.2)
bound = CompiledSubDomain("on_boundary")

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)



def integrate_f():
    tx = np.zeros(3)
    ty = np.zeros(3)
    I = 0.0
    NDIR = sheet[0]*np.sum(sheet[8])
    for fib in warp.fibrils[:NDIR]:
        tx[0] += assemble(fib.problem.forms['p_t0_1'])
        tx[1] += assemble(fib.problem.forms['p_t1_1'])
        tx[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils[NDIR:]:
        ty[0] += assemble(fib.problem.forms['p_t0_1'])
        ty[1] += assemble(fib.problem.forms['p_t1_1'])
        ty[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils[NDIR/2:]:
        I += assemble(fib.problem.forms['p_J_1'])
    return tx,ty,I
        
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
# warp.CG.OutputFile("post/RVE/gammaC.pvd" )

def dynamic_steps():
    for t in xrange(NT):
        if t%1==0:
            warp.create_contacts(cutoff=0.5)
        dirk.march()
        # output()

ground = Constant(0.0)
bound_1 = CompiledSubDomain("near(x[0],s) && x[1] < 0.0 && on_boundary",s=-2.0)
testvol = Constant(1.0)
bound_2 = CompiledSubDomain("near(x[0],s) && x[1] < 0.0 && on_boundary",s= 2.0)
bcground = MultiMeshDirichletBC(warp.spaces['S'], ground, bound_1)
bctest = MultiMeshDirichletBC(warp.spaces['S'], testvol, bound_2)

def solve_em():
    DelV = MultiMeshFunction(warp.spaces['S'])
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        warp.create_contacts(cutoff=0.5)
        R,AE = warp.assemble_forms(['FE','AE'],'S')
        bctest.apply(AE,R)
        bcground.apply(AE,R)
        solve(AE,DelV.vector(),R)
        warp.fields['Vol'].vector()[:] -= DelV.vector()[:]
        eps=np.linalg.norm(DelV.vector().array(), ord=np.Inf)
        warp.update()
        print "  ",itcnt," Norm:", eps
        itcnt += 1
        # output()
        
def static_solve():
    DelW = MultiMeshFunction(warp.spaces['W'])
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        warp.create_contacts(cutoff=0.5)
        R,AX,AV=sys(0.0)
        apply_BCs(AX,R,0.0,True)
        solve(AX,DelW.vector(),R)
        warp.fields['wx'].vector()[:] -= DelW.vector()[:]
        eps=np.linalg.norm(DelW.vector().array(), ord=np.Inf)
        warp.update()
        print "  ",itcnt," Norm:", eps
        itcnt += 1
        # output()

NITER = 20
probes = [np.zeros((NITER,3)),np.zeros((NITER,3)),np.zeros((NITER))]
for i in xrange(NITER):
    init_skew()
    dynamic_steps()
    init_freeze()
    dynamic_steps()
    init_freeze()
    # solve_em()
    # static_solve()
    output()
    tx,ty,I= integrate_f()
    probes[0][i,:] = tx[:]
    probes[1][i,:] = ty[:]
    probes[2][i] = I

from matplotlib import pylab as plt
plt.plot(probes[0],'-+')
plt.figure()
plt.plot(probes[1],'-+')
plt.show()
embed()
