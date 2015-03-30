from src import *

from IPython import embed

"""
Compute the effective properties of a textile square
"""

sheet = Geometries.PlainWeaveFibrils(4, 4.0,4.0, 4, 4.0,4.0, 
         0.0,0.35*1.5, [3,4,3], 0.3)
endpts = sheet.endpts()
defaults = { 'radius':0.15,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':1.0e0,
             'contact_penalty':5.0,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0)),
             'contact_em':0.1}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)



outputcnt = 21
def output():
    global outputcnt
    warp.output_states("post/RVE/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/RVE/mesh_{0}_"+str(outputcnt)+".pvd",1)
    # warp.CG.OutputFile("post/RVE/gammaC_"+str(outputcnt)+".pvd" )
    outputcnt+=1


# output()

sheet.initialize(warp,0)
warp.pull_fibril_fields()
def init_skew():
    for fib in warp.fibrils:
        fib.problem.fields['wv'].interpolate(Expression(("0.0","alpha*x[0]","0.0",
                                                         "0.0"," 0.0","0.0",
                                                         "0.0","0.0","0.0"),alpha=0.05))
    warp.pull_fibril_fields()
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

cpairs = sheet.contact_pairs()
warp.create_contacts(cpairs,cutoff=0.5)
output()



Tmax=5.0
NT = 100
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
# bound = CompiledSubDomain("(near(x[0],s) || near(x[1],s)) && on_boundary",s=-2.2)
bound = CompiledSubDomain("on_boundary")

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)



def integrate_f():
    tx0 = np.zeros(3)
    tx1 = np.zeros(3)
    ty0 = np.zeros(3)
    ty1 = np.zeros(3)
    I0 = 0.0
    I1 = 0.0
    NDIR = sheet.NX*np.sum(sheet.pattern)
    for fib in warp.fibrils[:NDIR]:
        tx0[0] += assemble(fib.problem.forms['p_t0_0'])
        tx0[1] += assemble(fib.problem.forms['p_t1_0'])
        tx0[2] += assemble(fib.problem.forms['p_t2_0'])
        tx1[0] += assemble(fib.problem.forms['p_t0_1'])
        tx1[1] += assemble(fib.problem.forms['p_t1_1'])
        tx1[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils[NDIR:]:
        ty0[0] += assemble(fib.problem.forms['p_t0_0'])
        ty0[1] += assemble(fib.problem.forms['p_t1_0'])
        ty0[2] += assemble(fib.problem.forms['p_t2_0'])
        ty1[0] += assemble(fib.problem.forms['p_t0_1'])
        ty1[1] += assemble(fib.problem.forms['p_t1_1'])
        ty1[2] += assemble(fib.problem.forms['p_t2_1'])
    for fib in warp.fibrils[:NDIR/2]:
        I0 += assemble(fib.problem.forms['p_J_0'])
        I1 += assemble(fib.problem.forms['p_J_1'])
    return tx0,tx1,ty0,ty1,I0,I1
        
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
# warp.CG.OutputFile("post/RVE/gammaC.pvd" )

def dynamic_steps(NT):
    for t in xrange(NT):
        if t%5==0:
            warp.create_contacts(cpairs,cutoff=0.5)
        dirk.march()
        if t%10==0:
            output()

ground = Constant(0.0)
zeroS = Constant(0.0)
bound_1 = CompiledSubDomain("near(x[0],s) && x[1] < 0.0 && x[1] > s2 && on_boundary",s=-sheet.restX, s2 = -sheet.restY/2.0)
testvol = Constant(1.0)
bound_2 = CompiledSubDomain("(near(x[0],s) || near(x[0],-s) ) && x[1] < 0.0 &&  x[1] > s2 && on_boundary",s= sheet.restX, s2 = -sheet.restY/2.0)
bcground = MultiMeshDirichletBC(warp.spaces['S'], ground, bound_1)
bctest = MultiMeshDirichletBC(warp.spaces['S'], testvol, bound_2)

em_bc = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound_2)
def solve_em(expr,em_bcs):
    print "Solving EM..."
    # Reset the potentials
    for fib in warp.fibrils:
        fib.problem.fields['Vol'].interpolate(expr)
    warp.pull_fibril_fields()
    DelV = MultiMeshFunction(warp.spaces['S'])
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        warp.create_contacts(cutoff=0.5)
        R,AE = warp.assemble_forms(['FE','AE'],'S')
        for bc in em_bcs:
            bc.apply(AE,R)
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

# init_freeze()
# dynamic_steps(NT)
# warp.save("data/plain_relaxed_data")

# embed()

def do_calculated_analysis():
    NT = 10
    NITER = 10
    probes = [np.zeros((NITER+1,3)),np.zeros((NITER+1,3)),np.zeros((NITER+1,3)),np.zeros((NITER+1,3)),np.zeros((NITER+1))]
    sample_num = 0
    def record_samples():
        global sample_num
        init_freeze()
        solve_em()
        # solve_temp()
        # output()
        
        tx0,tx1,ty0,ty1,I= integrate_f()
        probes[0][sample_num,:] = tx0[:]
        probes[1][sample_num,:] = ty0[:]
        probes[2][sample_num] = I
        sample_num += 1

    # init_freeze()
    # dynamic_steps(NT)

    # record_samples()
    for i in xrange(NITER):
        init_stretch()
        dynamic_steps(NT)
        init_freeze()
        dynamic_steps(NT)
        warp.save("data/plain_stretch_"+str(i))
        # record_samples()

    from matplotlib import pylab as plt
    plt.plot(probes[0],'-+')
    plt.figure()
    plt.plot(probes[1],'-+')
    plt.show()
    embed()



domain_probe_1 = CompiledSubDomain("(near(x[0],s) ) && x[1] < 0.0 &&  x[1] > s2 && on_boundary",s= sheet.restX, s2 = -sheet.restY/2.0)
domain_probe_2 = CompiledSubDomain("(near(x[0],-s) ) && x[1] < 0.0 &&  x[1] > s2 && on_boundary",s= sheet.restX, s2 = -sheet.restY/2.0)
domain_probe_3 = CompiledSubDomain("(near(x[0],-s) ) && x[1] > 0.0 &&  x[1] < s2 && on_boundary",s= sheet.restX, s2 = sheet.restY/2.0)
domain_probe_4 = CompiledSubDomain("(near(x[1], s) ) && x[0] < 0.0 &&  x[0] > s2 && on_boundary",s= sheet.restY, s2 = -sheet.restX/2.0)

em_bc_probe_1 =  MultiMeshDirichletBC(warp.spaces['S'], zeroS, domain_probe_1)
em_bc_probe_2 =  MultiMeshDirichletBC(warp.spaces['S'], zeroS, domain_probe_2)
em_bc_probe_3 =  MultiMeshDirichletBC(warp.spaces['S'], zeroS, domain_probe_3)
em_bc_probe_4 =  MultiMeshDirichletBC(warp.spaces['S'], zeroS, domain_probe_4)


em_tests = [ ( Expression("A*x[0]+B",A=0.5/sheet.restX,B=0.5), (em_bc_probe_1, em_bc_probe_2) ),
             ( Expression("A*x[0]+B",A=0.5/sheet.restX,B=0.5), (em_bc_probe_1, em_bc_probe_3) ),
             ( Expression("(x[0]<x[1]? (A) : (B) )",A=0.0,B=1.0), (em_bc_probe_1, em_bc_probe_4) ) ]


def analyze_state(fname):
    warp.load(fname)

    res = []
    for expr,bcs in em_tests:
        solve_em(expr,bcs)
        solve_temp()
        output()
        res.append(integrate_f())
    return res

import matplotlib
from matplotlib import pylab as plt
def plot_results(results):
    strains = 100*0.05*h*10*np.arange(len(results))
    font = {'family' : 'normal',
            'size'   : 16}
    
    matplotlib.rc('font', **font)
    plt.xlabel('Strain %')
    plt.ylabel('Reaction (N)')
    plt.plot(strains,[r[0][1][0] for r in results],'-+',label='txx')
    plt.plot(strains,[r[0][1][1] for r in results],'-+',label='txy')
    plt.plot(strains,[r[0][1][2] for r in results],'-+',label='txz')
    plt.legend(loc=4)
    plt.figure()
    plt.xlabel('Strain %')
    plt.ylabel('Reaction (N)')
    plt.plot(strains,[r[0][3][0] for r in results],'-+',label='tyx')
    plt.plot(strains,[r[0][3][1] for r in results],'-+',label='tyy')
    plt.plot(strains,[r[0][3][2] for r in results],'-+',label='tyz')
    plt.legend(loc=4)
    plt.figure()
    plt.xlabel('Strain %')
    plt.ylabel('Current (A)')
    plt.plot(strains,[-r[0][-1] for r in results],'-+',label='Probe 1')
    plt.plot(strains,[-r[1][-1] for r in results],'-+',label='Probe 2')
    plt.plot(strains,[-r[2][-1] for r in results],'-+',label='Probe 3')
    plt.legend(loc=4)
    plt.show()

# states = [ "data/plain_shear_"+str(t) for t in range(11) ]
# states = ["data/plain_relaxed_data"]+[ "data/plain_stretch_"+str(t) for t in range(10) ]
# results = map(analyze_state,states)
# plot_results(results)
# warp.load("data/plain_relaxed_data")
# do_calculated_analysis()
analyze_state("data/plain_shear_10")
analyze_state("data/plain_stretch_9")
embed()
