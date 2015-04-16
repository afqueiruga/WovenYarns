from src import *

from IPython import embed

"""
Lets go all out and shoot a smaller fibril at a fabric sheet!
"""

sheets = [
    Geometries.PlainWeaveFibrils(32,50.8,50.8, 32,50.8,50.8, 0.0,0.41, [ 1 ],0.81)
    ]

endpts = []
for s in sheets:
    endpts.extend( s.endpts() )
endpts.append([ [0, 0, 18.0],[ 0, 0, 12.25] ])

E = 1.26 #MPa
nu = 0.2
Bmag = 0.001
phi = np.pi/4.0
defaults = { 'mu':E/(2*(1 + nu)),
             'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
             'rho': 0.0000144, #0.00144,
             'radius':0.4,
             'em_B':Constant((0.0,Bmag*np.cos(phi),Bmag*np.sin(phi))),
             'contact_penalty':20.0,
             'dissipation':0.001,
             'contact_em': 0.0,
             'em_sig':1.0e5,
             'em_bc_r_0': 100.0,
             'em_bc_J_0': -100.0
             }
props = [ {} for i in endpts ]
for i in xrange( np.sum(sheets[0].pattern)*sheets[0].NX,np.sum(sheets[0].pattern)*(sheets[0].NX+sheets[0].NY) ):
    props[i]['em_bc_r_0'] = 0.001
    props[i]['em_bc_J_0'] = 0.0
props[-1] = { 'radius':12.0, 'dissipation':0.0, 'rho':0.01 }
props[-1]['em_bc_r_0'] = 0.001
props[-1]['em_bc_J_0'] = 0.0
props[-1]['em_sig']=1.0

Nelems = [ 80 for i in endpts ]
# for i in xrange(np.sum(sheets[0].pattern)*(sheets[0].NX/2-1),np.sum(sheets[0].pattern)*(sheets[0].NX/2+1)):
#     Nelems[i]= 80
# for i in xrange(np.sum(sheets[0].pattern)*(sheets[0].NY/2-1),np.sum(sheets[0].pattern)*(sheets[0].NY/2+1)):
#     Nelems[np.sum(sheets[0].pattern)*(sheets[0].NX) + i]= 80
Nelems[-1] = 1

warp = Warp(endpts,props,defaults, Nelems, MonolithicProblem)

outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/impact_em/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/impact_em/mesh_{0}_"+str(outputcnt)+".pvd",1)
    # warp.CG.OutputFile("post/impact/gammaC_"+str(outputcnt)+".pvd" )
    outputcnt+=1
warp.create_contacts(cutoff=1.2,candidate_buffer=0.1)
output()

istart=0
for s in sheets:
    s.initialize(warp,istart)
    istart += s.nfibril



# warp.load("data/impact_em_works")

fib =  warp.fibrils[-1]
fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0", "0.0","0.0")))
fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","-30.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0", "0.0","0.0")))
warp.pull_fibril_fields()

cpairs = []
for s in sheets:
    cpairs.extend(s.contact_pairs())
cpairs.extend((i,len(endpts)-1) for i in xrange(len(endpts)-1))
warp.create_contacts(pairs=cpairs,cutoff=6.5)

# outputcnt = 52
output()


# for fib in warp.fibrils:
#     temp_field = Function(fib.problem.spaces['S'])
#     temp_field.interpolate(Expression("A*x[0]+B",A=1.0/sheets[0].restX,B=0.0))
#     assign(fib.problem.fields['wv'].sub(4), temp_field)
#     temp_field.interpolate(Expression("1.0"))
#     assign(fib.problem.fields['wv'].sub(3), temp_field)
# warp.pull_fibril_fields()
# warp.update()

Tmax=0.5
NT = 500
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
bound_all = CompiledSubDomain("x[2] <= 0.0 && on_boundary")
bound_sides = CompiledSubDomain("x[2] <= 0.0 &&( near(x[0],y) || near(x[0],-y) ) && on_boundary",y=sheets[0].restX)
bound_one_side = CompiledSubDomain("x[2] <= 0.0 && ( near(x[0],y) ) && on_boundary",y= sheets[0].restX)
bound_both_sides = CompiledSubDomain("x[2] <= 0.0 && ( near(x[0],y) || near(x[1],y) ) && on_boundary",y= sheets[0].restX)

subq = MultiMeshSubSpace(warp.spaces['W'],0)
subT = MultiMeshSubSpace(warp.spaces['W'],3)
subVol = MultiMeshSubSpace(warp.spaces['W'],4)
zeroV = Constant((0.0,0.0,0.0))
zeroS = Constant(0.0)
bcq = MultiMeshDirichletBC(subq,zeroV, bound_all)
bcT = MultiMeshDirichletBC(subT,zeroS, bound_all)
bcVol = MultiMeshDirichletBC(subVol,zeroS, bound_one_side)

def apply_BCs(K,R,t,hold=False):
    bcq.apply(K,R)
    bcT.apply(K,R)
    bcVol.apply(K,R)
    
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[2], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
for t in xrange(NT):
    if t%1==0:
        warp.create_contacts(pairs=cpairs,cutoff=1.2,candidate_buffer=0.1)
    dirk.march()
    if t%10==0:
        output()

embed()
