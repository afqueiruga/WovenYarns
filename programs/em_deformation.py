from src import *

from IPython import embed

"""
Deform a sheet with a B field
"""

sheets = [
    Geometries.PlainWeaveFibrils(8,12.0,10.0, 8,12.0,10.0, 0.0,0.25, [ 1 ],0.41)
    ]

endpts = []
for s in sheets:
    endpts.extend( s.endpts() )

E = 1.26 #MPa
nu = 0.0
defaults = { 'mu':E/(2*(1 + nu)),
             'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
             'rho':0.00144,
             'radius':0.2,
             'em_B':Constant((0.0,0.005,0.0)),
             'contact_penalty': 500.0,
             'dissipation':0.01
             }
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, MonolithicProblem)


outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/em_deformation/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/em_deformation/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1

output()


istart=0
for s in sheets:
    s.initialize(warp,istart)
    istart += s.nfibril
warp.pull_fibril_fields()
for fib in warp.fibrils:
    temp_field = Function(fib.problem.spaces['S'])
    temp_field.interpolate(Expression("A*x[0]+B",A=5.0/sheets[0].restX,B=5.0))
    assign(fib.problem.fields['wv'].sub(4), temp_field)
warp.pull_fibril_fields()
warp.update()

cpairs = []
for s in sheets:
    cpairs.extend(s.contact_pairs())
cpairs.extend((i,len(endpts)-1) for i in xrange(len(endpts)-1))
warp.create_contacts(pairs=cpairs,cutoff=1.5)


output()

bound_all = CompiledSubDomain("on_boundary")
bound_sides = CompiledSubDomain("( near(x[0],s) || near(x[0],-s) ) && on_boundary",s=sheets[0].restX)
subq = MultiMeshSubSpace(warp.spaces['W'],0)
subVol = MultiMeshSubSpace(warp.spaces['W'],4)
zeroV = Constant((0.0,0.0,0.0))
zeroS = Constant(0.0)
bcq = MultiMeshDirichletBC(subq,zeroV, bound_all)
bcVol = MultiMeshDirichletBC(subVol,zeroS, bound_all)

def apply_BCs(K,R,t,hold=False):
    bcq.apply(K,R)
    bcVol.apply(K,R)
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')


Tmax=1.0
NT = 500
h = Tmax/NT
dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
warp.CG.OutputFile("post/impact/gammaC.pvd" )
for t in xrange(NT):
    if t%1==0:
        warp.create_contacts(pairs=cpairs,cutoff=1.5)
    dirk.march()
    if t%10==0:
        output()

embed()
