from src import *

from IPython import embed

"""
Compute the effective properties of a textile square
"""

sheet = (4, 4.2,4.0, 4, 4.2,4.0, 
         0.0,0.35*2.0, [3,4,3], 0.35)
endpts = Geometries.PlainWeaveFibrils_endpts(*sheet)
defaults = { 'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':-2.0e-1}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, CurrentBeamProblem)
warp.output_states("post/RVE/yarn_{0}_"+str(0)+".pvd",0)
warp.output_surfaces("post/RVE/mesh_{0}_"+str(0)+".pvd",0)

Geometries.PlainWeaveFibrils_initialize(warp,0, *sheet)
warp.pull_fibril_fields()

warp.output_states("post/RVE/yarn_{0}_"+str(1)+".pvd",0)
warp.output_surfaces("post/RVE/mesh_{0}_"+str(1)+".pvd",0)

warp.create_contacts(cutoff=0.5)


Tmax=10.0
NT = 100
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
bound = CompiledSubDomain("on_boundary")

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)

    
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
warp.CG.OutputFile("post/RVE/gammaC.pvd" )
for t in xrange(NT):
    if t%1==0:
        warp.create_contacts(cutoff=0.5)
    dirk.march()
    if t%1==0:
        warp.output_states("post/RVE/yarn_{0}_"+str(t/1+2)+".pvd",1)
        warp.output_surfaces("post/RVE/mesh_{0}_"+str(t/1+2)+".pvd",1)
