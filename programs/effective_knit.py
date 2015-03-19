from src import *

from IPython import embed

"""
Compute the effective properties of a knitted square
"""

sheet = Geometries.StockinetteFibrils(5.0,2.8, 3,3.0, [1,2], [0.15,0.15], 0.1)
endpts = sheet.endpts()
defaults = { 'radius':0.07,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':1.0e0,
             'contact_penalty':1000.0,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
Nelems = [ 40 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)


outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/KnitRVE/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/KnitRVE/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1

output()

sheet.initialize(warp,0)
warp.pull_fibril_fields()
output()

cpairs = sheet.contact_pairs()
warp.create_contacts(cutoff=0.5)

# warp.CG.desired_pairs = [ (x[0],x[1]) for x in warp.CG.active_pairs ]


Tmax=1.0
NT = 400
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
# bound = CompiledSubDomain("(near(x[0],s) || near(x[1],s)) && on_boundary",s=-2.2)
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

def dynamic_steps(NT):
    for t in xrange(NT):
        if t%1==0:
            warp.create_contacts(cutoff=0.5)
        dirk.march()
        output()
dynamic_steps(NT)
