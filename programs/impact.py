from src import *

from IPython import embed

"""
Lets go all out and shoot a smaller fibril at a fabric sheet!
"""

sheets = [
    Geometries.PlainWeaveFibrils(8,10.0,10.0, 8,10.0,10.0, 0.0,0.25, [ 3 ],0.41)
    ]

endpts = []
for s in sheets:
    endpts.extend( s.endpts() )
endpts.append([ [0, 0, 3.2],[ 0, 0, 2.9] ])

E = 1.26 #MPa
nu = 0.0
defaults = { 'mu':E/(2*(1 + nu)),
             'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
             'rho':0.00144,
             'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)),
             'contact_penalty':50.0,
             'dissipation':0.01
             }
props = [ {} for i in endpts ]
props[-1] = { 'radius':2.0, 'dissipation':0.0 }
Nelems = [ 20 for i in endpts ]
Nelems[-1] = 1

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)

outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/impact/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/impact/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1

output()

istart=0
for s in sheets:
    s.initialize(warp,istart)
    istart += s.nfibril

    
fib =  warp.fibrils[-1]
fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","-30.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
warp.pull_fibril_fields()

output()


cpairs = []
for s in sheets:
    cpairs.extend(s.contact_pairs())
cpairs.extend((i,len(endpts)-1) for i in xrange(len(endpts)-1))
warp.create_contacts(pairs=cpairs,cutoff=3.0)

Tmax=1.0
NT = 200
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
bound = CompiledSubDomain("x[2] <= 0.0 && on_boundary")
# bound = CompiledSubDomain("(near(x[0],LENGTH) || near(x[0],-LENGTH) || near(x[1],WIDTH) || near(x[1],-WIDTH) ) && on_boundary",LENGTH=restL,WIDTH=restW)

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)
	
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[2], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
warp.CG.OutputFile("post/impact/gammaC.pvd" )
for t in xrange(NT):
    if t%1==0:
        warp.create_contacts(pairs=cpairs,cutoff=3.0)
    dirk.march()
    if t%1==0:
        output()
