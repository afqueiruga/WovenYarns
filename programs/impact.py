from src import *

from IPython import embed

"""
Lets go all out and shoot a smaller fibril at a fabric sheet!
"""

sheets = [ (8,5.2,5.0, 8,5.2,5.0,  0.0,0.2),
           (8,5.2,5.0, 8,5.2,5.0, -0.8,0.2) ]

endpts = []

for z in sheets:
    endpts.extend(  Geometries.PlainWeave_endpts(*z) )
endpts.append([ [0, 0, 3.2],[ 0, 0, 2.9] ])

defaults = { 'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)) }
props = [ {} for i in endpts ]
props[-1] = { 'radius':2.0 }
Nelems = [ 20 for i in endpts ]
Nelems[-1] = 1

warp = Warp(endpts,props,defaults, Nelems, CurrentBeamProblem)


warp.output_states("post/impact/yarn_{0}_"+str(0)+".pvd",0)
warp.output_surfaces("post/impact/mesh_{0}_"+str(0)+".pvd",0)

istart=0
for z in sheets:
    Geometries.PlainWeave_initialize(warp, istart, *z)
    istart += z[0]+z[3]

    
fib =  warp.fibrils[-1]
fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","-0.5",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))


warp.output_states("post/impact/yarn_{0}_"+str(1)+".pvd",0)
warp.output_surfaces("post/impact/mesh_{0}_"+str(1)+".pvd",0)

exit()

warp.create_contacts(cutoff=2.0)

Tmax=20.0
NT = 200
h = Tmax/NT

zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
bound = CompiledSubDomain("(near(x[0],LENGTH) || near(x[0],-LENGTH) || near(x[1],WIDTH) || near(x[1],-WIDTH) ) && on_boundary",LENGTH=restL,WIDTH=restW)

subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(subs, zero, bound)
def apply_BCs(K,R,t,hold=False):
	bcall.apply(K,R)
	
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

dirk = DIRK_Monolithic(h,LDIRK[1], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
warp.CG.OutputFile("post/impact/gammaC.pvd" )
for t in xrange(NT):
    if t%1==0:
        warp.create_contacts(cutoff=2.5)
    dirk.march()
    if t%1==0:
        warp.output_states("post/impact/yarn_{0}_"+str(t/1+2)+".pvd",1)
        warp.output_surfaces("post/impact/mesh_{0}_"+str(t/1+2)+".pvd",1)
