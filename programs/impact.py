from src import *

from IPython import embed

"""
Lets go all out and shoot a smaller fibril at a fabric sheet!
"""

restL = 5.1
newL = 5.0
restW = 5.1
newW = 5.0
NX = 8
NY = 8

endpts = []

sheets = [ 0.0,-0.8  ]
def add_sheet(zh):
    for i in xrange(NX):
        Wp = newW - newW/(NX-1.0)
        p = 2.0*Wp/(NX-1.0)*i - Wp
        endpts.append([ [-restL, p,zh],[ restL, p,zh] ])
    for i in xrange(NY):
        Wp = newL - newL/(NX-1.0)
        p = 2.0*Wp/(NY-1.0)*i -Wp
        endpts.append([ [ p, -restW, zh],[p, restW,zh] ])

for z in sheets:
    add_sheet(z)
endpts.append([ [0, 0, 3.2],[ 0, 0, 2.9] ])

defaults = { 'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)) }
props = [ {} for i in endpts ]
props[-1] = { 'radius':2.0 }
Nelems = [ 20 for i in endpts ]
Nelems[-1] = 1

warp = Warp(endpts,props,defaults, Nelems, CurrentBeamProblem)

def initialize_sheet(startX,endX, startY,endY, restL,newL, height):
    for i in xrange(startX,endX):
        fib = warp.fibrils[i]
        fib.problem.fields['wx'].interpolate(Expression((
            " x[0]*sq",
            "0",
            "A1*sin(x[0]*p)",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0"),
            sq = -(restL-newL)/restL,
            p=np.pi/restL *(endY-startY)/2.0,
            A1=(-1.0 if i%2==0 else 1.0)*height
            ))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
        mdof = warp.spaces['W'].dofmap()
        warp.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]
    for i in xrange(startY,endY):
        fib = warp.fibrils[i]
        fib.problem.fields['wx'].interpolate(Expression((
            "0",
            " x[1]*sq",
            "A1*sin(x[1]*p)",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0"),
            sq = -(restL-newL)/restL,
            p=np.pi/restL *(endX-startX)/2.0,
            A1=(-1.0 if i%2==1 else 1.0)*height
            ))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
        mdof = warp.spaces['W'].dofmap()
        warp.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]

warp.output_states("post/impact/yarn_{0}_"+str(0)+".pvd",0)
warp.output_surfaces("post/impact/mesh_{0}_"+str(0)+".pvd",0)

for i in xrange(len(sheets)):
    base = i*(NX+NY)
    initialize_sheet(base,base+NX,base+NX,base+NX+NY, restL,newL, 0.2)

fib =  warp.fibrils[-1]
fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","-0.5",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
mdof = warp.spaces['W'].dofmap()
warp.fields['wx'].vector()[ mdof.part(len(warp.fibrils)-1).dofs() ] = fib.problem.fields['wx'].vector()[:]
warp.fields['wv'].vector()[ mdof.part(len(warp.fibrils)-1).dofs() ] = fib.problem.fields['wv'].vector()[:]

warp.output_states("post/impact/yarn_{0}_"+str(1)+".pvd",0)
warp.output_surfaces("post/impact/mesh_{0}_"+str(1)+".pvd",0)

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
