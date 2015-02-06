from src import *

"""
Lets go all out and shoot a smaller fibril at a fabric sheet!
"""

restL = 2.0
newL = 1.5
restW = 2.0
newW = 1.5
NX = 10

endpts = []

for i in xrange(NX):
    endpts.append([ [-restL, restW*(NX/2-i),0.0],[ restL,restW*(NX/2-i),0.0] ])
defaults = { 'radius':0.2,
             'em_B':Constant((0.0,0.0,0.0)) }
props = [ {} for i in endpts ]
warp = Warp(endpts,props,defaults, CurrentBeamProblem)

def initialize_sheet(startX,endX, startY,endY restL,newL, height):
    for i in xrange(startX,endX):
        fib = warp.fibrils[i]
        fib.problem.fields['wx'].interpolate(Expression((
            "x[0] + x[0]*sq",
            "0",
            "A1*sin(x[0]*p)",
            "0",
            "-1",
            "-((A1*pow(p,2)*sin(x[0]*p))/sqrt(pow(A1,2)*pow(p,4)*pow(sin(x[0]*p),2)))",
            "0",
            "(A1*pow(p,2)*(1 + sq)*sin(x[0]*p))/sqrt(pow(A1,2)*pow(p,4)*pow(1 + sq,2)*pow(sin(x[0]*p),2))",
            "-1"),
            sq = -(restL-newL)/restL,
            p=np.pi/width *4.0,
            A1=(-1.0 if i%2==0 else 1.0)*2.1*height
            ))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "0.0")))
        mdof = warp.spaces['W'].dofmap()
        warp.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]

warp.output_states("post/impact/yarn_{0}_"+str(0)+".pvd",0)
warp.output_surfaces("post/impact/mesh_{0}_"+str(0)+".pvd",0)
