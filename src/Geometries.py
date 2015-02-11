from dolfin import Expression
import numpy as np

from Warp import Warp

"""
Define routines for initializing textile geometries.
"""

def PlainWeave_endpts(NX,restX,setX, NY,restY,setY, zpos,height):
    endpts = []
    for i in xrange(NX):
        Wp = setY - setY/(NX-1.0)
        p = 2.0*Wp/(NX-1.0)*i - Wp
        endpts.append([ [-restX, p,zpos],[ restX, p,zpos] ])
    for i in xrange(NY):
        Wp = setX - setX/(NY-1.0)
        p = 2.0*Wp/(NY-1.0)*i -Wp
        endpts.append([ [ p, -restY, zpos],[p, restY,zpos] ])
    return endpts

def PlainWeave_initialize(warp, istart, NX,restX,setX, NY,restY,setY, zpos,height):
    for i in xrange(istart,istart+NX):
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
            sq = -(restX-setX)/restX,
            p=np.pi/restX *(NY)/2.0,
            A1=(-1.0 if i%2==0 else 1.0)*height
            ))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
    for i in xrange(istart+NX,istart+NX+NY):
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
            sq = -(restY-setY)/restY,
            p=np.pi/restY *(NX)/2.0,
            A1=(-1.0 if i%2==1 else 1.0)*height
            ))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))



