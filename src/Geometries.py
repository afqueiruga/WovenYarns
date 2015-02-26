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







def PlainWeaveFibrils_endpts(NX,restX,setX, NY,restY,setY, zpos,height, pattern,Dia):
    endpts = []
    for i in xrange(NX):
        p = -setY + 2.0*setY/(NX)*(i+0.5)
        endpts.extend( PackedYarn([ [-restX, p,zpos],[ restX, p,zpos] ], pattern,Dia) )
    for i in xrange(NY):
        p = -setX + 2.0*setX/(NY)*(i+0.5)
        endpts.extend( PackedYarn([ [ p, -restY, zpos],[p, restY,zpos] ], pattern,Dia))
    return endpts


def PlainWeaveFibrils_initialize(warp, istart, NX,restX,setX, NY,restY,setY, zpos,height, pattern,Dia):
    for i in xrange(istart,istart+NX):
        for j in xrange(np.sum(pattern)):
            fib = warp.fibrils[i*np.sum(pattern)+j]
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
    for i in xrange(istart+NX,istart+(NX+NY)):
        for j in xrange(np.sum(pattern)):

            fib = warp.fibrils[i*np.sum(pattern)+j]
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


def PackedYarn(centers, pattern,  Dia):
    
    centers[0] = np.array(centers[0])
    centers[1] = np.array(centers[1])
    E = centers[1]- centers[0]
    if E[1]==0.0 and E[2]==0.0:
        e1 = np.array([0.0,1.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
    elif E[0]==0.0 and E[2]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
    elif E[0]==0.0 and E[1]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,1.0,0.0])
    else:
        print "Error: Fibrils must be axis aligned! But I'm not going to stop."
        orientation=0
    endpts = []
    for i,row in enumerate(pattern):
        p2 = Dia*(1.0*i-(len(pattern)-1)/2.0)
        for j in xrange(row):
            p1 = Dia*(1.0*j-(row-1)/2.0)
            endpts.append([centers[0]+p1*e1+p2*e2, centers[1]+p1*e1+p2*e2])
    return endpts


def CoiledYarn_endpts(center, Rstart, restL, Ns, Dias,NTurns):
    center[0] = np.array(center[0],dtype=np.double)
    center[1] = np.array(center[1],dtype=np.double)
    E = center[1]- center[0]
    if E[1]==0.0 and E[2]==0.0:
        e1 = np.array([0.0,1.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][0])
    elif E[0]==0.0 and E[2]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][2])
    elif E[0]==0.0 and E[1]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,1.0,0.0])
        setL = np.abs(center[0][2])
    else:
        print "Error: Fibrils must be axis aligned! But I'm not going to stop."
        orientation=0
    endpts = []
    rad = Rstart
    for n,d in zip(Ns,Dias):
        for i in xrange(n):
            p1 = rad*np.cos(2.0*np.pi*float(i)/float(n))
            p2 = rad*np.sin(2.0*np.pi*float(i)/float(n))
            endpts.append([restL/setL*center[0]+p1*e1+p2*e2, restL/setL*center[1]+p1*e1+p2*e2])
        rad += d
    return endpts

def CoiledYarn_initialize(warp, istart, center,Rstart, restL, Ns,Dias,NTurns):
    ist = istart
    rad = Rstart
    center[0] = np.array(center[0])
    center[1] = np.array(center[1])
    E = center[1]- center[0]
    if E[1]==0.0 and E[2]==0.0:
        e1 = np.array([0.0,1.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][0])
    elif E[0]==0.0 and E[2]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][2])
    elif E[0]==0.0 and E[1]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,1.0,0.0])
        setL = np.abs(center[0][2])
    else:
        print "Error: Fibrils must be axis aligned! But I'm not going to stop."
        orientation=0
    for n,d,NT in zip(Ns,Dias,NTurns):
        for i in xrange(n):
            apply_helix(warp.fibrils[ist+i], restL,setL, NT, d/2.0+rad,d/2.0+rad, float(i)/float(n))
        ist += n
        rad += d


def apply_helix(fib, restX,setX,NTURN,A1,A2,phase):
    fib.problem.fields['wx'].interpolate(Expression((
        "x[0]*sq",
"A1*sin(x[0]*p + w)-x[1]",
"A2*cos(x[0]*p + w)-x[2]",
"-((A1*A2*pow(p,3))/sqrt(pow(p,4)*(pow(A1,2)*pow(A2,2)*pow(p,2) + pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*pow(1 + sq,2)*pow(sin(x[0]*p + w),2))))",
"-1 + (A2*pow(p,2)*(1 + sq)*cos(x[0]*p + w))/sqrt(pow(p,4)*(pow(A1,2)*pow(A2,2)*pow(p,2) + pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*pow(1 + sq,2)*pow(sin(x[0]*p + w),2)))",
"-((A1*pow(p,2)*(1 + sq)*sin(x[0]*p + w))/sqrt(pow(p,4)*(pow(A1,2)*pow(A2,2)*pow(p,2) + pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*pow(1 + sq,2)*pow(sin(x[0]*p + w),2))))",
"-((pow(A1,2) - pow(A2,2))*pow(p,3)*(1 + sq)*sin(2*(x[0]*p + w)))/(2.*sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(cos(x[0]*p + w),2) + pow(A2,2)*pow(p,2)*pow(sin(x[0]*p + w),2))*sqrt(pow(p,4)*(pow(A1,2)*pow(A2,2)*pow(p,2) + pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*pow(1 + sq,2)*pow(sin(x[0]*p + w),2))))",
"(A1*pow(p,2)*(pow(A2,2)*pow(p,2) + pow(1 + sq,2))*sin(x[0]*p + w))/(sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(cos(x[0]*p + w),2) + pow(A2,2)*pow(p,2)*pow(sin(x[0]*p + w),2))*sqrt(pow(p,4)*(pow(A1,2)*pow(A2,2)*pow(p,2) + pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*pow(1 + sq,2)*pow(sin(x[0]*p + w),2))))",
"(A2*pow(p,2)*(pow(A1,2)*pow(p,2) + pow(1 + sq,2))*cos(x[0]*p + w) - sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(cos(x[0]*p + w),2) + pow(A2,2)*pow(p,2)*pow(sin(x[0]*p + w),2))*sqrt(pow(p,4)*(pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*(pow(A2,2)*pow(p,2) + pow(1 + sq,2)*pow(sin(x[0]*p + w),2)))))/(sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(cos(x[0]*p + w),2) + pow(A2,2)*pow(p,2)*pow(sin(x[0]*p + w),2))*sqrt(pow(p,4)*(pow(A2,2)*pow(1 + sq,2)*pow(cos(x[0]*p + w),2) + pow(A1,2)*(pow(A2,2)*pow(p,2) + pow(1 + sq,2)*pow(sin(x[0]*p + w),2)))))"
        ),
        sq = -(restX-setX)/restX,
        p=np.pi/restX *(NTURN)/2.0,
        A1=A1,
        A2=A2,
        w=phase*np.pi*2.0
        ))
    fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                     "0.0"," 0.0","0.0",
                                                     "0.0","0.0","0.0")))
