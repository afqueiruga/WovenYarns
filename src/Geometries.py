from dolfin import Expression, Function, Constant, assign
import numpy as np

from Warp import Warp

import Geometry_Curves

"""
Define routines for initializing textile geometries.
"""

def make_bases(center):
    E = center[1]- center[0]
    if E[1]==0.0 and E[2]==0.0:
        e1 = np.array([0.0,1.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][0])
        orientation=0
    elif E[0]==0.0 and E[2]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][1])
        orientation=1
    elif E[0]==0.0 and E[1]==0.0:
        e1 = np.array([1.0,0.0,0.0])
        e2 = np.array([0.0,1.0,0.0])
        setL = np.abs(center[0][2])
        orientation=2
    else:
        print "Error: Fibrils must be axis aligned! But I'm not going to stop."
        e1 = np.array([0.0,1.0,0.0])
        e2 = np.array([0.0,0.0,1.0])
        setL = np.abs(center[0][0])
        orientation=0
    return orientation,setL,e1,e2

class TextileGeometry():
    def __init__(self):
        self.nfibril = 0
        self.istart = 0
    def endpts(self,warp,istart):
        return []
    def contact_pairs(self):
        return None

class PlainWeave(TextileGeometry):
    def __init__(self, NX,restX,setX, NY,restY,setY, zpos,height):
        self.NX = NX
        self.restX = restX
        self.setX = setX
        self.NY = NY
        self.restY = restY
        self.setY = setY

        self.zpos = zpos
        self.height = height

        self.nfibril = NX+NY
        self.istart = 0
    def endpts(self):
        endpts = []
        for i in xrange(self.NX):
            Wp = self.setY - self.setY/(self.NX-1.0)
            p = 2.0*Wp/(self.NX-1.0)*i - Wp
            endpts.append([ [-self.restX, p,self.zpos],[ self.restX, p,self.zpos] ])
        for i in xrange(self.NY):
            Wp = self.setX - self.setX/(self.NY-1.0)
            p = 2.0*Wp/(self.NY-1.0)*i -Wp
            endpts.append([ [ p, -self.restY, self.zpos],[p, self.restY,self.zpos] ])
        return endpts

    def initialize(self, warp, istart):
        self.istart = istart
        for i in xrange(istart,istart+self.NX):
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
                sq = -(self.restX-self.setX)/self.restX,
                p=np.pi/self.restX *(self.NY)/2.0,
                A1=(-1.0 if i%2==0 else 1.0)*self.height
                ))
            fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                             "0.0"," 0.0","0.0",
                                                             "0.0","0.0","0.0")))
        for i in xrange(istart+self.NX,istart+self.NX+self.NY):
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
                sq = -(self.restY-self.setY)/self.restY,
                p=np.pi/self.restY *(self.NX)/2.0,
                A1=(-1.0 if i%2==1 else 1.0)*self.height
                ))
            fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                                             "0.0"," 0.0","0.0",
                                                             "0.0","0.0","0.0")))
    def contact_pairs(self):
        pairs = []
        for A in xrange(self.NX):
            for B in xrange(self.NY):
                pairs.append((self.istart+A,self.istart+self.NX+B))
        return pairs


class PlainWeaveFibrils(TextileGeometry):
    def __init__(self, NX,restX,setX, NY,restY,setY, zpos,height, pattern,Dia):
        self.NX = NX
        self.restX = restX
        self.setX = setX
        self.NY = NY
        self.restY = restY
        self.setY = setY

        self.zpos = zpos
        self.height = height

        self.pattern = pattern
        self.Dia = Dia

        self.nfibril = (NX+NY)*np.sum(pattern)
        self.istart = 0
    def endpts(self):
        endpts = []
        for i in xrange(self.NX):
            p = -self.setY + 2.0*self.setY/(self.NX)*(i+0.5)
            endpts.extend( PackedYarn([ [-self.restX, p,self.zpos],[ self.restX, p,self.zpos] ],
                                      self.pattern,self.Dia) )
        for i in xrange(self.NY):
            p = -self.setX + 2.0*self.setX/(self.NY)*(i+0.5)
            endpts.extend( PackedYarn([ [ p, -self.restY, self.zpos],[p, self.restY,self.zpos] ],
                                      self.pattern,self.Dia))
        return endpts
    
    def initialize(self,warp,istart):
        self.istart = istart
        for i in xrange(istart,istart+self.NX):
            for j in xrange(np.sum(self.pattern)):
                fib = warp.fibrils[i*np.sum(self.pattern)+j]
                qhh = ( (
                     " x[0]*sq",
                     "0",
                     "A1*sin(x[0]*p)"
                    ),
                    ( "0","0","0"),
                    ( "0","0","0")
                    )
                temp_field = Function(fib.problem.spaces['V'])
                for fix in xrange(3):
                    temp_field.interpolate(Expression(
                        qhh[fix],
                        sq = -(self.restX-self.setX)/self.restX,
                        p=np.pi/self.restX *(self.NY)/2.0,
                        A1=(-1.0 if i%2==0 else 1.0)*self.height
                    ))
                    assign(fib.problem.fields['wx'].sub(fix),temp_field)
                    temp_field.interpolate(Constant((0.0,0.0,0.0)))
                    assign(fib.problem.fields['wv'].sub(fix),temp_field)
        for i in xrange(istart+self.NX,istart+(self.NX+self.NY)):
            for j in xrange(np.sum(self.pattern)):
                fib = warp.fibrils[i*np.sum(self.pattern)+j]
                vqhh = ((
                     "0",
                    " x[1]*sq",
                    "A1*sin(x[1]*p)"
                    ),
                    ( "0","0","0"),
                    ( "0","0","0")
                    )
                temp_field = Function(fib.problem.spaces['V'])
                for fix in xrange(3):
                    temp_field.interpolate(Expression(
                        vqhh[fix],
                        sq = -(self.restY-self.setY)/self.restY,
                        p=np.pi/self.restY *(self.NX)/2.0,
                        A1=(-1.0 if i%2==1 else 1.0)*self.height
                    ))
                    assign(fib.problem.fields['wx'].sub(fix), temp_field)
                    temp_field.interpolate(Constant((0.0,0.0,0.0)))
                    assign(fib.problem.fields['wv'].sub(fix),temp_field)                
                        
    def contact_pairs(self):
        lenp = np.sum(self.pattern)
        pairs = []
        for A in xrange(self.NX):
            stA = self.istart+lenp*A
            pairs.extend([ (stA+j,stA+i) for j in xrange(lenp) for i in xrange(j+1,lenp)  ])
            for B in xrange(self.NY):
                stB = self.istart+self.NX+lenp*B
                pairs.extend([ (stA+j,stB+i) for j in xrange(lenp) for i in xrange(lenp)  ])
        for B in xrange(self.NY):
            stB = self.istart+self.NX+lenp*B
            pairs.extend([ (stB+j,stB+i) for j in xrange(lenp) for i in xrange(j+1,lenp)  ])
            
        return pairs
        
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

class CoiledYarn(TextileGeometry):
    def __init__(self,center,Rstart, restL, Ns,Dias,NTurns):
        center[0] = np.array(center[0],dtype=np.double)
        center[1] = np.array(center[1],dtype=np.double)
        self.center = center
        orientation,self.setL,self.e1,self.e2 = make_bases(self.center)
        self.Rstart = Rstart
        self.restL = restL
        self.Ns = Ns
        self.Dias = Dias
        self.NTurns = NTurns
        
    def endpts(self):
        endpts = []
        rad = self.Rstart
        for n,d in zip(self.Ns,self.Dias):
            for i in xrange(n):
                p1 = rad*np.cos(2.0*np.pi*float(i)/float(n))
                p2 = rad*np.sin(2.0*np.pi*float(i)/float(n))
                endpts.append([self.restL/self.setL*self.center[0]+p1*self.e1+p2*self.e2,
                               self.restL/self.setL*self.center[1]+p1*self.e1+p2*self.e2])
            rad += d
        return endpts
    
    def initialize(self,warp,istart):
        ist = istart
        rad = self.Rstart
        for n,d,NT in zip(self.Ns,self.Dias,self.NTurns):
            for i in xrange(n):
                apply_helix(warp.fibrils[ist+i], self.restL,self.setL,
                            NT, d/2.0+rad,d/2.0+rad, float(i)/float(n))
            ist += n
            rad += d
            
    def contact_pairs():
        return None

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



class StockinetteFibrils(TextileGeometry):
    def __init__(self, restL,setL, N, width, Ns, Dias, height):
        self.restL = restL
        self.setL = setL
        self.N = N
        self.width = width
        self.Ns = Ns
        self.Dias = Dias
        self.e1 = np.array([0.0,1.0,0.0])
        self.e2 = np.array([0.0,0.0,1.0])
        self.height = height
        
    def endpts(self):
        endpts = []
        for i in xrange(self.N):
            X = -self.width + 2.0*self.width/(self.N)*(i+0.5)
            rad = 0.0
            for n,d in zip(self.Ns,self.Dias):
                for i in xrange(n):
                    p1 = 0.0 #rad*np.cos(2.0*np.pi*float(i)/float(n))
                    p2 = 0.0 #rad*np.sin(2.0*np.pi*float(i)/float(n))
                    endpts.append([np.array([-self.restL,X,0])+p1*self.e1+p2*self.e2,
                                   np.array([ self.restL,X,0])+p1*self.e1+p2*self.e2])
                rad += d
        return endpts

    def initialize(self,warp,istart):
        self.istart = istart
        for i in xrange(istart,istart+self.N):
            rad = 0.0
            ist = 0
            for n,d in zip(self.Ns,self.Dias):
                for j in xrange(n):
                    print j
                    fib = warp.fibrils[i*np.sum(self.Ns)+ist]
                    qhh = Geometry_Curves.qhh_stockinette2
                    temp_field = Function(fib.problem.spaces['V'])
                    p1 = rad*np.cos(2.0*np.pi*float(j)/float(n))
                    p2 = rad*np.sin(2.0*np.pi*float(j)/float(n))
                    for fix in xrange(3):
                        temp_field.interpolate(Expression(
                            qhh[fix],
                            sq = -(self.restL-self.setL)/self.restL,
                            p=np.pi/self.restL *(4.0),
                            o=self.setL/3.5,
                            A1=1.3*self.width/self.N,A2=self.width/self.N,
                            y1=p1, y2=p2
                        ))
                        assign(fib.problem.fields['wx'].sub(fix),temp_field)
                        temp_field.interpolate(Constant((0.0,0.0,0.0)))
                        assign(fib.problem.fields['wv'].sub(fix),temp_field)
                    ist += 1
                rad += d


