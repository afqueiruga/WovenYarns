#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from IPython import embed

"""
Set up the warp with a single fibril
"""

E = 10.0
nu = 0.0
B = 1.0
radius = 0.02
Phi = np.pi/4.0

props =  [{
    'mu':E/(2*(1 + nu)),
    'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
    'radius':radius,
    'rho':2.0,
    'em_B':Constant((B*np.cos(Phi),B*np.sin(Phi),0.0)),
    'dissipation':0.01
    }]

endpts = [ [[-1.0,0.0,0.0],[1.0,0.0,0.0]] ]
warp = Warp(endpts, props, {}, [40], MonolithicProblem)


"""
Boundary conditions on the velocity updates
"""
zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
bound = CompiledSubDomain("on_boundary")
# subs = MultiMeshSubSpace(warp.spaces['W'],0)
bcall = MultiMeshDirichletBC(warp.spaces['W'], zero, bound)
def apply_BCs(K,R,t,hold=False):
    bcall.apply(K,R)


"""
DIRK's assembling routine
"""
def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')


"""
Initialize routine
"""
def initialize():
    for i,fib in enumerate(warp.fibrils):
        fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "0.0")))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "x[0]/5.0")))
        mdof = warp.spaces['W'].dofmap()
        warp.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]



probes = [ (np.array([0.0,0.0, 0.0],dtype=np.double),2,'x'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),1,'x'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),9,'v'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),10,'v')]
weval = np.zeros(11)


def solve(order,NT):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    time_series[0] = 0.0
    initialize()
    dirk = DIRK_Monolithic(h,LDIRK[order], sys,warp.update,apply_BCs,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
    warp.output_states("post/dirk/dirk_{0}_"+str(0)+".pvd",0)
    warp.output_solids("post/dirk/mesh_{0}_"+str(0)+".pvd",0)
    for t in xrange(NT):
        dirk.march()
        warp.output_states("post/dirk/dirk_{0}_"+str(t+1)+".pvd",0)
        warp.output_solids("post/dirk/mesh_{0}_"+str(t+1)+".pvd",0)

        for g,p in enumerate(probes):
            if p[2]=='x':
                warp.fibrils[0].problem.fields['wx'].eval(weval,p[0])
            else:
                warp.fibrils[0].problem.fields['wv'].eval(weval,p[0])
            time_series[t+1,g] = weval[p[1]]
        times[t+1] = (t+1)*h
    return (times,time_series,h,order)


Tmax = 5.0
NTS = [100] #(50,301,50)
# NTS.append(500)
orders = [2]
all_series = [ ]

for order in orders:
    results = []
    for NT in NTS:
        results.append( solve(order,NT) )
    all_series.append( (order, results) )

def make_plots(all_series):
    for g in xrange(len(probes)):
        plt.figure()
        for series in all_series:
            for ts,ys,o,h in series[1]:
                plt.plot(ts,ys[:,g])
        plt.figure()
        exact = all_series[-1][1][-1][1][-1,g]
        for series in all_series:
            plt.loglog([ x[2] for x in series[1]],
                        [ np.abs(x[1][-1,g]-exact) for x in series[1] ],'-+')

def compute_convergence(all_series):
    import scipy.stats
    for g in xrange(len(probes)):
        exact = all_series[-1][1][-1][1][-1,g]
        for ix,series in enumerate(all_series):
        
            print (scipy.stats.linregress([ np.log(x[2]) for x in series[1][:(-1 if ix==len(all_series)-1 else -2)] ],
                                          [ np.log(np.abs(x[1][-1,g]-exact)) for x in series[1][:(-1 if ix==len(all_series)-1 else -2)] ]))[0],
        print ""
make_plots(all_series)
plt.show()

embed()

compute_convergence(all_series)
