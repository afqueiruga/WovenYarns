#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from IPython import embed

from src.RKnew import RKbase,exRK
"""
Define the problem properties
"""
Tmax = 5.0

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
    'em_seebeck':0.1,
    'dissipation':0.01,
    'mu_alpha':-0.01
    }]

endpts = [ [[-1.0,0.0,0.0],[1.0,0.0,0.0]] ]


"""
Helper routine that creates a new decoupled warp
"""
def DecoupledSetup():
    warp = Warp(endpts, props, {}, [40], DecoupledProblem)

    """
    Boundary conditions on the updates
    """
    zeroW = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
    zeroV = Constant((0.0,0.0,0.0))
    zeroS = Constant(0.0)
    bound = CompiledSubDomain("near(x[0],-1.0) && on_boundary")
    bcR = MultiMeshDirichletBC(warp.spaces['W'], zeroW, bound)
    bcT = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
    bcV = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
    # Do the mechanical field
    def sys_mech(time):
        return warp.assemble_form('F','W') #,'AX','AV'],'W')
    def bcapp_mech(K,R,t,hold=False):
        print "Mech"
        if K!=None:
            bcR.apply(K,R)
        else:
            bcR.apply(R)
    rkf_mech = RKbase.RK_field(2, [warp.fields['wx'].vector(),
                                   warp.fields['wv'].vector()],
                                warp.assemble_form('M','W'),
                                sys_mech,bcapp_mech,warp.update)
    # Do the thermal field
    def sys_temp(time):
        return warp.assemble_form('FT','S') #,'AT'],'S')
    def bcapp_temp(K,R,t,hold=False):
        print "temp"
        if K!=None:
            bcT.apply(K,R)
        else:
            bcT.apply(R)
    rkf_temp = RKbase.RK_field(1,[warp.fields['T'].vector()],
                               warp.assemble_form('MT','S'),
                               sys_temp,bcapp_temp,warp.update)
    # Do the em field
    def sys_em(time):
        return warp.assemble_forms(['FE','AE'],'S')
    def bcapp_em(K,R,t,hold=False):
        print "yo"
        bcV.apply(K,R)
    rkf_em = RKbase.RK_field(0,[warp.fields['Vol'].vector()],
                             None,
                             sys_em,bcapp_em,warp.update)
    
    # Return the fields
    fields = rkf_temp
    return warp,[rkf_temp] #rkf_mech,rkf_em]



"""
Helper routine that cleans and solves the warp with a new method
"""
probes_mono = [ (np.array([0.0,0.0, 0.0],dtype=np.double),2,'wx'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),1,'wx'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),9,'wv'),
            (np.array([0.5,0.0, 0.0],dtype=np.double),10,'wv')]
probes_deco = [ (np.array([0.0,0.0, 0.0],dtype=np.double),2,'wx',9),
            (np.array([0.5,0.0, 0.0],dtype=np.double),1,'wx',9),
            (np.array([0.5,0.0, 0.0],dtype=np.double),9,'T',1),
            (np.array([0.5,0.0, 0.0],dtype=np.double),10,'Vol',1)]
    
probes = probes_deco
def solve(NT,stepclass,tag,scheme,warp,fields, rescache=None,outdir=None):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    time_series[0] = 0.0
    #init
    step = stepclass(h,scheme, fields)

    if outdir:
        warp.output_states(outdir+"/dirk_{0}_"+str(0)+".pvd",0)
        warp.output_solids(outdir+"/dirk_{0}_"+str(0)+".pvd",0)

    for t in xrange(NT):
        print "Step ",t,"/",NT
        times[t+1] = (t+1)*h
        step.march()
        if outdir and (t % (NT/100) ==0):
            warp.output_states(outdir+"/dirk_{0}_"+str(t+1)+".pvd",0)
            warp.output_solids(outdir+"/dirk_{0}_"+str(t+1)+".pvd",0)
        for g,p in enumerate(probes):
            ev = np.zeros(p[3])
            warp.fibrils[0].problem.fields[p[2]].eval(ev,p[0])
            time_series[t+1,g] = ev[p[1]]

    if rescache:
        f = open("rescache","a")
        f.write("{0} {1}".format(tag, NT))
        for g in xrange(4):
            f.write( "{0} ".format(time_series[-1,g] ) )
        f.write("\n")
        f.close()

    return tag,NT,h,times,time_series
tests = {}
warp,fields = DecoupledSetup()
R=solve(1000,exRK.exRK,"RK4",exRK.exRK_table["RK4"], warp,fields, None,"post/exrk/")
embed()
