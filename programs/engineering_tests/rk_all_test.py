#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from IPython import embed
import time

from src.RKnew import RKbase,exRK,imRK

set_log_level(CRITICAL)

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
# def DecoupledSetup():
warp = Warp(endpts, props, {}, [40], DecoupledProblem)

"""
Boundary conditions on the updates
"""
subR = MultiMeshSubSpace(warp.spaces['W'],0)
zeroW = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
zeroV = Constant((0.0,0.0,0.0))
zeroS = Constant(0.0)
bound = CompiledSubDomain(" on_boundary")
bcR = MultiMeshDirichletBC(subR, zeroV, bound)
bcT = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
bcV = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
# Do the mechanical field
def sys_mech(time,tang=False):
    if tang:
        return warp.assemble_forms(['F','AX','AV'],'W')
    else:
        return warp.assemble_form('F','W')
def bcapp_mech(K,R,t,hold=False):
    if K!=None:
        bcR.apply(K)
    if R!=None:
        bcR.apply(R)
rkf_mech = RKbase.RK_field(2, [warp.fields['wv'].vector(),
                                   warp.fields['wx'].vector()],
                                warp.assemble_form('M','W'),
                                sys_mech,bcapp_mech,warp.update)
    # Do the thermal field
def sys_temp(time,tang=False):
    if tang:
        return warp.assemble_forms(['FT','AT'],'S')
    else:
        return warp.assemble_form('FT','S')
def bcapp_temp(K,R,t,hold=False):
    if K!=None:
        bcT.apply(K)
    if R!=None:
        bcT.apply(R)
rkf_temp = RKbase.RK_field(1,[warp.fields['T'].vector()],
                               warp.assemble_form('MT','S'),
                               sys_temp,bcapp_temp,warp.update)
    # Do the em field
def sys_em(time,tang=False):
    return warp.assemble_forms(['FE','AE'],'S')
def bcapp_em(K,R,t,hold=False):
    bcV.apply(K,R)
rkf_em = RKbase.RK_field(0,[warp.fields['Vol'].vector()],
                             None,
                             sys_em,bcapp_em,warp.update)
    
    # Return the fields
fields = [rkf_mech,rkf_temp,rkf_em]
#    return warp,[rkf_temp] #rkf_mech,rkf_em]





"""
Build the monolothic warp
"""
warp_m =  Warp(endpts, props, {}, [40], MonolithicProblem)
subR_m = MultiMeshSubSpace(warp_m.spaces['W'],0)
subT_m = MultiMeshSubSpace(warp_m.spaces['W'],3)
subV_m = MultiMeshSubSpace(warp_m.spaces['W'],4)
# bcall_m = MultiMeshDirichletBC(warp_m.spaces['W'], zero, bound)
bcR_m = MultiMeshDirichletBC(subR_m, zeroV, bound)
bcT_m = MultiMeshDirichletBC(subT_m, zeroS, bound)
bcV_m = MultiMeshDirichletBC(subV_m, zeroS, bound)

def sys_m(time,tang=False):
    if tang:
        return warp_m.assemble_forms(['F','AX','AV'],'W')
    else:
        return warp_m.assemble_form('F','W')
def bcapp_m(K,R,t,hold=False):
    if K!=None:
        bcR_m.apply(K)
        bcT_m.apply(K)
        bcV_m.apply(K)
    if R!=None:
        bcR_m.apply(R)
        bcT_m.apply(R)
        bcV_m.apply(R)
rkf_mono = RKbase.RK_field(2,[warp_m.fields['wv'].vector(),
                              warp_m.fields['wx'].vector()],
                              warp_m.assemble_form('M','W'),
                              sys_m,bcapp_m,warp_m.update)
fields_mono = [rkf_mono]



"""
Helper routine that cleans and solves the warp with a new method
"""
probes_mono = [ (np.array([0.0,0.0, 0.0],dtype=np.double),2,'wx',11),
            (np.array([0.5,0.0, 0.0],dtype=np.double),1,'wx',11),
            (np.array([0.5,0.0, 0.0],dtype=np.double),9,'wv',11),
            (np.array([0.5,0.0, 0.0],dtype=np.double),10,'wv',11)]
probes_deco = [ (np.array([0.0,0.0, 0.0],dtype=np.double),2,'wx',9),
            (np.array([0.5,0.0, 0.0],dtype=np.double),1,'wx',9),
            (np.array([0.5,0.0, 0.0],dtype=np.double),0,'T',1),
            (np.array([0.5,0.0, 0.0],dtype=np.double),0,'Vol',1)]


def initialize_mono():
    for i,fib in enumerate(warp_m.fibrils):
        fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "0.0")))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",
                                       "0.0", "x[0]/5.0")))
        mdof = warp_m.spaces['W'].dofmap()
        warp_m.fields['wx'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp_m.fields['wv'].vector()[ mdof.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]
def initialize_deco():
    for i,fib in enumerate(warp.fibrils):
        fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0",)))
        fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
        fib.problem.fields['T'].interpolate(Expression(("0.0")))
        fib.problem.fields['Vol'].interpolate(Expression(("x[0]/5.0")))
        mdofW = warp.spaces['W'].dofmap()
        mdofS = warp.spaces['S'].dofmap()
        warp.fields['wx'].vector()[ mdofW.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdofW.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]
        warp.fields['T'].vector()[ mdofS.part(i).dofs() ] = fib.problem.fields['T'].vector()[:]
        warp.fields['Vol'].vector()[ mdofS.part(i).dofs() ] = fib.problem.fields['Vol'].vector()[:]


        
probes = probes_deco#mono
initer = initialize_deco#mono
def solver(NT,stepclass,tag,scheme,warp,fields, rescache=None,outdir=None):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    time_series[0] = 0.0
    initer()
    step = stepclass(h,scheme, fields)

    if outdir:
        # warp.output_states(outdir+"/dirk_{0}_"+str(0)+".pvd",0)
        warp.output_solids(outdir+"/solid_{0}_"+str(0)+".pvd",0)

    starttime = time.clock()
    for t in xrange(NT):
        times[t+1] = float(t+1.0)*h
        print "Step ",t,"/",NT, " at ",times[t+1], " h", h

        try:
            step.march()
        except:
            print "Failed to converge!"
            return
        if outdir and ((t+1) % (NT/100) ==0):
            # warp.output_states(outdir+"/dirk_{0}_"+str((t+1) / (NT/100))+".pvd",0)
            warp.output_solids(outdir+"/solid_{0}_"+str((t+1) / (NT/100))+".pvd",0)
        for g,p in enumerate(probes):
            ev = np.zeros(p[3])
            warp.fibrils[0].problem.fields[p[2]].eval(ev,p[0])
            time_series[t+1,g] = ev[p[1]]

    endtime = time.clock()
    if rescache:
        f = open(rescache,"a")
        f.write("{0} {1} {2} {3} ".format(tag, NT, h, endtime-starttime))
        for g in xrange(4):
            f.write( "{0} ".format(time_series[-1,g] ) )
        f.write("\n")
        f.close()
        warp.save(rescache+"_data/warp_{0}_{1}".format(tag,NT))

    return tag,NT,h,times,time_series
tests = [
    # [exRK.exRK,"FWEuler",exRK.exRK_table["FWEuler"],  [] ],
    # [exRK.exRK,"RK2-mid",exRK.exRK_table["RK2-mid"],  [1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4] ],
        # [exRK.exRK,"RK2-trap",exRK.exRK_table["RK2-trap"],  [6e3,7e3,8e3,9e3,1e4] ],

    # [exRK.exRK,"RK3-1",exRK.exRK_table["RK3-1"],  [1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4] ],
    [exRK.exRK,"RK4",exRK.exRK_table["RK4"],  [5e3] ],
    # [ imRK.DIRK,"BWEuler",imRK.LDIRK["BWEuler"], [1e3,2e3,4e3,5e3]],
    # [ imRK.DIRK,"LSDIRK2",imRK.LDIRK["LSDIRK2"], [100,200,300,400,500]],
    # [ imRK.DIRK,"LSDIRK3",imRK.LDIRK["LSDIRK3"], [100,200,300,400,500]]

]

import cProfile
for cl,tag,tab, NTS in tests:
    for NT in NTS:
        cProfile.run('solver(int(NT), cl,tag,tab, warp,fields, "data/rk_study/rk_pin3",None)','rescache')
        # R=solver(int(NT), cl,tag,tab, warp,fields, "data/rk_study/rk_pin3",None)


tests_mono = [
    # [ imRK.DIRK,"mBWEuler",imRK.LDIRK["BWEuler"], [1e3,2e3,4e3,5e3]],
    # [ imRK.DIRK,"mLSDIRK2",imRK.LDIRK["LSDIRK2"], [100,200,300,400,500]],
    [ imRK.DIRK,"mLSDIRK3",imRK.LDIRK["LSDIRK3"], [50]] #,200,300,400,500]]
    
]

#
# for cl,tag,tab, NTS in tests_mono:
#     for NT in NTS:
#         cProfile.run('solver(int(NT), cl,tag,tab, warp_m,fields_mono, "data/rk_study/rk_pin3",None)','rescache')
#         # R=solver(int(NT), cl,tag,tab, warp_m,fields_mono, "data/rk_study/rk_pin3",None)


# warp,fields = DecoupledSetup()
# R=solve(int(1e4),exRK.exRK,"RK4",exRK.exRK_table["RK4"], warp,fields, "exrk","post/exrk/")
embed()
