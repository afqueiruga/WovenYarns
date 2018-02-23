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
Tmax = 2.5

E = 10.0
nu = 0.0
B = 1.0
radius = 0.02
Phi = np.pi/4.0

sigma = 1.0
Ryarn = 2.0/(sigma*np.pi* (radius**2.) )
REXT = Ryarn
VAPP = - (1.0) / ( Ryarn/(REXT+Ryarn) )

props =  [{
    'mu':E/(2*(1 + nu)),
    'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
    'radius':radius,
    'rho':2.0,
    'em_B':Constant((B*np.cos(Phi),B*np.sin(Phi),0.0)),
    'em_seebeck':0.1,
    'dissipation':0.01,
    'mu_alpha':-0.01,

    'em_sig':sigma,
    'em_bc_J_1': VAPP/(REXT* (np.pi*radius**2.) ),
    'em_bc_r_1': 1.0/(REXT* (np.pi*radius**2.) )
    }]

endpts = [ [[-1.0,0.0,0.0],[1.0,0.0,0.0]] ]

# print VAPP/(REXT* (np.pi*radius**2.) )
# print 1.0/(REXT* (np.pi*radius**2.) )
# exit()

"""
Helper routine that creates a new decoupled warp
"""
# def DecoupledSetup():
warp = Warp(endpts, props, {}, [40], DecoupledProblem, order=(1,1))

"""
Boundary conditions on the updates
"""
zeroW = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
zeroV = Constant((0.0,0.0,0.0))
zeroS = Constant(0.0)
bound = CompiledSubDomain(" on_boundary ")
boundV = CompiledSubDomain(" on_boundary && near(x[0],-1.0)")
subQ = MultiMeshSubSpace(warp.spaces['W'],0)
subH1 = MultiMeshSubSpace(warp.spaces['W'],1)
subH2 = MultiMeshSubSpace(warp.spaces['W'],2)
bcQ = MultiMeshDirichletBC(subQ, zeroV, bound)
bcH1 = MultiMeshDirichletBC(subH1, zeroV, bound)
bcH2 = MultiMeshDirichletBC(subH2, zeroV, bound)
bcT = MultiMeshDirichletBC(warp.spaces['S'], zeroS, bound)
bcV = MultiMeshDirichletBC(warp.spaces['S'], zeroS, boundV)
# Do the mechanical field
def sys_mech(time,tang=False):
    if tang:
        return warp.assemble_forms(['F','AX','AV'],'W')
    else:
        return warp.assemble_form('F','W')
def bcapp_mech(K,R,t,hold=False):
    if K!=None:
        bcQ.apply(K)
        bcH1.apply(K)
        bcH2.apply(K)
    if R!=None:
        bcQ.apply(R)
        bcH1.apply(R)
        bcH2.apply(R)
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
warp_m =  Warp(endpts, props, {}, [40], MonolithicProblem, order=(1,1))
subQ_m = MultiMeshSubSpace(warp_m.spaces['W'],0)
subH1_m = MultiMeshSubSpace(warp_m.spaces['W'],1)
subH2_m = MultiMeshSubSpace(warp_m.spaces['W'],2)
subT_m = MultiMeshSubSpace(warp_m.spaces['W'],3)
subV_m = MultiMeshSubSpace(warp_m.spaces['W'],4)
# bcall_m = MultiMeshDirichletBC(warp_m.spaces['W'], zero, bound)
bcQ_m = MultiMeshDirichletBC(subQ_m, zeroV, bound)
bcH1_m = MultiMeshDirichletBC(subH1_m, zeroV, bound)
bcH2_m = MultiMeshDirichletBC(subH2_m, zeroV, bound)
bcT_m = MultiMeshDirichletBC(subT_m, zeroS, bound)
bcV_m = MultiMeshDirichletBC(subV_m, zeroS, boundV)

def sys_m(time,tang=False):
    if tang:
        return warp_m.assemble_forms(['F','AX','AV'],'W')
    else:
        return warp_m.assemble_form('F','W')
def bcapp_m(K,R,t,hold=False):
    if K!=None:
        bcQ_m.apply(K)
        bcH1_m.apply(K)
        bcH2_m.apply(K)
        bcT_m.apply(K)
        bcV_m.apply(K)
    if R!=None:
        bcQ_m.apply(R)
        bcH1_m.apply(R)
        bcH2_m.apply(R)
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
                                       "0.0", "1.0*(x[0]+1.0)/2.0")))
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
        fib.problem.fields['Vol'].interpolate(Expression(("1.0*(x[0]+1.0)/2.0")))
        mdofW = warp.spaces['W'].dofmap()
        mdofS = warp.spaces['S'].dofmap()
        warp.fields['wx'].vector()[ mdofW.part(i).dofs() ] = fib.problem.fields['wx'].vector()[:]
        warp.fields['wv'].vector()[ mdofW.part(i).dofs() ] = fib.problem.fields['wv'].vector()[:]
        warp.fields['T'].vector()[ mdofS.part(i).dofs() ] = fib.problem.fields['T'].vector()[:]
        warp.fields['Vol'].vector()[ mdofS.part(i).dofs() ] = fib.problem.fields['Vol'].vector()[:]


        

def solver(NT,stepclass,tag,scheme,warp,fields, rescache=None,outdir=None):
    h = Tmax/NT
    time_series = np.zeros((NT+1,len(probes)))
    times = np.zeros(NT+1)
    time_series[0] = 0.0
    initer()
    step = stepclass(h,scheme, fields)
    # embed()
    if outdir:
        # warp.output_states(outdir+"/dirk_{0}_"+str(0)+".pvd",0)
        warp.output_solids(outdir+"/solid_{0}_"+str(0)+".pvd",0)

    for g,p in enumerate(probes):
            ev = np.zeros(p[3])
            warp.fibrils[0].problem.fields[p[2]].eval(ev,p[0])
            time_series[0,g] = ev[p[1]]
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
        warp.dump_fibril_io_fields(rescache+"_data/io_{0}_{1}".format(tag,NT))

    return tag,NT,h,times,time_series

tests = [
    # [exRK.exRK,"FWEuler",exRK.exRK_table["FWEuler"],  [100,200,300,400,500,600,700,800,900,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4]],
    # [exRK.exRK,"RK2-mid",exRK.exRK_table["RK2-mid"],  [100,200,300,400,500,600,700,800,900,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4] ],
        # [exRK.exRK,"RK2-trap",exRK.exRK_table["RK2-trap"],  [2e4,3e4,4e4,5e4,6e4] ],

    # [exRK.exRK,"RK3-1",exRK.exRK_table["RK3-1"],  [100,200,300,400,500,600,700,800,900,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4] ],
    [exRK.exRK,"RK4",exRK.exRK_table["RK4"],  [1e4] ],
    # [ imRK.DIRK,"BWEuler",imRK.LDIRK["BWEuler"], [10,20,30,40,50,60,70,80,90]],
    # [ imRK.DIRK,"LSDIRK2",imRK.LDIRK["LSDIRK2"], [2000,3000,4000,5000,6000,7000]],
    # [ imRK.DIRK,"LSDIRK3",imRK.LDIRK["LSDIRK3"], [100]],
    # [ imRK.DIRK,"ImTrap",imRK.LDIRK["ImTrap"], [6e3,7e3,8e3,9e3,1e4]],
    # [ imRK.DIRK,"DIRK2",imRK.LDIRK["DIRK2"], [100,200,300,400,500,600,700,800,900,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4]],
    # [ imRK.DIRK,"DIRK3",imRK.LDIRK["DIRK3"], [100,200,300,400,500,600,700,800,900,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4]]

]
probes = probes_deco
initer = initialize_deco
import cProfile
for cl,tag,tab, NTS in tests:
    for NT in NTS:
        # cProfile.run('solver(int(NT), cl,tag,tab, warp,fields, "data/rk_study/rk_pin3",None)','rescache')
        R=solver(int(NT), cl,tag,tab, warp,fields, "data/rk_study/REVISION",None)#"post/exrk/")

tests_mono = [
    # [ imRK.DIRK,"mBWEuler",imRK.LDIRK["BWEuler"], [2250,2500,2750,3000,3500,4000]],
    # [ imRK.DIRK,"mLSDIRK2",imRK.LDIRK["LSDIRK2"], [1000,2000,3000,4000,5000,6000,7000]], 
    # [ imRK.DIRK,"mLSDIRK3",imRK.LDIRK["LSDIRK3"], [100]]
]
probes = probes_mono
initer = initialize_mono
for cl,tag,tab, NTS in tests_mono:
    for NT in NTS:
        # cProfile.run('solver(int(NT), cl,tag,tab, warp_m,fields_mono, "data/rk_study/rk_pin3",None)','rescache')
        R=solver(int(NT), cl,tag,tab, warp_m,fields_mono, "data/rk_study/clamplong",None) 


def plot_fields():
    plt.close('all')
    font = {'size'   : 10}
    matplotlib.rc('font', **font)
    fig = plt.figure()
    fig.set_size_inches(4,3,dpi=320)
    plt.xlabel('Time t (ms)')
    plt.ylabel('Vertical displacement y(0) (mm)')
    plt.plot(R[3],R[4][:,0])
    fig.tight_layout()
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/longer_y0.pdf")
    fig = plt.figure()
    fig.set_size_inches(4,3,dpi=320)
    plt.xlabel('Time t (ms)')
    plt.ylabel('Lateral displacement z(0.5) (mm)')
    plt.plot(R[3],R[4][:,1])
    fig.tight_layout()
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/longer_zhalf.pdf")
    fig = plt.figure()
    fig.set_size_inches(4,3,dpi=320)
    plt.xlabel('Time t (ms)')
    plt.ylabel('Temperature T(0.5) (K)')
    plt.plot(R[3],R[4][:,2])
    fig.tight_layout()
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/longer_Thalf.pdf")
    fig = plt.figure()
    fig.set_size_inches(4,3,dpi=320)
    plt.xlabel('Time t (ms)')
    plt.ylabel('Voltage V(0.5) (V)')
    plt.plot(R[3],R[4][:,3])
    fig.tight_layout()
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/longer_Vhalf.pdf")


# embed()
