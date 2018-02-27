import numpy as np
import matplotlib
from matplotlib import pylab as plt
import copy

from collections import defaultdict
from itertools import cycle

datadirs = ["data/rk_study/clamplong"]
# linclamp is the good but short one

loose = []
for di in datadirs:
    f = open(di)
    for l in f:
        if l[0]=="#":
            continue
        s = l.split()
        loose.append([s[0],np.array(s[1:],np.double)])
    f.close()

dd = defaultdict(lambda : [])
for l in loose:
    dd[l[0]].append(l[1])
for k in dd.iterkeys():
    dd[k] = np.vstack(dd[k])
    dd[k] = dd[k][ dd[k][:,0].argsort() ]

bestkey = "RK4"
best = np.load(datadirs[0]+"_data/io_{0}_{1}.npz".format(bestkey,int(dd[bestkey][-1,0])))
field_labels = [
    'f0_q',
    'f0_vq',
    'f0_h1',
    'f0_vh1',
    'f0_h2',
    'f0_vh2',
    'f0_T',
    'f0_Vol']
label_keys = {
    'f0_Vol':'Vol',
    'f0_vq':'vr',
    'f0_vh1':'vg1',
    'f0_vh2':'vg2',
    'f0_q':'r',
    'f0_h1':'g1',
    'f0_h2':'g2',
    'f0_T':'T'}

def compare_nodal(best,targ):
    return [ np.linalg.norm(best[k]-targ[k])/np.linalg.norm(best[k]) for k in field_labels ]

ee = defaultdict(lambda : [])
for k in dd:
    ee[k] = np.zeros([dd[k].shape[0],len(field_labels)],dtype=np.double)
    if k[0]=='m': # I'm going on the hopefully good assumption that the decop. and mono. methods have the same exact solutions
        kludge=k#[1:]
    else:
        kludge=k # But it doesn't work
    for i,entry in enumerate(dd[kludge]):
        nodal = np.load(datadirs[0]+"_data/io_{0}_{1}.npz".format(kludge,int(entry[0])))
        ee[k][i,:]=compare_nodal(best,nodal)


best_probes = dd[bestkey][-1,-4:]

#
# Input truncations
#
import copy
td = copy.deepcopy(dd)
te = copy.deepcopy(ee)
def trunk(k,n):
    td[k]=td[k][n:,:]
    te[k]=te[k][n:,:]

trunk("RK2-mid",3)
trunk("RK3-1",1)
trunk("LSDIRK2",27)
trunk("LSDIRK3",25)
# trunk("FWEuler",5)
trunk("DIRK2",21)
trunk("DIRK3",21)
trunk("ImTrap",22)
# trunk("mBWEuler",6)
# trunk("mLSDIRK2",4)
# trunk("mLSDIRK3",5)

#
# Do some renaming
#
def rename(d,n,o):
    d[n]=d[o]
    d.pop(o,None)
def rerename(n,o):
    rename(dd,n,o)
    rename(ee,n,o)
    rename(td,n,o)
    rename(te,n,o)
rerename("ImMid","ImTrap")


colors = cycle("rgbycmk")
markers = cycle("+x*^do")

colorkey = defaultdict(lambda :colors.next())
markerkey = defaultdict(lambda :markers.next())
def make_probe_plots(dd,ee):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 11}
    matplotlib.rc('font', **font)
    for i,f in enumerate([ 'y(0)','z(0.5)','T(0.5)','V(0.5)' ]):
        plt.figure()
        plt.xlabel("Logarithm of time step size log(h)")
        plt.ylabel("Logarithm of error in "+f)
        for k in dd:
            plt.loglog( dd[k][:,1], np.abs(best_probes[i]-dd[k][:,3+i]), markerkey[k]+'-',color=colorkey[k],label=k)
        plt.legend(loc=2)
def make_error_plots(dd,ee,td,te,subit=None):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 10}
    matplotlib.rc('font', **font)
    if subit:
        fig,ax = plt.subplots(nrows=subit[0],ncols=subit[1])
        
        fig.set_size_inches(8,10,dpi=320)

    for i,f in enumerate(field_labels):
        if subit:
            pp = ax[i/subit[1],i%subit[1]]
            pp.set_xlabel("Logarithm of time step size log(h)")
            pp.set_ylabel("Logarithm of error in "+label_keys[f])
        else:
            pp = plt
            plt.figure(figsize=(8,10))
            plt.xlabel("Logarithm of time step size log(h)")
            plt.ylabel("Logarithm of error in "+label_keys[f])
        for k in ee:
            if k[0]=='m': # Kludge to skip replotting mono's
                continue
            pp.loglog( dd[k][:,1], ee[k][:,i], '-',color=colorkey[k])
            pp.loglog( td[k][:,1], te[k][:,i], markerkey[k]+'-',color=colorkey[k],label=k)
        if i==0:
            pp.legend(bbox_to_anchor=(0., 1.05, 2., .105), loc=3,
                ncol=3, mode="expand", borderaxespad=0.)
    if subit:
        fig.tight_layout(rect=(0,0,1,0.92))
    # plt.show()
    
def make_sum_error_plots(dd,ee,td,te):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 9}
    matplotlib.rc('font', **font)
    fig = plt.figure()
    fig.set_size_inches(6,5,dpi=320)
    plt.xlabel("Logarithm of time step size, log(h)")
    plt.ylabel("Logarithm of error in all fields, log||e||")
    for k in ee:
        if k[0]=='m': # Kludge to skip replotting mono's
            continue
        plt.loglog( dd[k][:,1], np.linalg.norm(ee[k],axis=1), '-',color=colorkey[k])
        plt.loglog( td[k][:,1], np.linalg.norm(te[k],axis=1), markerkey[k]+'-',color=colorkey[k],label=k)
    plt.legend(loc=4,ncol=2)
    fig.tight_layout()
    # plt.show()
    
def make_time_plots(dd,ee, subit=None):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 10}
    matplotlib.rc('font', **font)
    if subit:
        fig,ax = plt.subplots(nrows=subit[0],ncols=subit[1])
    for i,f in enumerate(field_labels):
        if subit:
            pp = ax[i/subit[1],i%subit[1]]
        else:
            pp = plt
            plt.figure()
        
        pp.set_xlabel("Logarithm of Runetime (log(s))")
        pp.set_ylabel("Logarithm of error in "+f[3:])
        for k in ee:
            x,y = dd[k][:,2], ee[k][:,i]
            if k==bestkey:
                x = dd[k][:-1,2]
                y = ee[k][:-1,i]
            pp.loglog( x,y,markerkey[k]+'-',color=colorkey[k],label=k)
    pp.legend()
    # plt.show()
def make_sum_time_plots(dd,ee, mets=None):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 9}
    matplotlib.rc('font', **font)
    fig = plt.figure()
    fig.set_size_inches(6,5,dpi=320)
    plt.xlabel("Logarithm of Runetime (log(s))")
    plt.ylabel("Logarithm of error in in all fields, log||e||")
    if mets==None:
        mets=ee.keys()
    for k in mets:
        x,y = dd[k][:,2], np.linalg.norm(ee[k],axis=1)
        if k==bestkey:
            x = dd[k][:-1,2]
            y = y[:-1]
        plt.loglog( x,y,markerkey[k]+'-',color=colorkey[k],label=k)
    plt.legend(loc=1,ncol=2)
    fig.tight_layout()
    
def compute_convergences(dd,ee):
    import copy
    import scipy.stats
    print "    ",
    for k in dd:
        print k, "        ",
    print ""
    for i,f in enumerate(field_labels):
        print f,
        for k in dd:
            h,e = dd[k][:,1], ee[k][:,i]
            if k==bestkey:
                h = dd[k][:-1,1]
                e = ee[k][:-1,i]
            print scipy.stats.linregress([ np.log(x) for x in h ], 
                                         [np.log(y) for y in e])[0],
        print ""
def compute_sum_convergences(dd,ee):
    import scipy.stats
    for k in dd:
        print k, "  ",
        h,e = dd[k][:,1], np.linalg.norm(ee[k],axis=1)
        if k==bestkey:
            h = h[:-1]
            e = e[:-1]
        print scipy.stats.linregress([ np.log(x) for x in h ], 
                                     [np.log(y) for y in e])[0]

def save_my_figs():
    make_error_plots(dd,ee,td,te,(4,2))
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/individualerror.pdf")
    make_sum_error_plots(dd,ee,td,te)
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/allerror.pdf")
    make_sum_time_plots(dd,ee)
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/allruntime.pdf")
    make_sum_time_plots(dd,ee,mets=["LSDIRK2","mLSDIRK2","LSDIRK3","mLSDIRK3"])
    plt.savefig("/Users/afq/Documents/Research/Berkeley/Papers/Yarn DAE/sections/results/newtimeplots/monoallruntime.pdf")
save_my_figs()
