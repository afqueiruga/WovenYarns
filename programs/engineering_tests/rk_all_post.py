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
field_labels = ['f0_Vol',
 'f0_vq',
 'f0_vh1',
 'f0_vh2',
 'f0_q',
 'f0_h1',
 'f0_h2',
 'f0_T']

def compare_nodal(best,targ):
    return [ np.linalg.norm(best[k]-targ[k]) for k in field_labels ]

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
trunk("LSDIRK2",23)
trunk("LSDIRK3",21)
# trunk("FWEuler",5)
# trunk("DIRK2",6)
# trunk("DIRK3",6)
# trunk("mBWEuler",6)
# trunk("mLSDIRK2",4)
# trunk("mLSDIRK3",5)


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
def make_error_plots(dd,ee,td,te):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 11}
    matplotlib.rc('font', **font)
    for i,f in enumerate(field_labels):
        plt.figure()
        plt.xlabel("Logarithm of time step size log(h)")
        plt.ylabel("Logarithm of error in "+f[3:])
        for k in ee:
            plt.loglog( dd[k][:,1], ee[k][:,i], '-',color=colorkey[k])
            plt.loglog( td[k][:,1], te[k][:,i], markerkey[k]+'-',color=colorkey[k],label=k)
        plt.legend(loc=4,ncol=2)

def make_sum_error_plots(dd,ee,td,te):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 11}
    matplotlib.rc('font', **font)
    plt.figure()
    plt.xlabel("Logarithm of time step size, log(h)")
    plt.ylabel("Logarithm of error in all fields, log||e||")
    for k in ee:
        plt.loglog( dd[k][:,1], np.linalg.norm(ee[k],axis=1), '-',color=colorkey[k])
        plt.loglog( td[k][:,1], np.linalg.norm(te[k],axis=1), markerkey[k]+'-',color=colorkey[k],label=k)
    plt.legend(loc=4,ncol=2)
    plt.show()
    
def make_time_plots(dd,ee, subit=None):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 11}
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
    plt.show()
def make_sum_time_plots(dd,ee):
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 11}
    matplotlib.rc('font', **font)
    plt.figure()
    plt.xlabel("Logarithm of Runetime (log(s))")
    plt.ylabel("Logarithm of error in in all fields, log||e||")
    for k in ee:
        x,y = dd[k][:,2], np.linalg.norm(ee[k],axis=1)
        if k==bestkey:
            x = dd[k][:-1,2]
            y = y[:-1]
        plt.loglog( x,y,markerkey[k]+'-',color=colorkey[k],label=k)
    plt.legend()

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



# make_time_plots(dd,ee, subit=(4,2))
