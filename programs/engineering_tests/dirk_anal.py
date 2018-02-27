import numpy as np
import matplotlib
from matplotlib import pylab as plt
import copy

files = ["dirkdata_3", "dirkdata_5"]
data = np.vstack([ np.loadtxt(f) for f in files ])

split_data = [[],[],[]]
for l in data:
    split_data[int(l[0])-1].append(l[1:])
for i in xrange(len(split_data)):
    split_data[i] = np.array(split_data[i])
    split_data[i] = split_data[i][split_data[i][:,0].argsort()]
    # split_data[i].sort(axis=0)

split_data[1] = split_data[1][0:-6,:]
# split_data[2] = split_data[2][:-1,:]
sdconv = copy.deepcopy(split_data)
sdconv[0] = sdconv[0][4 :  ,:]
sdconv[1] = sdconv[1][12:,:]
sdconv[2] = sdconv[2][5 :  ,:]

Tmax = 5.0
labels = [ 'y(0)','z(0.5)','T(0.5)','V(0.5)' ]
def make_error_plots(sd, sdconv, labels):
    font = {'family' : 'normal',
            'size'   : 16}
    matplotlib.rc('font', **font)
    hs = [ Tmax / dat[:,0] for dat in sd]
    hsconv = [ Tmax / dat[:,0] for dat in sdconv]
    colors = ["b","g","r"]
    for i in xrange(4):
        label = labels[i]
        plt.figure()
        plt.xlabel("Logarithm of time step size log(h)")
        plt.ylabel("Logarithm of error in "+label)
        best = sd[-1][-1,i+1]
        for o,(dat,datconv,c) in enumerate(zip(sd,sdconv,colors)):
            plt.loglog( hs[o], [np.abs(y-best) for y in dat[:,i+1]],'-',label='s='+str(o+1),color=c)
            plt.loglog( hsconv[o], [np.abs(y-best) for y in datconv[:,i+1]],'+',color=c)
        plt.legend()
    plt.show()


def compute_convergence(sd):
    import copy
    import scipy.stats
    sd = copy.deepcopy(sd)
    hs = [ Tmax / dat[:,0] for dat in sd]
    hs[-1] = hs[-1][:-1]    
    best = sd[-1][-1,:]
    sd[-1] = sd[-1][:-1,:]
    for i in xrange(4):
        b = best[i+1]
        for h,s in zip(hs,sd):
            print scipy.stats.linregress([ np.log(x) for x in h ], 
                                         [np.log(np.abs(y-b)) for y in s[:,i+1]])[0],
        print ""
plt.close('all')
compute_convergence(sdconv)
make_error_plots(split_data, sdconv, labels)
