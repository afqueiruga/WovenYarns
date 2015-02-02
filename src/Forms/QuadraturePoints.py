"""

This file contains rules for cross section points.

The 1D and thus outer product rules are commonly known Gauss-Legendre points.
The circle points are the taken from those listed by Pavel Holoborodko on his page

http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/cubature-formulas-for-the-unit-disk/

"""

import numpy as np

#
# These are 1D standard Gauss-Legendre Points.
#
GPS1D = {
    2:  [ [ np.sqrt(1.0/3.0), 1.0],
              [-np.sqrt(1.0/3.0), 1.0] ],
    3:  [ [ 0.0,       8.0/9.0 ],
               [ np.sqrt(3.0/5.0), 5.0/9.0 ],
               [-np.sqrt(3.0/5.0), 5.0/9.0 ] ]
    }

def RectOuterProd(ordx,ordy=None):
    " Create an outer product rule for rectangles. "
    if ordy==None:
        ordy=ordx
    GPS2D = []
    for z1,w1 in GPS1D[ordx]:
        for z2,w2 in GPS1D[ordy]:
            GPS2D.append([z1,z2,w1*w2])
    return GPS2D

#
# Generate the rules of radially symmetric circule rules in polar coordinates 
#
CircPolar2D = {4:[],8:[],16:[]}
for i in xrange(4):
    CircPolar2D[4].append([np.sqrt(2.0)/2.0,np.pi*(i/2.0), np.pi/4.0])
for i in xrange(4):
    CircPolar2D[8].append(
        [np.sqrt(1.0+np.sqrt(3.0)/3.0), np.pi*((2*i)/4.0), np.pi*(2.0-np.sqrt(3.0))/16.0])
    CircPolar2D[8].append(
        [np.sqrt(1.0-np.sqrt(3.0)/3.0), np.pi*((2*i+1)/4.0), np.pi*(2.0+np.sqrt(3.0))/16.0])
for i in xrange(8):
    CircPolar2D[16].append(
        [np.sqrt(0.5+np.sqrt(3.0)/6.0), np.pi*((2*i)/8.0), np.pi/16.0])
    CircPolar2D[16].append(
        [np.sqrt(0.5-np.sqrt(3.0)/6.0), np.pi*((2*i+1)/8.0), np.pi/16.0])

# Now put them into cartisian coordinates
CircCart2D = {}
for k,rule in CircPolar2D.iteritems():
    CircCart2D[k] = [ [r*np.cos(t),r*np.sin(t),w] for r,t,w in rule ]
# TODO: I don't like the numerical roundoff... should turn this into
# a Mathematica script that does exact arithmetic until the last stage

def apply_2D_rule(f,rule):
    " Nonoptimally compute a given 2D rule on a function "
    res = 0.0
    for z1,z2,w in rule:
        res+=f(z1,z2)*w
    return res

if __name__=="__main__":
    def plot_all_rules():
        " Look at all of the rules... Kinda annoying to do every time "
        from matplotlib import pylab as plt
        def plot_points(pts):
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.plot([ z1 for z1,z2,w in pts], [z2 for z1,z2,w in pts],'x')
        plot_points(RectOuterProd(2,3))
        plt.figure()
        plot_points(RectOuterProd(2))
        plt.figure()
        plot_points(CircCart2D[4])
        plt.figure()
        plot_points(CircCart2D[8])
        plt.figure()
        plot_points(CircCart2D[16])
        plt.show()
    
    # These functions and values are generated in the
    # mathematica notebook QuadratureUnitTests.nb
    # TODO: file input, copy and paste is messy
    trialfunctions = [
        lambda x, y: 1,
        lambda x, y: x,
        lambda x, y: (x**2),
        lambda x, y: (x**3),
        lambda x, y: (x**4),
        lambda x, y: (x**2) + (y**2),
        lambda x, y: (x**2)*(y**2),
        lambda x, y: (x**3)*(y**3)]
    circvalues = [
        3.1415926535897932385,
        0,
        0.78539816339744830962,
        0,
        0.39269908169872415481,
        1.5707963267948966192,
        0.13089969389957471827,
        0
    ]
    squarevalues = [
        4.0000000000000000000,
        0,
        1.3333333333333333333,
        0,
        0.80000000000000000000,
        2.6666666666666666667,
        0.44444444444444444444,
        0
    ]
    
    
    def test_rule(rule,trials,exacts):
        " Test a given rule on the trial functions "
        for f,ex in zip(trials,exacts):
            print np.abs(ex-apply_2D_rule(f,rule)), " ",
        print
    for numpts,rule in CircCart2D.iteritems():
        print "Testing Circle rule ",numpts
        test_rule(rule,trialfunctions,circvalues)
    for ordx,ordy in [(2,2),(2,3),(3,2),(3,3)]:
        print "Testing Outer product rule ",ordx," x ",ordy
        rule = RectOuterProd(ordx,ordy)
        test_rule(rule,trialfunctions,squarevalues)
