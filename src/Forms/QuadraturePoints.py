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

if __name__=="__main__":
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

    # These functions and values are generated in the
    # mathematica notebook QuadratureUnitTests.nb
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
        6.2831853071795864769,
        0,
        1.0471975511965977462,
        0,
        0.47123889803846898577,
        2.0943951023931954923,
        0.15707963267948966192,
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
    plt.show()
    
    
