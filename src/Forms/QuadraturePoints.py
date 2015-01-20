"""

This file contains rules for cross section points.

"""
import numpy as np

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



if __name__=="__main__":
    from matplotlib import pylab as plt
    def plot_points(pts):
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.plot([ z1 for z1,z2,w in pts], [z2 for z1,z2,w in pts],'x')
    plot_points(RectOuterProd(2,3))
    plt.show()
    print RectOuterProd(2,2)
    print RectOuterProd(2,3)
    print RectOuterProd(3)
