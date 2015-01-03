#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [-1.0, 0.0,-0.1],  [1.0, 0.0, -0.1] ],
           [ [ 0.0, -1.0,0.0],  [0.0, 1.0, 0.0] ] ]

warp = Warp(endpts)
DelT = MultiMeshFunction(warp.Tmmfs)


