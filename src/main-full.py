#!/usr/bin/python

from dolfin import *

from matplotlib import pylab as plt
import numpy as np

from IPython import embed

"""

Main entry point of the simulation.

"""

from Warp import Warp

endpts = [ [ [-10.0, 0.0,-1.0],  [10.0, 0.0, -1.0] ] ]

warp = Warp(endpts)

Delw = MultiMeshFunction(warp.mmfs)

warp.assemble_system()

embed()
maxiter = 10
tol = 1.0e-9
