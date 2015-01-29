"""
Just build some forms and make sure they compile.
"""

from dolfin import *
import numpy as np

from ProximityTree import create_line
from Forms import MultiphysicsProblem

me = create_line(np.array([0.0,0.0,0.0]),np.array([1.0,0.0,0.0]), 10)

props = { 'mu' : 10.0 }
mp = MultiphysicsProblem(me,props,orientation=0)
