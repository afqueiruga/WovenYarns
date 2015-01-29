"""
Just build one of the new Fibril objects
"""

from Fibril import Fibril
from Forms import MultiphysicsProblem

props = { 'mu' : 10.0 }

fib = Fibril([ [0.0,0.0,0.0],[1.0,0.0,0.0]], 10, props,
             MultiphysicsProblem)

fib.WriteFile("unit_tests/unit_fibril_test.pvd")
fib.WriteSurface("unit_tests/unit_fibril_mesh_test.pvd")
