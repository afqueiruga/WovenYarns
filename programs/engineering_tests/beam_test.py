"""

Take a single fiber and check it against standard Euler-Bernoulli beam theory. Check convergences as we refine, too.

"""
from dolfin import *

from src import Fibril
from src.Forms import MultiphysicsProblem

props = { 'mu' : 10.0 }
fib = Fibril.Fibril([ [0.0,0.0,0.0],[1.0,0.0,0.0]], 10, props,
                MultiphysicsProblem)

R = assemble(fib.problem.forms['F'])
K = assemble(fib.problem.forms['AX'])
K.ident_zeros()

DelW = Function(fib.problem.spaces['W'])

solve(K,DelW.vector(),R)


from IPython import embed
embed()
