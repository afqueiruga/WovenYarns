from src import *

from IPython import embed

"""
Compute the effective properties of a textile square
"""

sheet = Geometries.PlainWeaveFibrils(4, 4.0,4.0, 4, 4.0,4.0, 
         0.0,0.35*1.5, [3,4,3], 0.3)
endpts = sheet.endpts()
defaults = { 'radius':0.15,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':1.0e0,
             'contact_penalty':5.0,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)

