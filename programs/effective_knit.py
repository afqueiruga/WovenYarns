from src import *

from IPython import embed

"""
Compute the effective properties of a knitted square
"""

sheet = Geometries.StockinetteFibrils(2.0,1.0, 5,2.0, [1,3], [0.2,0.2], 0.1)
endpts = sheet.endpts()
defaults = { 'radius':0.1,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':1.0e0,
             'contact_penalty':5.0,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0))}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)


outputcnt = 0
def output():
    global outputcnt
    warp.output_states("post/KnitRVE/yarn_{0}_"+str(outputcnt)+".pvd",1)
    warp.output_solids("post/KnitRVE/mesh_{0}_"+str(outputcnt)+".pvd",1)
    outputcnt+=1

output()

sheet.initialize(warp,0)

output()
