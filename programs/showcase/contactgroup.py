from src import *

from IPython import embed

"""
Just output a contact group to make a figure
"""

sheet = Geometries.PlainWeaveFibrils(2, 2.0,2.0, 2, 2.0,2.0, 
         0.0,0.15*1.0, [2], 0.35)
endpts = sheet.endpts()
defaults = { 'radius':0.15,
             'em_B':Constant((0.0,0.0,0.0)),
             'dissipation':1.0e0,
             'contact_penalty':5.0,
             'mech_bc_trac_0':Constant((0.0,0.0,0.0)),
             'contact_em':0.1}
props = [ {} for i in endpts ]
Nelems = [ 20 for i in endpts ]

warp = Warp(endpts,props,defaults, Nelems, DecoupledProblem)


sheet.initialize(warp,0)
warp.pull_fibril_fields()


cpairs = sheet.contact_pairs()
warp.create_contacts(cpairs,cutoff=0.5)

warp.CG.OutputFile("post/showcase/gammaC.pvd")
warp.output_solids("post/showcase/mesh_{0}_0.pvd",1)
