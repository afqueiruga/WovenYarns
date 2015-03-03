from src import *

tests = [ 
    Geometries.PlainWeave(3,1.0,1.0,3,1.0,1.0, 0.0,0.1),
    Geometries.PlainWeaveFibrils(3,1.0,1.0,3,1.0,1.0, 0.0,0.1, [1],1.0)
    ]
    
for p in tests:
    ep = p.endpts()

    defp = {}
    props = [ {} for l in ep ]
    NS = [ 1 for l in ep ]

    warp = Warp(ep,props,defp,NS, Forms.DecoupledProblem)
    
    p.initialize(warp,0)


    print p.contact_pairs(0)
