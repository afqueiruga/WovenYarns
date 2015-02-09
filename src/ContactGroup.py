from dolfin import *
import numpy as np

import ProximityTree
ProximityTree3D = ProximityTree.compiled_module.ProximityTree3D

"""

This is a remake for ContactPair that does all of 
the meshes at once for effeciency.

"""

class ContactGroup():
    def __init__(self,reference_meshes,current_meshes=None,pairs=None,radii=None, cutoff=0.5):
        self.refs = reference_meshes
        if current_meshes is not None:
            self.curs = current_meshes
        else:
            self.curs = reference_meshes
        if pairs is None:
            pairs = [ (j,i) for j in xrange(len(self.refs)) for i in xrange(j+1,len(self.refs))  ]

        self.desired_pairs = pairs
        self.active_pairs = pairs

        self.trees = None
        self.cutoff = cutoff
        if not radii:
            radii = [ cutoff/2.0 for m in self.refs ]
        self.radii = radii
        
    def CreateTables(self, current_meshes = None):
        if current_meshes is not None:
            self.current_meshes = current_meshes
        if self.trees is None or current_meshes is not None:
            self.trees = [ ProximityTree3D() for m in self.current_meshes ]
            for i,m in enumerate(self.current_meshes):
                self.trees[i].build(m,1)
        
        for a,b in self.desired_pairs:
            ret = None
            
            self.active_pairs.append((a,b, ret))

    def create_table(self, a,b):
        meA = self.curs[a]
        meB = self.curs[b]
        pt = self.trees[a]
        elem_pairs = []
        for i,c in enumerate(cells(meB)):
            ent = pt.compute_proximity_collisions(c.midpoint(),self.cutoff)
            for e in ent:
                elem_pairs.append([e,i])
        
        refA = self.refs[a]
        refB = self.refs[b]
        return None
        return chi_s_table, chi_X_table, chi_dist_table
