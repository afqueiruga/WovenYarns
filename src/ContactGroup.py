from dolfin import *
import numpy as np

import ProximityTree
ProximityTree3D = ProximityTree.compiled_module.ProximityTree3D

"""

This is a remake for ContactPair that does all of 
the meshes at once for efficiency.

"""
truncate = np.vectorize(lambda x: 0.0 if x<0.0 else 1.0 if x>1.0 else x, otypes=[np.double])

class ContactGroup():
    def __init__(self,reference_meshes,current_meshes=None,pairs=None,radii=None, cutoff=0.15,candidate_buffer=0.1):
        self.reference_meshes = reference_meshes
        if current_meshes is not None:
            self.current_meshes = current_meshes
        else:
            self.current_meshes = reference_meshes
        if pairs is None:
            pairs = [ (j,i) for j in xrange(len(self.reference_meshes)) for i in xrange(j+1,len(self.reference_meshes))  ]

        self.desired_pairs = pairs
        self.active_pairs = pairs

        self.trees = None
        self.cutoff = cutoff
        if not radii:
            radii = [ cutoff/2.0 for m in self.reference_meshes ]
        self.radii = radii
        
        self.chi_n_max = 10
        self.candidate_buffer = candidate_buffer
        
    def CreateTables(self, current_meshes = None):
        if current_meshes is not None:
            self.current_meshes = current_meshes
        if self.trees is None or current_meshes is not None:
            self.trees = [ ProximityTree3D() for m in self.current_meshes ]
            for i,m in enumerate(self.current_meshes):
                self.trees[i].build(m,1)

        self.active_pairs = []
        for a,b in self.desired_pairs:
            ret = self.create_mesh_pair_table(a,b)
            if ret:
                self.active_pairs.append((a,b, ret))

        
    def create_mesh_pair_table(self, a,b):
        meA = self.current_meshes[a]
        meB = self.current_meshes[b]
        pt = self.trees[a]
        elem_pairs = []

        minsearch = (self.radii[a]+self.radii[b]+self.candidate_buffer)
        cutoff = np.max((minsearch,self.cutoff))
        for i,c in enumerate(cells(meB)):
            ent = pt.compute_proximity_collisions(c.midpoint(), cutoff)
            for e in ent:
                elem_pairs.append([e,i])
        if len(elem_pairs)==0:
            return None
        
        refA = self.reference_meshes[a]
        refB = self.reference_meshes[b]

        cellsA = meA.cells()
        vertsA = meA.coordinates()
        cellsB = meB.cells()
        vertsB = meB.coordinates()
        
        ref_cellsA = refA.cells()
        ref_vertsA = refA.coordinates()
        ref_cellsB = refB.cells()
        ref_vertsB = refB.coordinates()

        stables = []
        xtables = []
        Xtables = []
        disttables = []
        used_pairs = []
        for mid,sid in elem_pairs:
            stable,xtable,Xtable,disttable = self.create_elem_pair_table(vertsA[cellsA[mid,0]],vertsA[cellsA[mid,1]],
                                        vertsB[cellsB[sid,0]],vertsB[cellsB[sid,1]],
                                        ref_vertsA[ref_cellsA[mid,0]],ref_vertsA[ref_cellsA[mid,1]],
                                        ref_vertsB[ref_cellsB[sid,0]],ref_vertsB[ref_cellsB[sid,1]]
                )
            if np.min(disttable) < minsearch :
                stables.append(stable)
                xtables.append(xtable)
                Xtables.append(Xtable)
                disttables.append(disttable)
                used_pairs.append((mid,sid))
        if len(stables)==0:
            return None
        pair_table = np.array(used_pairs,dtype=np.intc)

        chi_s_table = np.vstack(stables)
        chi_x_table = np.vstack(xtables)
        chi_X_table = np.vstack(Xtables)
        chi_dist_table = np.hstack(disttables)
        
        return pair_table, chi_s_table, chi_x_table, chi_X_table, chi_dist_table

    def create_elem_pair_table(self, xA1,xA2,xB1,xB2, XA1,XA2,XB1,XB2):
        self.chi_n_max
        stable = np.zeros([self.chi_n_max,2],dtype=np.double)
        Xtable = np.zeros([self.chi_n_max,6],dtype=np.double)
        xtable = np.zeros([self.chi_n_max,6],dtype=np.double)
        disttable = np.ones([self.chi_n_max],dtype=np.double)

        delB = xB1-xB2
        denom =    delB.dot(delB)
        constant = ( xA2.dot(delB) - xB2.dot(delB) ) / denom
        linear =   ( xA1.dot(delB) - xA2.dot(delB) ) / denom
            
        stable[:,0] = np.linspace(0.0,1.0,self.chi_n_max)
        stable[:,1] = truncate(constant + linear * stable[:,0])

        for c in xrange(self.chi_n_max):
            Xtable[c,:3] = XA1 * stable[c,0] + XA2*(1.0-stable[c,0])
            Xtable[c,3:] = XB1 * stable[c,1] + XB2*(1.0-stable[c,1])
            xtable[c,:3] = xA1 * stable[c,0] + xA2*(1.0-stable[c,0])
            xtable[c,3:] = xB1 * stable[c,1] + xB2*(1.0-stable[c,1])
            disttable[c] = np.linalg.norm(xtable[c,:3]-xtable[c,3:])
            
        return stable,xtable,Xtable,disttable

    def chi_to_mesh(self, limits = None):
        mesh = Mesh()
        edit = MeshEditor()
        edit.open(mesh,1,3)
        nelem = 0
        for a,b,(etab,stab,xtab,Xtab,dtab) in self.active_pairs:
            if (limits is None) or ( (a in limits) or (b in limits) ):
                nelem += len(xtab)
        edit.init_vertices(2*nelem)
        edit.init_cells(nelem)
        cnt = 0
        for a,b,(etab,stab,xtab,Xtab,dtab) in self.active_pairs:
            if (limits is None) or ( (a in limits) or (b in limits) ):
                for i in xrange(len(xtab)):
                    edit.add_vertex(2*cnt, xtab[i,:3])
                    edit.add_vertex(2*cnt+1, xtab[i,3:])
                    edit.add_cell(cnt,2*cnt,2*cnt+1)
                    cnt+=1
        edit.close()
        return mesh
    
    def OutputFile(self,fname, limits = None):
        GammaC = self.chi_to_mesh(limits)
        mfC = CellFunction("double",GammaC)
        cnt = 0
        for a,b,(etab,stab,xtab,Xtab,dtab) in self.active_pairs:
            if (limits is None) or ( (a in limits) or (b in limits) ):
                for i in xrange(len(xtab)):
                    mfC.set_value(cnt,dtab[i])
                    cnt+=1
        ff = File(fname)
        ff << mfC
