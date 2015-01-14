from dolfin import *
import numpy as np

import ProximityTree

from IPython import embed

"""
This calculates a contact mapping between two meshes.
Only line meshes in 3D currently supported.
"""

class ContactPair():
    def __init__(self,meA, refA, meB, refB, chi_n_max):
        self.meshA = meA
        self.meshB = meB
        self.ref_meshA = refA
        self.ref_meshB = refB
        self.chi_n_max = chi_n_max
        
    def make_table(self):
        meA = self.meshA
        meB = self.meshB
        refA = self.ref_meshA
        refB = self.ref_meshB
        pt = ProximityTree.compiled_module.ProximityTree3D()
        pt.build(meA,1)
        pairs = []
        for i,c in enumerate(cells(meB)):
            # TODO: this needs to be a paramter
            ent = pt.compute_proximity_collisions(c.midpoint(),1.6)
            for e in ent:
                pairs.append([e,i])
                
        cs = ProximityTree.plotpairs(pairs,meA,meB)
                
        cellsA = meA.cells()
        vertsA = meA.coordinates()
        cellsB = meB.cells()
        vertsB = meB.coordinates()
        
        ref_cellsA = refA.cells()
        ref_vertsA = refA.coordinates()
        ref_cellsB = refB.cells()
        ref_vertsB = refB.coordinates()
        
        npairs = len(pairs)
        pair_table = np.array(pairs,dtype=np.intc)
        chi_s_table =    np.zeros([npairs*self.chi_n_max,2],dtype=np.double)
        chi_X_table =    np.zeros([npairs*self.chi_n_max,6],dtype=np.double)
        chi_dist_table = np.ones([npairs*self.chi_n_max],dtype=np.double)
        # truncate = np.vectorize(lambda x: 0.1 if x<0.1 else 0.9 if x>0.9 else x, otypes=[np.double])
        truncate = np.vectorize(lambda x: 0.0 if x<0.0 else 1.0 if x>1.0 else x, otypes=[np.double])

        lda = self.chi_n_max
        for i,(mid,sid) in enumerate(pair_table):
            xA1 = vertsA[cellsA[mid,0]]
            xA2 = vertsA[cellsA[mid,1]]
            xB1 = vertsB[cellsB[sid,0]]
            xB2 = vertsB[cellsB[sid,1]]
            
            XA1 = ref_vertsA[ref_cellsA[mid,0]]
            XA2 = ref_vertsA[ref_cellsA[mid,1]]
            XB1 = ref_vertsB[ref_cellsB[sid,0]]
            XB2 = ref_vertsB[ref_cellsB[sid,1]]
            itr = 0
            delB = xB1-xB2
            denom =    delB.dot(delB)
            constant = ( xA2.dot(delB) - xB2.dot(delB) ) / denom
            linear =   ( xA1.dot(delB) - xA2.dot(delB) ) / denom
            chi_s_table[i*lda:(i+1)*lda,0] = np.linspace(0.1,0.9,lda)
            chi_s_table[i*lda:(i+1)*lda,1] = constant + linear * chi_s_table[i*lda:(i+1)*lda,0]
            chi_s_table[i*lda:(i+1)*lda,1] = truncate(chi_s_table[i*lda:(i+1)*lda,1])
            for c in xrange(lda):
                chi_X_table[i*lda+c,:3] = XA1 * chi_s_table[i*lda+c,0] + XA2*(1.0-chi_s_table[i*lda+c,0])
                chi_X_table[i*lda+c,3:] = XB1 * chi_s_table[i*lda+c,1] + XB2*(1.0-chi_s_table[i*lda+c,1])    
                chi_dist_table[i*lda+c] = np.linalg.norm(chi_X_table[i*lda+c,:3]-chi_X_table[i*lda+c,3:])

        # embed()
        self.pair_table = pair_table
        self.chi_s_table = chi_s_table
        self.chi_X_table = chi_X_table
        self.chi_dist_table = chi_dist_table
        self.pair_flattened = self.pair_table.flatten()
        
    def chi_to_mesh(self):
        mesh = Mesh()
        edit = MeshEditor()
        edit.open(mesh,1,3)
        nelem = len(self.chi_X_table)
        edit.init_vertices(2*nelem)
        edit.init_cells(nelem)
        for i in xrange(nelem):
            edit.add_vertex(2*i,  self.chi_X_table[i,:3])
            edit.add_vertex(2*i+1,self.chi_X_table[i,3:])
            edit.add_cell(i, 2*i,2*i+1)
        edit.close()
        return mesh

    def output_file(self,fname_pairs,fname_gamma):
        if fname_gamma:
            GammaC = self.chi_to_mesh()
            mfC = CellFunction("double",GammaC)
            for i in xrange(len(self.chi_X_table)):
                mfC.set_value(i,self.chi_s_table[i,1])
            ff = File(fname_gamma)
            ff << mfC

        if fname_pairs:
            cs = ProximityTree.plotpairs(self.pair_table,self.meshA,self.meshB)
            ff = File(fname_pairs)
            ff << cs
