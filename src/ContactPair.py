from dolfin import *
import ProximityTree

"""
This calculates a contact mapping between two meshes.
Only line meshes in 3D currently supported.
"""

class ContactPair():
    def __init__(self,meA, meB, chi_n_max):
        self.meshA = meA
        self.meshB = meB
        self.chi_n_max = chi_n_max
        
    def make_table(self):
        meA = self.meshA
        meB = self.meshB
        pt = ProximityTree.compiled_module.ProximityTree3D()
        pt.build(meA,1)
        pairs = []
        for i,c in enumerate(cells(meB)):
            ent = pt.compute_proximity_collisions(c.midpoint(),0.3)
            for e in ent:
                pairs.append([e,i])
                
        cs = ProximityTree.plotpairs(pairs,meA,meB)
                
        cellsA = meA.cells()
        vertsA = meA.coordinates()
        cellsB = meB.cells()
        vertsB = meB.coordinates()

        npairs = len(pairs)
        pair_table = np.array(pairs,dtype=np.intc)
        chi_s_table =    np.zeros([npairs*self.chi_n_max,2],dtype=np.double)
        chi_X_table =    np.zeros([npairs*self.chi_n_max,6],dtype=np.double)
        chi_dist_table = np.ones([npairs*self.chi_n_max],dtype=np.double)
        truncate = np.vectorize(lambda x: 0.0 if x<0.0 else 1.0 if x>1.0 else x, otypes=[np.double])

        lda = self.chi_n_max
        for i,(mid,sid) in enumerate(pair_table):
            xA1 = vertsA[cellsA[mid,0]]
            xA2 = vertsA[cellsA[mid,1]]
            xB1 = vertsB[cellsB[sid,0]]
            xB2 = vertsB[cellsB[sid,1]]
            itr = 0
            delB = xB1-xB2
            denom =    delB.dot(delB)
            constant = ( xA2.dot(delB) - xB2.dot(delB) ) / denom
            linear =   ( xA1.dot(delB) - xA2.dot(delB) ) / denom
            chi_s_table[i*lda:(i+1)*lda,0] = np.linspace(0.0,1.0,lda)
            chi_s_table[i*lda:(i+1)*lda,1] = constant + linear * chi_s_table[i*lda:(i+1)*lda,0]
            chi_s_table[i*lda:(i+1)*lda,1] = truncate(chi_s_table[i*lda:(i+1)*lda,1])
            for c in xrange(lda):
                chi_X_table[i*lda+c,:3] = xA1 * chi_s_table[i*lda+c,0] + xA2*(1.0-chi_s_table[i*lda+c,0])
                chi_X_table[i*lda+c,3:] = xB1 * chi_s_table[i*lda+c,1] + xB2*(1.0-chi_s_table[i*lda+c,1])    
                chi_dist_table[i*lda+c] = np.linalg.norm(chi_X_table[i*lda+c,:3]-chi_X_table[i*lda+c,3:])


        self.pair_table = pair_table
        self.chi_s_table = chi_s_table
        self.chi_X_table = chi_X_table
        self.chi_dist_table = chi_dist_table

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

