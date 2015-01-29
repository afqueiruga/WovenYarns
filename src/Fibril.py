from dolfin import *
import numpy as np

from ProximityTree import create_line

class Fibril():
    """ This is a fibril that relies on Problems """
    def __init__(self, pts, Nelem, properties, Prob):
        me = create_line(np.array(pts[0]), np.array(pts[1]), Nelem)
        self.mesh = me
        
        E = np.array(pts[1])- np.array(pts[0])
        if E[1]==0.0 and E[2]==0.0:
            orientation=0
        elif E[0]==0.0 and E[2]==0.0:
            orientation=1
        elif E[0]==0.0 and E[1]==0.0:
            orientation=2
        else:
            print "Error: Fibrils must be axis aligned! But I'm not going to stop."
            orientation=0
        
        self.problem = Prob(self.mesh,properties, orientation = orientation)

    def WriteFile(self,fname,i=0):
        self.problem.WriteFile(fname,i)
    def WriteSurface(self,fname,i=0,NT=16):
        fields = self.problem.split_for_io()

        qN = fields['q'].compute_vertex_values()
        qN = qN.reshape([3,qN.shape[0]/3])
        g1N = fields['g1'].compute_vertex_values()
        g1N = g1N.reshape([3,g1N.shape[0]/3])
        g2N = fields['g2'].compute_vertex_values()
        g2N = g2N.reshape([3,g2N.shape[0]/3])
        coords = fields['q'].function_space().mesh().coordinates()

        hullmesh = Mesh()
        edit = MeshEditor()
        edit.open(hullmesh,2,3)
        edit.init_vertices(NT*qN.shape[1])
        for ix in xrange(qN.shape[1]):
            cent = qN[:,ix]+coords[ix]
            g1c = g1N[:,ix]
            g2c = g2N[:,ix]
            for jt,theta in enumerate(np.linspace(0.0,2.0*np.pi,NT-1)):
                edit.add_vertex(NT*ix+jt,
                                np.array(cent+np.cos(theta)*g1c+np.sin(theta)*g2c,
                                         dtype=np.float_))

        edit.init_cells( qN.shape[1]*2*NT)
        for ix in xrange(1,qN.shape[1]):
            for jt in xrange(NT-1):
                edit.add_cell(2*NT*(ix-1)+2*jt,   NT*ix+jt+1   , NT*(ix-1)+jt, NT*(ix-1)+jt+1)
                edit.add_cell(2*NT*(ix-1)+2*jt+1, NT*ix+jt+1   , NT*ix    +jt, NT*(ix-1)+jt)
            edit.add_cell(2*NT*(ix-1)+2*NT,       NT*ix+0   ,    NT*(ix-1)+NT-1, NT*(ix-1)+0)
            edit.add_cell(2*NT*(ix-1)+2*NT+1,     NT*ix+0   ,    NT*ix    +NT-1, NT*(ix-1)+NT-1)

        edit.close()
        
        vf = File(fname,"ascii")
        vf << hullmesh
