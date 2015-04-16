from dolfin import *
import numpy as np

from ProximityTree import create_line

class Fibril():
    """ This is a fibril that relies on Problems """
    def __init__(self, pts, Nelem, properties, Prob, order=(1,1)):
        me = create_line(np.array(pts[0]), np.array(pts[1]), Nelem)
        self.mesh = me

        # Initialize mesh function for boundary domains
        self.boundary_domains = [
            CompiledSubDomain("near(x[0],X) && near(x[1],Y) && near(x[2],Z) && on_boundary",
                              X=pts[0][0],Y=pts[0][1],Z=pts[0][2]),
            CompiledSubDomain("near(x[0],X) && near(x[1],Y) && near(x[2],Z) && on_boundary",
                              X=pts[1][0],Y=pts[1][1],Z=pts[1][2])
        ]
        self.boundaries = FacetFunction("size_t", self.mesh)
        self.boundaries.set_all(0)
        for i,b in enumerate(self.boundary_domains):
            b.mark(self.boundaries,i+1)
        
        
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
        
        self.problem = Prob(self.mesh,properties, boundaries=self.boundaries, orientation = orientation, order=order)

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
            for jt,theta in enumerate(np.linspace(0.0,2.0*np.pi,NT)): # This is a bug!
                edit.add_vertex(NT*ix+jt,
                                np.array(cent+np.cos(theta)*g1c+np.sin(theta)*g2c,
                                         dtype=np.float_))

        edit.init_cells( qN.shape[1]*2*NT)
        for ix in xrange(1,qN.shape[1]):
            for jt in xrange(NT-1):
                edit.add_cell(2*NT*(ix-1)+2*jt,   NT*ix+jt+1   , NT*(ix-1)+jt, NT*(ix-1)+jt+1)
                edit.add_cell(2*NT*(ix-1)+2*jt+1, NT*ix+jt+1   , NT*ix    +jt, NT*(ix-1)+jt)
            edit.add_cell(2*NT*(ix-1)+2*NT-2,       NT*ix+0   ,    NT*(ix-1)+NT-1, NT*(ix-1)+0)
            edit.add_cell(2*NT*(ix-1)+2*NT-1,     NT*ix+0   ,    NT*ix    +NT-1, NT*(ix-1)+NT-1)

        edit.close()
        
        vf = File(fname,"ascii")
        vf << hullmesh

    def WriteSolid(self,fname,i=0,NT=8):
        fields = self.problem.split_for_io()

        qN = fields['q'].compute_vertex_values()
        qN = qN.reshape([3,qN.shape[0]/3])
        g1N = fields['g1'].compute_vertex_values()
        g1N = g1N.reshape([3,g1N.shape[0]/3])
        g2N = fields['g2'].compute_vertex_values()
        g2N = g2N.reshape([3,g2N.shape[0]/3])
        coords = fields['q'].function_space().mesh().coordinates()

        solidmesh = Mesh()
        edit = MeshEditor()
        edit.open(solidmesh,3,3)
        lda = NT+1
        edit.init_vertices(lda*qN.shape[1])
        for ix in xrange(qN.shape[1]):
            cent = qN[:,ix]+coords[ix]
            g1c = g1N[:,ix]
            g2c = g2N[:,ix]
            for jt,theta in enumerate(np.linspace(0.0,2.0*np.pi,NT)): # This is a bug!
                edit.add_vertex(lda*ix+jt,
                                np.array(cent+np.cos(theta)*g1c+np.sin(theta)*g2c,
                                         dtype=np.float_))
            edit.add_vertex(lda*ix+NT,
                            np.array(cent,dtype=np.float_))
        edit.init_cells( qN.shape[1]*3*NT)
        for ix in xrange(1,qN.shape[1]):
            for jt in xrange(NT-1):
                a = lda*ix+jt
                b = lda*ix+jt+1
                c = lda*ix+NT
                d = lda*(ix-1)+jt+0
                e = lda*(ix-1)+jt+1
                f = lda*(ix-1)+NT
                edit.add_cell(3*NT*(ix-1)+3*jt,   a, b, c, f)
                edit.add_cell(3*NT*(ix-1)+3*jt+1, e, d, f, b)
                edit.add_cell(3*NT*(ix-1)+3*jt+2, d, a, f, b)
            # edit.add_cell(3*NT*(ix-1)+3*NT-3,      lda*ix+NT-1   , lda*ix+NT-1, lda*ix+NT, lda*(ix-1)+NT-1)
            # edit.add_cell(3*NT*(ix-1)+3*NT-2,       NT*ix+0   ,    NT*(ix-1)+NT-1, NT*(ix-1)+0)
            # edit.add_cell(3*NT*(ix-1)+3*NT-1,     NT*ix+0   ,    NT*ix    +NT-1, NT*(ix-1)+NT-1)

        edit.close()
        from IPython import embed


        MS = FunctionSpace(solidmesh,"CG",1)
        MV = VectorFunctionSpace(solidmesh,"CG",1)
        v2dS = cpp.fem.vertex_to_dof_map(MS)
        v2dV = cpp.fem.vertex_to_dof_map(MV)
        newlist = []
        for n,f in fields.iteritems():
            F = Function(MS if f.rank()==0 else MV)
            v2d = v2dS if f.rank()==0 else v2dV
            fN = f.compute_vertex_values()
            if f.rank()==1:
                fN = fN.reshape([3,fN.shape[0]/3])
            for ix in xrange(qN.shape[1]):
                for j in xrange(lda):
                    if f.rank()==0:
                        F.vector()[v2d[lda*ix+j]] = fN[ix]
                    else:
                        F.vector()[v2d[3*(lda*ix+j):3*(lda*ix+j+1)]] = np.array(fN[:,ix])
            F.rename(n,"field")
            newlist.append(F)
        
        from multiwriter.multiwriter import VTKAppender
        vfile = VTKAppender(fname,"ascii")
        vfile.write(i,newlist, [])
