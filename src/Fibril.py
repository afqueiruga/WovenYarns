from dolfin import *
import numpy as np

from Forms import BeamForm, ThermalForm, MultiphysicsForm, IterativeForm

"""

This is a base class for the director-based Fibril

"""
from IPython import embed
class Fibril():
    def __init__(self,mesh,orientation=0, monolithic=True, radius=0.075, wx=None,wv=None):
        """
        Create a new Fibril on a given mesh. It better be a line mesh.
        Dirrichlet BCs by default.
        """
        self.mesh=mesh
        self.current_mesh = Mesh(mesh)
        self.V = VectorFunctionSpace(mesh,"CG",1)
        self.S = FunctionSpace(mesh,"CG",1)
        if monolithic:
            self.W = MixedFunctionSpace([self.V,self.V,self.V,self.S,self.S])
        else:
            self.W = MixedFunctionSpace([self.V,self.V,self.V])
        if wx and wv:
            build_form(wx,wv)
        self.monolithic = monolithic
        self.orientation=orientation
        self.radius=radius
        
    def build_multi_form(self,wx=None,wv=None):
        if not wx and not wv:
            self.wx = Function(self.W)
            self.wv = Function(self.W)
        else:
            self.wx = Function(wx)
            self.wv = Function(wv)

        self.X0 = Function(self.V)
        self.X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
        
        self.Fform,self.Mform,self.AXform,self.AVform, self.Ez,self.E1,self.E2 = \
          MultiphysicsForm(self.W,self.V,self.S,self.wx,self.wv,self.X0, self.orientation,self.radius)

        self.Height = 1.0
        self.Width = 1.0
    def build_iterative_form(self):
        self.wx = Function(self.W)
        self.wv = Function(self.W)
        self.T = Function(self.S)
        self.Vol = Function(self.S)
        
        self.X0 = Function(self.V)
        self.X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
        
        self.Fform,self.Mform,self.AXform,self.AVform, \
        self.FTform,self.MTform,self.ATform, \
        self.FVform,self.AVform, \
        self.Ez,self.E1,self.E2 = \
          IterativeForm(self.W,self.V,self.S,
                           self.wx,self.wv,self.T,self.Vol,
                           self.X0, self.orientation,self.radius)
        
    def build_thermal_form(self):
        self.X0 = Function(self.V)
        self.X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
        
        self.T = Function(self.S)
        self.FTform, self.MTform, self.ATform = ThermalForm(self.S,self.T,self.X0)
        
    def build_form(self,wx=None,wv=None):
        if not wx and not wv:
            self.wx = Function(self.W)
            self.wv = Function(self.W)
        else:
            self.wx = Function(wx)
            self.wv = Function(wv)

        self.X0 = Function(self.V)
        self.X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
        
        self.Fform,self.Mform,self.AXform,self.AVform = \
          BeamForm(self.W,self.V,self.wx,self.wv,self.X0)

        left =  CompiledSubDomain(
            "near(x[0], s0) && near(x[1], s1) && near(x[2], s2) && on_boundary",
            s0 = 0.0, s1 = 0.0, s2 = 0.0)
        cl = Expression(("0.0","0.0","0.0",  "0.0"," 0.0","0.0",  "0.0","0.0","0.0"))
        bc1 = DirichletBC(self.W, cl, left)
        right =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 10.0)
        cr = Expression(("0.0","0.0","0.0",  "0.0"," 0.0","0.0",  "0.0","0.0","0.0"))
        bc2 = DirichletBC(self.W, cr, right)
        self.bcs=[bc1]

        self.Height = 1.0
        self.Width = 0.5

        self.T = Function(self.S)
        self.FTform, self.MTform, self.ATform = ThermalForm(self.S,self.T,self.X0)
        
    def start_file(self,fname):
        "Open an empty file with fname."
        pass
    
    def write_file(self,fname,i):
        """
        Pop out one file on the currently active file.
        """
        from multiwriter.multiwriter import VTKAppender
        vfile = VTKAppender(fname,"ascii")
        if self.monolithic:
            q,h1,h2, Tnull, Vnull    = self.wx.split()
            vq,vh1,vh2,T,Vol = self.wv.split()
        else:
            q,h1,h2    = self.wx.split()
            vq,vh1,vh2 = self.wv.split()
            T = self.T
            Vol = self.Vol
        q.rename("q","field")
        h1.rename("h1","field")
        h2.rename("h2","field")

                
        vq.rename("vq","field")
        vh1.rename("vh1","field")
        vh2.rename("vh2","field")
        T.rename("T","field")
        Vol.rename("Vol","field")
        # q1 = Function(V)
        # q1.interpolate(Height/2.0*Constant((0.0,1.0,0.0))+h1)
        # q2 = Function(V)
        # q2.interpolate(Width/2.0*Constant((0.0,0.0,1.0))+h2)
        g1 = project(self.radius*Constant((0.0,1.0,0.0))+h1,self.V)
        g1.rename("g1","field")
        g2 = project(self.radius*Constant((0.0,0.0,1.0))+h2,self.V)
        g2.rename("g2","field")

        self.X0.rename("X0","field")
        vfile.write(i,[q,h1,h2,g1,g2, vq,vh1,vh2, T, Vol,self.X0],[])
        # vfile = VTKAppender(fname,"ascii")
    def write_surface(self,fname,i,NT=16,Lnum=200,Lmax=100.0):
        """
        Pop out one file on the currently active file.
        """
        from multiwriter.multiwriter import VTKAppender
        vfile = VTKAppender(fname,"ascii")
        if self.monolithic:
            q,h1,h2, Tnull, Vnull    = self.wx.split()
            vq,vh1,vh2,T,Vol = self.wv.split()
        else:
            q,h1,h2    = self.wx.split()
            vq,vh1,vh2 = self.wv.split()
            T = self.T
            Vol = self.Vol

        g1 = project(self.radius*self.E1+self.radius*h1,self.V)
        g1.rename("g1","field")
        g2 = project(self.radius*self.E2+self.radius*h2,self.V)
        g2.rename("g2","field")

        hullmesh = Mesh()
        edit = MeshEditor()
        edit.open(hullmesh,2,3)
        edit.init_vertices(Lnum*4)

        qN = q.compute_vertex_values()
        g1N = g1.compute_vertex_values()
        g2N = g2.compute_vertex_values()
        qN = qN.reshape([3,qN.shape[0]/3])
        g1N = g1N.reshape([3,g1N.shape[0]/3])
        g2N = g2N.reshape([3,g2N.shape[0]/3])
        coords = q.function_space().mesh().coordinates()
        for ix in xrange(qN.shape[1]):
            xi3 = (1.0*ix)/(1.0*Lnum)*Lmax
            cent = qN[:,ix]+coords[ix]
            g1c = g1N[:,ix]
            g2c = g2N[:,ix]
            for jt,theta in enumerate(np.linspace(0.0,2.0*np.pi,NT)):
                edit.add_vertex(NT*ix+jt,
                                np.array(cent+np.cos(theta)*g1c+np.sin(theta)*g2c,dtype=np.float_))

        edit.init_cells( (Lnum-1)*2*NT)
        for ix in xrange(1,qN.shape[1]):
            for jt in xrange(NT-1):
                edit.add_cell(2*NT*(ix-1)+2*jt, NT*ix+jt+1   , NT*(ix-1)+jt, NT*(ix-1)+jt+1)
                edit.add_cell(2*NT*(ix-1)+2*jt+1, NT*ix+jt+1   , NT*ix    +jt, NT*(ix-1)+jt)
            edit.add_cell(2*NT*(ix-1)+2*NT, NT*ix+0   , NT*(ix-1)+NT, NT*(ix-1)+0)
            edit.add_cell(2*NT*(ix-1)+2*NT+1, NT*ix+0   , NT*ix    +NT, NT*(ix-1)+NT)

        edit.close()
        
        vf = File(fname,"ascii")
        vf << hullmesh
        # vf.close()
    def close_file(self):
        "Self explanatory."
        # self.vfile.close()
        pass
