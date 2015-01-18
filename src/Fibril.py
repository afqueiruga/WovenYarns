from dolfin import *
import numpy as np

from BeamForm import BeamForm
from ThermalForm import ThermalForm
from MultiphysicsForm import MultiphysicsForm

"""

This is a base class for the director-based Fibril

"""
from IPython import embed
class Fibril():
    def __init__(self,mesh,orientation=0,wx=None,wv=None):
        """
        Create a new Fibril on a given mesh. It better be a line mesh.
        Dirrichlet BCs by default.
        """
        self.mesh=mesh
        self.current_mesh = Mesh(mesh)
        self.V = VectorFunctionSpace(mesh,"CG",1)
        self.S = FunctionSpace(mesh,"CG",1)
        self.W = MixedFunctionSpace([self.V,self.V,self.V,self.S,self.S])
        if wx and wv:
            build_form(wx,wv)
        self.orientation=orientation

    def build_multi_form(self,wx=None,wv=None):
        if not wx and not wv:
            self.wx = Function(self.W)
            self.wv = Function(self.W)
        else:
            self.wx = Function(wx)
            self.wv = Function(wv)

        self.X0 = Function(self.V)
        self.X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
        
        self.Fform,self.Mform,self.AXform,self.AVform = \
          MultiphysicsForm(self.W,self.V,self.S,self.wx,self.wv,self.X0, self.orientation)

        self.Height = 1.0
        self.Width = 1.0
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

        q,h1,h2, Tnull, Vnull    = self.wx.split()
        vq,vh1,vh2,T,Vol = self.wv.split()
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
        g1 = project(self.Height/2.0*Constant((0.0,1.0,0.0))+h1,self.V)
        g1.rename("g1","field")
        g2 = project(self.Width/2.0*Constant((0.0,0.0,1.0))+h2,self.V)
        g2.rename("g2","field")

        self.T.rename("Tself","field")
        self.X0.rename("X0","field")
        vfile.write(i,[q,h1,h2,g1,g2, vq,vh1,vh2, T, Vol,self.X0,self.T],[])
        # vfile = VTKAppender(fname,"ascii")
    def write_surface(self,fname,i,Lnum=200,Lmax=100.0):
        """
        Pop out one file on the currently active file.
        """
        from multiwriter.multiwriter import VTKAppender
        vfile = VTKAppender(fname,"ascii")

        q,h1,h2, Tnull, Vnull    = self.wx.split()
        vq,vh1,vh2,T,Vol = self.wv.split()

        g1 = project(self.Height/2.0*Constant((0.0,1.0,0.0))+h1,self.V)
        g1.rename("g1","field")
        g2 = project(self.Width/2.0*Constant((0.0,0.0,1.0))+h2,self.V)
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

            edit.add_vertex(4*ix+0,np.array(cent+g1c+g2c,dtype=np.float_))
            edit.add_vertex(4*ix+1,np.array(cent-g1c+g2c,dtype=np.float_))
            edit.add_vertex(4*ix+2,np.array(cent-g1c-g2c,dtype=np.float_))
            edit.add_vertex(4*ix+3,np.array(cent+g1c-g2c,dtype=np.float_))
        edit.init_cells( (Lnum-1)*8)
        for ix in xrange(1,qN.shape[1]):
            edit.add_cell(8*(ix-1)+0, 4*ix+0   , 4*(ix-1)+3, 4*(ix-1)+0)
            edit.add_cell(8*(ix-1)+1, 4*ix+0   , 4*ix    +3, 4*(ix-1)+3)
            
            edit.add_cell(8*(ix-1)+2, 4*ix+1   , 4*ix    +0, 4*(ix-1)+0)
            edit.add_cell(8*(ix-1)+3, 4*ix+1   , 4*(ix-1)+0, 4*(ix-1)+1)
            
            edit.add_cell(8*(ix-1)+4, 4*ix+1   , 4*(ix-1)+1, 4*(ix  )+2)
            edit.add_cell(8*(ix-1)+5, 4*ix+2   , 4*(ix-1)+1, 4*(ix-1)+2)
            
            edit.add_cell(8*(ix-1)+6, 4*ix+3   , 4*ix    +2, 4*(ix-1)+2)
            edit.add_cell(8*(ix-1)+7, 4*ix+3   , 4*(ix-1)+2, 4*(ix-1)+3)
        edit.close()
        vf = File(fname,"ascii")
        vf << hullmesh
        # vf.close()
    def close_file(self):
        "Self explanatory."
        # self.vfile.close()
        pass
