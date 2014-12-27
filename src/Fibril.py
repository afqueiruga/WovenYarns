from dolfin import *
import numpy as np

from BeamForm import BeamForm

"""

This is a base class for the director-based Fibril

"""
from IPython import embed
class Fibril():
    def __init__(self,mesh,wx=None,wv=None):
        """
        Create a new Fibril on a given mesh. It better be a line mesh.
        Dirrichlet BCs by default.
        """
        self.mesh=mesh
        self.V = VectorFunctionSpace(mesh,"CG",1)
        self.W = MixedFunctionSpace([self.V,self.V,self.V])
        if wx and wv:
            build_form(wx,wv)
        
    def build_form(self,wx=None,wv=None):
        if not wx and not wv:
            self.wx = Function(self.W)
            self.wv = Function(self.W)
        else:
            self.wx = Function(wx)
            self.wv = Function(wv)

        self.Fform,self.Mform,self.AXform,self.AVform = \
          BeamForm(self.W,self.V,self.wx,self.wv)

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
    def start_file(self,fname):
        "Open an empty file with fname."
        pass
    
    def write_file(self,fname,i):
        """
        Pop out one file on the currently active file.
        """
        from multiwriter.multiwriter import VTKAppender
        vfile = VTKAppender(fname,"ascii")

        q,h1,h2    = self.wx.split()
        vq,vh1,vh2 = self.wv.split()
        q.rename("q","field")
        h1.rename("h1","field")
        h2.rename("h2","field")
        vq.rename("vq","field")
        vh1.rename("vh1","field")
        vh2.rename("vh2","field")
        # q1 = Function(V)
        # q1.interpolate(Height/2.0*Constant((0.0,1.0,0.0))+h1)
        # q2 = Function(V)
        # q2.interpolate(Width/2.0*Constant((0.0,0.0,1.0))+h2)
        q1 = project(self.Height/2.0*Constant((0.0,1.0,0.0))+h1,self.V)
        q1.rename("q1","field")
        q2 = project(self.Width/2.0*Constant((0.0,0.0,1.0))+h2,self.V)
        q2.rename("q2","field")

        vfile.write(i,[q,h1,h2,q1,q2, vq,vh1,vh2],[])
        vfile = VTKAppender(fname,"ascii")

    def close_file(self):
        "Self explanatory."
        # self.vfile.close()
        pass
