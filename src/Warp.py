from dolfin import *
import numpy as np

from Fibril import Fibril
from ContactPair import ContactPair


class Warp():
    """
    Container class of a Fibril assembly. Doesn't neccessarily have to be a warp.

    Handles assembly, ContactPair management, io, etc.
    """
    def __init__(self, endpts):
        """
        Initialize a warp from a list of end points.
        """
        import ProximityTree
        self.fibrils = []
        self.mdof = MultiMeshDofMap()
        for i,pts in enumerate(endpts):
            me = ProximityTree.create_line(np.array(pts[0]), np.array(pts[1]), 50)
            self.fibrils.append( Fibril(me) )
            # Initialize the position (zero for now, but I want it to be x)
            temp = Function(self.fibrils[i].V)
            temp.interpolate(Expression(("0.0","0.0","0.0")))
            assign(self.fibrils[i].wx.sub(0), temp)
            # Initialize the velocity (zero for now)
            temp.interpolate(Expression(("0.0","0.0","0.0")))
            assign(self.fibrils[i].wv.sub(0), temp)
            self.mdof.add(self.fibrils[i].W.dofmap())
        mmfs = MultiMeshFunctionSpace()
        self.mdof.build(mmfs, np.array([],dtype=np.intc) )
        self.contacts = []

    def output_states(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.write_file(fname.format(j),i)

    def output_contacts(self,fname):
        for j,c in enumerate(self.contacts):
            c.output_file(fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"pairs"),
                          fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"gamma") )
        
    def create_contacts(self,pairs=None):
        """
        Create all of the neccessary contact pairs.
        If a list of pairs isn't specified, just create the n^2 list.
        """

        if not pairs:
            pairs = [ (j,i) for j in xrange(len(self.fibrils)) for i in xrange(j+1,len(self.fibrils))  ]

        self.fibril_pairs = pairs
        self.contacts = []
        for i,p in enumerate(pairs):
            cp = ContactPair(self.fibrils[p[0]].mesh, self.fibrils[p[1]].mesh,20)
            cp.make_table()
            self.contacts.append(cp)
        
