from dolfin import *
import numpy as np

from Fibril import Fibril
from ContactPair import ContactPair
from ContactMultiMesh import ContactMultiMesh

class Warp():
    """
    Container class of a Fibril assembly. Doesn't neccessarily have to be a warp.

    Handles assembly, ContactPair management, io, etc.
    """
    def __init__(self, endpts, props, defprops, Prob, order=(1,1)):
        """
        Initialize a warp from a list of end points.
        """
        self.fibrils = []
        self.CMM = ContactMultiMesh()

        # Initialize all of the fibrils
        for e,p in zip(endpts, props):
            prop = defprops.update(p)
            fib = Fibril(e,20,p,Prob, order=order)

            self.fibrils.append(fib)
            self.CMM.add(fib.mesh)
            
        # Tabulate all of the function spaces
        self.spaces = {}
        for name,space in self.fibrils[0].problem.spaces.iteritems():
            self.spaces[name] = MultiMeshFunctionSpace()
        for fib in self.fibrils:
            for name,space in fib.problem.spaces.iteritems():
                self.spaces[name].add(space)
        for name,space in self.spaces.iteritems():
            space.build(self.CMM, np.array([],dtype=np.intc) )

        # Build all of the fields on the multimeshspaces
        self.fields = {}
        for name in self.fibrils[0].problem.fields:
            self.fields[name] = MultiMeshFunction(self.spaces[ self.fibrils[0].problem.space_key[name] ] )

        self.contacts = []

    def output_states(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.write_file(fname.format(j),i)
    def output_surfaces(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.write_surface(fname.format(j),i)
    def output_contacts(self,fname):
        for j,c in enumerate(self.contacts):
            c.output_file(fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"pairs"),
                          fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"gamma") )

    
