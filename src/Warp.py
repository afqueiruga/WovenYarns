from dolfin import *
import numpy as np

from Fibril import Fibril
from ContactMultiMesh import ContactMultiMesh
from ContactGroup import ContactGroup

from IPython import embed
class Warp():
    """
    Container class of a Fibril assembly. Doesn't neccessarily have to be a warp.

    Handles assembly, ContactPair management, io, etc.
    """
    def __init__(self, endpts, props, defprops, Nelems,Prob, order=(1,1),cutoff=1.0):
        """
        Initialize a warp from a list of end points.
        """
        self.fibrils = []
        self.CMM = ContactMultiMesh()

        # Initialize all of the fibrils
        for e,p,ne in zip(endpts, props,Nelems):
            prop = defprops.copy()
            prop.update(p)

            fib = Fibril(e,ne,prop,Prob, order=order)

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
        self.space_key = {}
        for name in self.fibrils[0].problem.fields:
            self.fields[name] = MultiMeshFunction(self.spaces[ self.fibrils[0].problem.space_key[name] ] )
            self.space_key[name] = self.fibrils[0].problem.space_key[name]

        self.CG = ContactGroup([fib.mesh for fib in self.fibrils],
                               radii=[fib.problem.properties['radius'].vector()[0] for fib in self.fibrils],
                               cutoff=cutoff)
        self.mcache = {}

    def output_states(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.WriteFile(fname.format(j),i)
    def output_surfaces(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.WriteSurface(fname.format(j),i)
    def output_solids(self,fname,i):
        for j,fib in enumerate(self.fibrils):
            fib.WriteSolid(fname.format(j),i)
    def output_contacts(self,fname):
        for j,c in enumerate(self.contacts):
            c.output_file(fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"pairs"),
                          fname.format(self.fibril_pairs[j][0],self.fibril_pairs[j][1],"gamma") )

    
    def create_contacts(self,pairs=None,cutoff=0.3):
        """
        Create all of the neccessary contact pairs.
        If a list of pairs isn't specified, just create the n^2 list.
        """

        for i,fib in enumerate(self.fibrils):
            fib.current_mesh = Mesh(fib.problem.mesh)
            fields = fib.problem.split_for_io()
            vv = fields['q'].compute_vertex_values()
            # embed()
            vv = vv.reshape([3,vv.shape[0]/3])
            for ix in xrange(vv.shape[1]):
                fib.current_mesh.coordinates()[ix] += vv[:,ix]
            # fib.current_mesh.move(fib.problem.fields['wx'].sub(0))
        self.CG.cutoff = cutoff    
        self.CG.CreateTables([f.current_mesh for f in self.fibrils])
    
        # for i,p in enumerate(pairs):
        #     cp = ContactPair(self.fibrils[p[0]].current_mesh,self.fibrils[p[0]].mesh,
        #                      self.fibrils[p[1]].current_mesh,self.fibrils[p[1]].mesh,10,cutoff)
        #     cp.make_table()
        #     self.contacts.append(cp)

    # TODO: Build a table of forms -> spaces?
    def assemble_form(self, form_key, space_key):
        from BroadcastAssembler import BroadcastAssembler
        rank = self.fibrils[0].problem.forms[form_key].rank()
        mmdofmap = self.spaces[space_key].dofmap()
        gN = mmdofmap.global_dimension()

        if rank==1:
            A = Vector()
            dim = np.array([gN],dtype=np.intc)
            local_dofs = np.array([0,gN],dtype=np.intc)
        elif rank==2:
            A = Matrix()
            dim = np.array([gN,gN],dtype=np.intc)
            local_dofs = np.array([0,gN,0,gN],dtype=np.intc)
        else:
            print "I don't do other ranks."

        assem = BroadcastAssembler()
        assem.init_global_tensor(A,dim,rank,0, local_dofs, mmdofmap.off_process_owner())
        if rank==2:
            for i,fib in enumerate(self.fibrils):
                assem.sparsity_form(self.fibrils[i].problem.forms[form_key], mmdofmap.part(i))
            for i, (a,b, (etab,stab,xtab,Xtab,dtab)) in enumerate(self.CG.active_pairs):
                assem.sparsity_cell_pair(self.fibrils[a].problem.forms[form_key], 
                                     self.fibrils[a].mesh, mmdofmap.part(a),
                                     self.fibrils[b].mesh, mmdofmap.part(b),
                                     etab.flatten() )
        assem.sparsity_apply()

        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].problem.forms[form_key], mmdofmap.part(i))

        for i, (a,b, (etab,stab,xtab,Xtab,dtab)) in enumerate(self.CG.active_pairs):
            assem.assemble_cell_pair(self.fibrils[a].problem.forms[form_key], 
                                     self.fibrils[a].mesh, mmdofmap.part(a),
                                     self.fibrils[b].problem.forms[form_key], 
                                     self.fibrils[b].mesh, mmdofmap.part(b),
                                     etab.flatten(),
                                     Xtab.flatten(),
                                     self.CG.chi_n_max)
        A.apply('add')

        self.mcache[form_key] = A

        return A

    def assemble_forms(self,keys,spaces):
        if spaces is not list:
            spaces = [ spaces for x in keys ]
        r =  [self.assemble_form(f,s) for f,s in zip(keys,spaces)]
        return r

    def update(self):
        for key,f in self.fields.iteritems():
            mdof = self.spaces[self.space_key[key]].dofmap()
            for i,fib in enumerate(self.fibrils):
                fib.problem.fields[key].vector()[:] = f.vector()[mdof.part(i).dofs()]

    def pull_fibril_fields(self):
        for key,f in self.fields.iteritems():
            mdof = self.spaces[self.space_key[key]].dofmap()
            for i,fib in enumerate(self.fibrils):
                 f.vector()[mdof.part(i).dofs()] = fib.problem.fields[key].vector()[:]
