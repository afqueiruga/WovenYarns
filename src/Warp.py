from dolfin import *
import numpy as np

from Fibril import Fibril
from ContactPair import ContactPair
from ContactMultiMesh import ContactMultiMesh

from IPython import embed
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
        self.CMM = ContactMultiMesh()
        self.mdof = MultiMeshDofMap()
        self.mmfs = MultiMeshFunctionSpace()
        self.Tmdof = MultiMeshDofMap()
        self.Tmmfs = MultiMeshFunctionSpace()
        for i,pts in enumerate(endpts):
            me = ProximityTree.create_line(np.array(pts[0]), np.array(pts[1]), 20)
            fib = Fibril(me)
            self.fibrils.append( fib )
            self.CMM.add( fib.mesh)
            self.mmfs.add( fib.W )
            self.mdof.add( fib.W.dofmap() )
            self.Tmmfs.add( fib.S )
            self.Tmdof.add( fib.S.dofmap() )

        self.CMM.build()
        self.mmfs.build(self.CMM, np.array([],dtype=np.intc) )
        # self.mdof = self.mmfs.dofmap()
        self.mdof.build( self.mmfs, np.array([],dtype=np.intc) )
        self.Tmmfs.build(self.CMM, np.array([],dtype=np.intc) )
        self.Tmdof.build( self.Tmmfs, np.array([],dtype=np.intc) )

        self.wx = MultiMeshFunction(self.mmfs)
        self.wv = MultiMeshFunction(self.mmfs)

        for i,fib in enumerate( self.fibrils ):
            fib.build_form() #self.wx.part(i),self.wv.part(i))
            # Initialize the position (zero for now, but I want it to be x)
            # temp = Function(self.fibrils[i].V)
            # temp.interpolate(Expression(("0.0","0.0","0.0")))
            # assign(self.fibrils[i].wx.sub(0), temp)
            # Initialize the velocity (zero for now)
            # temp.interpolate(Expression(("0.0","0.0","0.0")))
            # assign(self.fibrils[i].wv.sub(0), temp)
            # self.mdof.add(self.fibrils[i].W.dofmap())
            # self.mmfs.add(self.fibrils[i].W)
        # self.mdof.build(self.mmfs, np.array([],dtype=np.intc) )
        
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
        for i,fib in enumerate(self.fibrils):
            fib.current_mesh = Mesh(fib.mesh)
            fib.current_mesh.move(fib.wx.sub(0))
            
        self.fibril_pairs = pairs
        self.contacts = []
        for i,p in enumerate(pairs):
            cp = ContactPair(self.fibrils[p[0]].mesh,self.fibrils[p[0]].mesh,
                             self.fibrils[p[1]].mesh,self.fibrils[p[1]].mesh,20)
            cp.make_table()
            self.contacts.append(cp)

    def assemble_thermal_system(self):
        from BroadcastAssembler import BroadcastAssembler
        gN = self.Tmdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)

        M = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(M,dim,2,0, local_dofs, self.Tmdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].MTform, self.Tmdof.part(i)) # crashing here?
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].MTform, self.Tmdof.part(i))
        M.apply('add')

        
        AX = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(AX,dim,2,0, local_dofs, self.Tmdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].ATform, self.Tmdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.sparsity_cell_pair(self.fibrils[self.fibril_pairs[i][0]].ATform, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened)
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].ATform, self.Tmdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].ATform, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].ATform, 
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        AX.apply('add')

        R = Vector()
        
        assem = BroadcastAssembler()
        dim = np.array([gN],dtype=np.intc)
        local_dofs = np.array([0,gN],dtype=np.intc)
        assem.init_global_tensor(R,dim,1,0, local_dofs,self.Tmdof.off_process_owner())
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].FTform, self.Tmdof.part(i))
        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].FTform, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].FTform, 
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        R.apply('add')

        self.MT = M
        self.AT = AX
        self.RT = R
        
    def assemble_system(self):
        from BroadcastAssembler import BroadcastAssembler
        gN = self.mdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)

        M = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(M,dim,2,0, local_dofs, self.mdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].Mform, self.mdof.part(i)) # crashing here?
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].Mform, self.mdof.part(i))
        M.apply('add')
        
        AX = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(AX,dim,2,0, local_dofs, self.mdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].AXform, self.mdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.sparsity_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AXform, 
                                     cp.meshA, self.mdof.part(self.fibril_pairs[i][0]),
                                     cp.meshB, self.mdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened)
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].AXform, self.mdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AXform, 
                                     cp.meshA, self.mdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].AXform, 
                                     cp.meshB, self.mdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        AX.apply('add')

        AV = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(AV,dim,2,0, local_dofs, self.mdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].AVform, self.mdof.part(i))
        for i,cp in enumerate(self.contacts):
            assem.sparsity_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AVform, 
                                     cp.meshA, self.mdof.part(self.fibril_pairs[i][0]),
                                     cp.meshB, self.mdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened)
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].AVform, self.mdof.part(i))
        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AVform, 
                                     cp.meshA, self.mdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].AVform, 
                                     cp.meshB, self.mdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        AV.apply('add')

        R = Vector()
        
        assem = BroadcastAssembler()
        dim = np.array([gN],dtype=np.intc)
        local_dofs = np.array([0,gN],dtype=np.intc)
        assem.init_global_tensor(R,dim,1,0, local_dofs,self.mdof.off_process_owner())
        # assem.sparsity_cell_pair(fibrils[0].Fform, 
        #                           fibrils[0].mesh, mdof.part(0),
        #                           fibrils[1].mesh, mdof.part(1),
        #                           cp.pair_flattened)
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].Fform, self.mdof.part(i))
        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].Fform, 
                                     cp.meshA, self.mdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].Fform, 
                                     cp.meshB, self.mdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        R.apply('add')
        
        self.M = M
        self.AX = AX
        self.AV = AV
        self.R = R

    def apply_thermal_bcs(self,uend=None):
        zero = Constant(0.0)
        if not uend:
            extend = Constant(0.0)
        else:
            extend = uend
        allbound =CompiledSubDomain("on_boundary")
        bcall = MultiMeshDirichletBC(self.Tmmfs,zero,allbound)

        
        left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -1.0)
        bcleft = MultiMeshDirichletBC(self.Tmmfs, zero, left)
        right = CompiledSubDomain("near(x[0], side) && near(x[2], 0.0) && on_boundary", side = 1.0)
        bcright = MultiMeshDirichletBC(self.Tmmfs, extend, right)

        # bcleft.apply(self.AX,self.R)
        # bcright.apply(self.AX,self.R)
        bcall.apply(self.AT,self.RT)
        bcright.apply(self.AT,self.RT)

    def apply_bcs(self,uend=None):
        zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
        if not uend:
            extend = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
        else:
            extend = uend

        left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -1.0)
        bcleft = MultiMeshDirichletBC(self.mmfs, zero, left)
        right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
        bcright = MultiMeshDirichletBC(self.mmfs, extend, right)

        bcleft.apply(self.AX,self.R)
        bcright.apply(self.AX,self.R)
        pass
