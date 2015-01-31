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
    def __init__(self, endpts, monolithic=True,cutoff=0.3):
        """
        Initialize a warp from a list of end points.
        """
        import ProximityTree
        self.fibrils = []
        self.CMM = ContactMultiMesh()

        self.mdof = MultiMeshDofMap()
        self.mmfs = MultiMeshFunctionSpace()
        if not monolithic:
            self.Tmdof = MultiMeshDofMap()
            self.Tmmfs = MultiMeshFunctionSpace()
            self.Vmdof = MultiMeshDofMap()
            self.Vmmfs = MultiMeshFunctionSpace()
        for i,pts in enumerate(endpts):
            me = ProximityTree.create_line(np.array(pts[0]), np.array(pts[1]), 40)
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
            fib = Fibril(me,orientation,monolithic)
            self.fibrils.append( fib )
            self.CMM.add( fib.mesh )
            self.mmfs.add( fib.W )
            self.mdof.add( fib.W.dofmap() )
            if not monolithic:
                self.Tmmfs.add( fib.S )
                self.Tmdof.add( fib.S.dofmap() )
                self.Vmmfs.add( fib.S )
                self.Vmdof.add( fib.S.dofmap() )
                
        self.CMM.build()
        self.mmfs.build(self.CMM, np.array([],dtype=np.intc) )
        # self.mdof = self.mmfs.dofmap()
        self.mdof.build( self.mmfs, np.array([],dtype=np.intc) )
        if not monolithic:
            self.Tmmfs.build(self.CMM, np.array([],dtype=np.intc) )
            self.Tmdof.build( self.Tmmfs, np.array([],dtype=np.intc) )
            self.Vmmfs.build(self.CMM, np.array([],dtype=np.intc) )
            self.Vmdof.build( self.Vmmfs, np.array([],dtype=np.intc) )

        self.wx = MultiMeshFunction(self.mmfs)
        self.wv = MultiMeshFunction(self.mmfs)
        if not monolithic:
            self.T = MultiMeshFunction(self.Tmmfs)
            self.V = MultiMeshFunction(self.Vmmfs)
            
        for i,fib in enumerate( self.fibrils ):
            if monolithic:
                fib.build_multi_form()
            else:
                fib.build_iterative_form()
            # fib.build_thermal_form()
            
        self.contacts = []
        self.contact_cutoff = cutoff

        self.monolithic = monolithic
    
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
                             self.fibrils[p[1]].mesh,self.fibrils[p[1]].mesh,10,self.contact_cutoff)
            cp.make_table()
            self.contacts.append(cp)

    def assemble_thermal_system(self):
        from BroadcastAssembler import BroadcastAssembler
        gN = self.Tmdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)

        AT = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(AT,dim,2,0, local_dofs, self.Tmdof.off_process_owner())
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
        AT.apply('add')

        RT = Vector()
        
        assem = BroadcastAssembler()
        dim = np.array([gN],dtype=np.intc)
        local_dofs = np.array([0,gN],dtype=np.intc)
        assem.init_global_tensor(RT,dim,1,0, local_dofs,self.Tmdof.off_process_owner())
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
        RT.apply('add')

        self.AT = AT
        self.RT = RT

        
    def assemble_voltage_system(self):
        from BroadcastAssembler import BroadcastAssembler
        gN = self.Vmdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)

        AVol = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(AVol,dim,2,0, local_dofs, self.Tmdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].AVolForm, self.Tmdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.sparsity_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AVolForm, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened)
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].AVolForm, self.Tmdof.part(i))

        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].AVolForm, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].AVolForm, 
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        AVol.apply('add')

        RT = Vector()
        
        assem = BroadcastAssembler()
        dim = np.array([gN],dtype=np.intc)
        local_dofs = np.array([0,gN],dtype=np.intc)
        assem.init_global_tensor(RVol,dim,1,0, local_dofs,self.Tmdof.off_process_owner())
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].FVform, self.Tmdof.part(i))
        for i,cp in enumerate(self.contacts):
            assem.assemble_cell_pair(self.fibrils[self.fibril_pairs[i][0]].FVform, 
                                     cp.meshA, self.Tmdof.part(self.fibril_pairs[i][0]),
                                     self.fibrils[self.fibril_pairs[i][1]].FVform, 
                                     cp.meshB, self.Tmdof.part(self.fibril_pairs[i][1]),
                                     cp.pair_flattened,
                                     cp.chi_X_table.flatten(),
                                     cp.chi_n_max)
        RVol.apply('add')

        self.AVol = AVol
        self.RVol = RVol

        
    def assemble_mass(self):
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
        
        self.M = M

        if self.monolithic:
            return
        gN = self.Tmdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)
        
        MT = Matrix()
        assem = BroadcastAssembler()
        assem.init_global_tensor(MT,dim,2,0, local_dofs, self.Tmdof.off_process_owner())
        for i,fib in enumerate(self.fibrils):
            assem.sparsity_form(self.fibrils[i].MTform, self.Tmdof.part(i)) # crashing here?
        assem.sparsity_apply()
        for i,fib in enumerate(self.fibrils):
            assem.assemble_form(self.fibrils[i].MTform, self.Tmdof.part(i))
        MT.apply('add')

        self.MT = MT
        
    def assemble_system(self):
        from BroadcastAssembler import BroadcastAssembler
        gN = self.mdof.global_dimension()
        dim = np.array([gN,gN],dtype=np.intc)
        local_dofs = np.array([0,gN,0,gN],dtype=np.intc)

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

    def apply_multi_bcs(self,uend=None):
        zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0))
        if not uend:
            extend = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0))
        else:
            extend = uend

        left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -10.0)
        bcleft = MultiMeshDirichletBC(self.mmfs, zero, left)
        right = CompiledSubDomain("near(x[0], side) && near(x[2], -1.0) && on_boundary", side = 10.0)
        bcright = MultiMeshDirichletBC(self.mmfs, extend, right)

        bcleft.apply(self.AX,self.R)
        bcright.apply(self.AX,self.R)
        
    def apply_bcs(self,uend=None):
        zero = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
        if not uend:
            extend = Constant((0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0))
        else:
            extend = uend

        left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -10.0)
        bcleft = MultiMeshDirichletBC(self.mmfs, zero, left)
        right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 10.0)
        bcright = MultiMeshDirichletBC(self.mmfs, extend, right)

        bcleft.apply(self.AX,self.R)
        bcright.apply(self.AX,self.R)
        pass

    def update(self):
        " Push the global vector into the individual fibril vectors "
        for i,fib in enumerate(self.fibrils):
            fib.wx.vector()[:] = self.wx.vector()[ self.mdof.part(i).dofs() ]
            fib.wv.vector()[:] = self.wv.vector()[ self.mdof.part(i).dofs() ]
