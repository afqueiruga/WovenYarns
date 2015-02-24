from ProblemDescription import *
from QuadraturePoints import RectOuterProd,CircCart2D

E = 10.0
nu = 0.0
default_properties = {
    'mu':E/(2*(1 + nu)),
    'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
    'mu_alpha':-0.0,
    'rho': 1.0,
    'T_cp':1.0,
    'T_k': 1.0,
    'em_B':Constant((0.0,0.0,0.0)),
    'em_sig':10.0,
    'radius':0.15,
    'f_dens_ext':Constant((0.0,0.0,0.0)),
    'dissipation':1.0e-1,
    'contact_penalty': 4.0,
    'mech_bc_trac_0': Constant((0.0,0.0,0.0)),
    'mech_bc_trac_1': Constant((0.0,0.0,0.0)),
    'em_bc_J_0': 0.0,
    'em_bc_J_1': 0.0,
    'em_bc_r_0': 0.0,
    'em_bc_r_1': 0.0
    

}
class DecoupledProblem(ProblemDescription):
    """ This creates a multiphysics problem with different spaces for each field """
    def __init__(self,mesh,properties, boundaries=None,buildform=True, orientation=0, order=(1,1)):
        if properties.has_key("orientation"):
            orientation = properties["orientation"]
            properties.pop("orientation",None)

        if orientation == 0:
            properties["Ez"] = Constant((1.0,0.0,0.0))
            properties["E1"] = Constant((0.0,1.0,0.0))
            properties["E2"] = Constant((0.0,0.0,1.0))
        elif orientation == 1:
            properties["Ez"] = Constant((0.0,1.0,0.0))
            properties["E1"] = Constant((0.0,0.0,1.0))
            properties["E2"] = Constant((1.0,0.0,0.0))
        else:
            properties["Ez"] = Constant((0.0,0.0,1.0))
            properties["E1"] = Constant((1.0,0.0,0.0))
            properties["E2"] = Constant((0.0,1.0,0.0))
        
        p = default_properties.copy()
        p.update(properties)

        self.orientation = orientation
        self.order = order
        
        ProblemDescription.__init__(self,mesh,p,boundaries,False)
        
        if not properties.has_key("X0"):
            X0 = Function(self.spaces['V'])
            X0.interpolate(Expression(("x[0]","x[1]","x[2]")))
            self.properties["X0"] = X0
            
        if buildform:
            self.forms = self.Build_Forms()

    def Declare_Spaces(self):
        V = VectorFunctionSpace(self.mesh,"CG",self.order[0])
        S = FunctionSpace(self.mesh,"CG",self.order[1])
        d = {
            'S': S,
            'V': V,
            'W': MixedFunctionSpace([V,V,V])
        }
        return d

    def Declare_Fields(self):
        self.space_key = {'wx':'W',
                'wv':'W',
                'T':'S',
                'Vol':'S'}
        return {'wx':Function(self.spaces['W']),
                'wv':Function(self.spaces['W']),
                'T' :Function(self.spaces['S']),
                'Vol':Function(self.spaces['S'])}


    def Build_Forms(self):
        # Set up the functions, trials, and tests.
        W = self.spaces['W']
        space_S = self.spaces['S']
        space_V = self.spaces['V']
        
        wx = self.fields['wx']
        wv = self.fields['wv']
        T = self.fields['T']
        Vol = self.fields['Vol']
        
        vq,vh1,vh2 = split(wv)
        q,h1,h2 = split(wx)

        tw = TestFunction(W)
        tvq,tvh1,tvh2 = split(tw)
        tT = TestFunction(space_S)
        tVol = TestFunction(space_S)
        
        Delw = TrialFunction(W)
        DelT = TrialFunction(space_S)
        DelVol = TrialFunction(space_S)

        dvdtW = TrialFunction(W)
        dvqdt,dvh1dt,dvh2dt = split(dvdtW)
        dTdt = TrialFunction(space_S)

        # Set up the measures
        ds = Measure("ds")[self.boundaries]


        # Fetch the material properties
        PROP = self.properties
        mu       = PROP['mu']
        lmbda    = PROP['lambda']
        mu_alpha = PROP['mu_alpha']
        rho      = PROP['rho']
        T_cp     = PROP['T_cp']
        T_k      = PROP['T_k']

        em_B     = PROP['em_B']
        em_sig   = PROP['em_sig']

        radius   = PROP['radius']

        f_dens_ext = PROP['f_dens_ext']
        dissipation = PROP['dissipation']

        contact_penalty = PROP['contact_penalty']
        
        # Neumann BC values
        em_bc_J_0 = PROP['em_bc_J_0']
        em_bc_r_0 = PROP['em_bc_r_0']
        em_bc_J_1 = PROP['em_bc_J_1']
        em_bc_r_1 = PROP['em_bc_r_1']
        mech_bc_trac_0 = PROP['mech_bc_trac_0']
        mech_bc_trac_1 = PROP['mech_bc_trac_1']
        
        orientation = self.orientation
        Ez = PROP['Ez']
        E1 = PROP['E1']
        E2 = PROP['E2']

        X0 = PROP['X0']
        I = Identity(W.cell().geometric_dimension())


        # Perform the cross section integration
        FMechTot = None
        FTherTot = None
        FElecTot = None
        MassMech = None
        MassTher = None

        p_t0_0 = None
        p_t1_0 = None
        p_t2_0 = None
        p_t0_1 = None
        p_t1_1 = None
        p_t2_1 = None
        p_J_0 = None
        p_J_1 = None
        GPS2D = CircCart2D[4]
        J0 = radius*radius
        for z1,z2,weight in GPS2D:
            # The fields at these points
            # TODO: what if I get rid of Constant?
            u = q + radius*Constant(z1)*h1 + radius*Constant(z2)*h2
            v = vq + radius*Constant(z1)*vh1 + radius*Constant(z2)*vh2
            dvdt = dvqdt + radius*Constant(z1)*dvh1dt + radius*Constant(z2)*dvh2dt

            # Take the Gauteax derivatives to get test fields
            tu = derivative(u,wx,tw)
            tv = derivative(v,wv,tw)
            
            # Mechanical strain energy
            F = I + outer(u.dx(orientation),Ez) + outer(h1,E1) + outer(h2, E2 )
            C = F.T*F
            Ic = tr(C)
            J = det(F)
            mu_pt = mu+mu_alpha*T
            Psi = ((mu_pt/2)*(Ic - 3) - mu_pt*ln(J) + (lmbda/2)*(ln(J))**2)

            # Take the Gateaux derivative of the strain energy to get internal force
            FInt = derivative(-Psi,wx,tw)

            # Voltage Form
            vB_ref = dot(F.T*cross(v,em_B),Ez)
            # It shouldn't be sig in here, but it actually doesn't matter
            V_FLoc = tVol.dx(orientation) * em_sig * Vol.dx(orientation) \
              - inner(tVol.dx(orientation),em_sig*(vB_ref))

            em_Egal = inv(F).T*(-Vol.dx(orientation) + vB_ref)*Ez
            em_J = em_sig*em_Egal
            
            
            # Thermal Form
            Gradv = outer(v.dx(orientation),Ez) + outer(vh1,E1) + outer(vh2,E2)
            S = (mu_pt)*I+(-mu_pt+lmbda*ln(J))*inv(C).T
            T_FLoc = -(tT.dx(orientation) * T_k * T.dx(orientation)) + tT*dot(em_Egal,em_J)

            # Forces
            ey = (Ez+q.dx(orientation)) /sqrt( inner(Ez+q.dx(orientation),Ez+q.dx(orientation)) )
            FExt = inner(tv,cross(em_J,em_B)) + inner(tv,-dissipation*v) + inner(tv, f_dens_ext)

            # Neumann type BCs:
            Mech_FBCLoc =  -weight*J0*inner(tv,mech_bc_trac_0)*ds(0) \
                        + weight*J0*inner(tv,mech_bc_trac_1)*ds(1)
            V_FBCLoc = -weight*J0*tVol*(em_bc_r_0*Vol+em_bc_J_0)*ds(0) \
                      + weight*J0*tVol*(em_bc_r_1*Vol+em_bc_J_1)*ds(1)

            # These are evaluation forms, no test functions
            p_t0_0_Loc = -weight*J0*dot(Constant((1.0,0.0,0.0)),F*S*Ez)*ds(0)
            p_t1_0_Loc = -weight*J0*dot(Constant((0.0,1.0,0.0)),F*S*Ez)*ds(0)
            p_t2_0_Loc = -weight*J0*dot(Constant((0.0,0.0,1.0)),F*S*Ez)*ds(0)
            p_t0_1_Loc =  weight*J0*dot(Constant((1.0,0.0,0.0)),F*S*Ez)*ds(1)
            p_t1_1_Loc =  weight*J0*dot(Constant((0.0,1.0,0.0)),F*S*Ez)*ds(1)
            p_t2_1_Loc =  weight*J0*dot(Constant((0.0,0.0,1.0)),F*S*Ez)*ds(1)

            p_J_0_Loc = -weight*J0*dot(Ez,em_J)*ds(0)
            p_J_1_Loc =  weight*J0*dot(Ez,em_J)*ds(1)
            
            # Finalize
            FMechLoc = weight*J0*( FInt + FExt ) * dx
            FTherLoc = weight*J0*( T_FLoc ) * dx
            FElecLoc = weight*J0*( V_FLoc ) *dx + V_FBCLoc
            MMechLoc = weight*J0*( inner(tv,rho*dvdt) )*dx
            MTherLoc = weight*J0*( inner(tT,rho*T_cp*dTdt) )*dx
            
            FMechTot = FMechLoc if FMechTot is None else FMechTot + FMechLoc
            FTherTot = FTherLoc if FTherTot is None else FTherTot + FTherLoc
            FElecTot = FElecLoc if FElecTot is None else FElecTot + FElecLoc
            MassMech = MMechLoc if MassMech is None else MassMech + MMechLoc
            MassTher = MTherLoc if MassTher is None else MassTher + MTherLoc

            p_t0_0 = p_t0_0_Loc if p_t0_0 is None else p_t0_0 + p_t0_0_Loc
            p_t1_0 = p_t1_0_Loc if p_t1_0 is None else p_t1_0 + p_t1_0_Loc
            p_t2_0 = p_t2_0_Loc if p_t2_0 is None else p_t2_0 + p_t2_0_Loc
            p_t0_1 = p_t0_1_Loc if p_t0_1 is None else p_t0_1 + p_t0_1_Loc
            p_t1_1 = p_t1_1_Loc if p_t1_1 is None else p_t1_1 + p_t1_1_Loc
            p_t2_1 = p_t2_1_Loc if p_t2_1 is None else p_t2_1 + p_t2_1_Loc
            p_J_0 = p_J_0_Loc if p_J_0 is None else p_J_0 + p_J_0_Loc
            p_J_1 = p_J_1_Loc if p_J_1 is None else p_J_1 + p_J_1_Loc
            
        # Contact Forms
        xr = X0 + q
        dist = sqrt(dot(jump(xr),jump(xr)))
        overlap = (2.0*avg(radius)-dist)
        ContactForm = -dot(jump(tvq),
                        conditional(ge(overlap,0.0), -contact_penalty*overlap,0.0)*jump(xr)/dist) \
                        *dc(0, metadata={"num_cells": 2,"special":"contact"})
        
        # Functional derivatives
        FMechTot += ContactForm
        AX = derivative(FMechTot,wx,Delw)
        AV = derivative(FMechTot,wv,Delw)
        AT = derivative(FTherTot,T,DelT)
        AE = derivative(FElecTot,Vol,DelVol)

        # 
        # Dictionize and return
        return {
            'F' : Form(FMechTot),
            'AX': Form(AX),
            'AV': Form(AV),
            'M' : Form(MassMech),
            'FT': Form(FTherTot),
            'AT': Form(AT),
            'MT': Form(MassTher),
            'FE': Form(FElecTot),
            'AE': Form(AE),
            
            'p_t0_0':Form(p_t0_0),
            'p_t1_0':Form(p_t1_0),
            'p_t2_0':Form(p_t2_0),
            'p_t0_1':Form(p_t0_1),
            'p_t1_1':Form(p_t1_1),
            'p_t2_1':Form(p_t2_1),
            'p_J,0':Form(p_J_0),
            'p_J_1':Form(p_J_1)
            }

    def split_for_io(self):
        q,h1,h2,     = self.fields['wx'].split()
        vq,vh1,vh2 = self.fields['wv'].split()
        
        g1 = project(self.properties['radius']*(self.properties['E1']+h1),
                     self.spaces['V'])
        g2 = project(self.properties['radius']*(self.properties['E2']+h2),
                     self.spaces['V'])     
        return {
                'q':q, 'h1':h1, 'h2':h2,
                'vq':vq, 'vh1':vh1, 'vh2':vh2,
                'g1':g1, 'g2':g2,
                'T':self.fields['T'], 'Vol':self.fields['Vol']
            }
