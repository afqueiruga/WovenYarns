from ProblemDescription import *
from QuadraturePoints import RectOuterProd,CircCart2D

import numpy as np

class MultiphysicsBaseProblem(ProblemDescription):
    """ This creates a multiphysics problem with different spaces for each field """

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
        'em_seebeck':0.0,
        'radius':0.15,
        'f_dens_ext':Constant((0.0,0.0,0.0)),
        'dissipation':1.0e-1,
        'contact_penalty': 4.0,
        'contact_em': 1.0,
        'contact_temp': 0.1,
        'mech_bc_trac_0': Constant((0.0,0.0,0.0)),
        'mech_bc_trac_1': Constant((0.0,0.0,0.0)),
        'em_bc_J_0': 0.0,
        'em_bc_J_1': 0.0,
        'em_bc_r_0': 0.0,
        'em_bc_r_1': 0.0
    }
    
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
        
        p = self.default_properties.copy()
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
        pass

    def Declare_Fields(self):
        pass

    def fields_and_tests(self):
        pass
        
    def fem_forms(self):
        pass
    
    def Build_Forms(self):
        # Fetch the fields
        wx,q,h1,h2, wv, vq,vh1,vh2, T, Vol, \
          tw, tvq,tvh1,tvh2, tT,tVol, \
          Delw,DelT,DelVol, \
          dvdtW,dvqdt,dvh1dt,dvh2dt,dTdt = self.fields_and_tests()
        
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
        em_seebeck   = PROP['em_seebeck']

        radius   = PROP['radius']

        f_dens_ext = PROP['f_dens_ext']
        dissipation = PROP['dissipation']

        contact_penalty = PROP['contact_penalty']
        contact_em = PROP['contact_em']
        contact_temp = PROP['contact_temp']

        
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
        I = Identity(self.spaces['W'].cell().geometric_dimension())

        
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

        f_I = None
        f_Pi1 = None
        
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
            F0 = I + outer(q.dx(orientation),Ez) + outer(h1,E1) + outer(h2, E2 )
            ey = (Ez+q.dx(orientation)) /sqrt( inner(Ez+q.dx(orientation),Ez+q.dx(orientation)) )

            vB_ref = dot(F0.T*cross(vq,em_B),Ez)
            # It should be perm in here, but it actually doesn't matter
            V_FLoc = tVol.dx(orientation)  * Vol.dx(orientation) \
              - inner(tVol.dx(orientation),(vB_ref)) \
              - inner(tVol.dx(orientation),-em_seebeck*T.dx(orientation))

              # what's up with these equations?
              # OK, em_Egal is in the current configuration,
              # Oh, but the texturing of sigma might make J and E not allign...
              # But... inv(F).T vs 1/J F? I think they line up

            em_Egal = inv(F0).T*(-Vol.dx(orientation) + vB_ref)*Ez
            em_J = 1/det(F0)*F0*em_sig*(-Vol.dx(orientation) + vB_ref)*Ez
            em_Joule = 1/det(F0)*(-Vol.dx(orientation) + vB_ref)*em_sig*(-Vol.dx(orientation) + vB_ref)
            
            # Thermal Form
            Gradv = outer(v.dx(orientation),Ez) + outer(vh1,E1) + outer(vh2,E2)
            S = (mu_pt)*I+(-mu_pt+lmbda*ln(J))*inv(C).T
            T_FLoc = -(tT.dx(orientation) * T_k * T.dx(orientation)) + tT*em_Joule + tT*inner(S,Gradv)

            # Forces
            FExt = inner(tv,cross(em_J,em_B)) + inner(tv,-dissipation*v) + inner(tv, f_dens_ext)

            # Neumann type BCs:
            Mech_FBCLoc =  -weight*J0*inner(tv,mech_bc_trac_0)*ds(1) \
                        + weight*J0*inner(tv,mech_bc_trac_1)*ds(2)
            V_FBCLoc = -weight*J0*tVol*(em_bc_r_0*Vol+em_bc_J_0)*ds(1) \
                      + weight*J0*tVol*(em_bc_r_1*Vol+em_bc_J_1)*ds(2)

            # These are evaluation forms, no test functions
            p_t0_0_Loc = -weight*J0*dot(Constant((1.0,0.0,0.0)),F*S*Ez)*ds(1)
            p_t1_0_Loc = -weight*J0*dot(Constant((0.0,1.0,0.0)),F*S*Ez)*ds(1)
            p_t2_0_Loc = -weight*J0*dot(Constant((0.0,0.0,1.0)),F*S*Ez)*ds(1)
            p_t0_1_Loc =  weight*J0*dot(Constant((1.0,0.0,0.0)),F*S*Ez)*ds(2)
            p_t1_1_Loc =  weight*J0*dot(Constant((0.0,1.0,0.0)),F*S*Ez)*ds(2)
            p_t2_1_Loc =  weight*J0*dot(Constant((0.0,0.0,1.0)),F*S*Ez)*ds(2)

            p_J_0_Loc = -weight*J0*dot(ey,em_J)*ds(1)
            p_J_1_Loc =  weight*J0*dot(ey,em_J)*ds(2)

            f_I_Loc = weight*J0*em_J
            f_Pi1_Loc = weight*J0*F*S*Ez
            
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

            f_I = f_I_Loc if f_I is None else f_I + f_I_Loc
            f_Pi1 = f_Pi1_Loc if f_Pi1 is None else f_Pi1 + f_Pi1_Loc
            
        # Contact Forms
        XX = Expression(("x[0]","x[1]","x[2]"))
        xr =  XX + q
        # xr = X0 + q
        dist = sqrt(dot(jump(xr),jump(xr)))
        overlap = (2.0*avg(radius)-dist)
        Rstar = 1.0/( 1.0/radius('-') + 1.0/radius('+') )
        Estar = self.E #1.0/( 1.0/E('-') + 1.0/E('+') )
        cont_pres = contact_penalty*overlap
        a_hertz = sqrt( 4.0/np.pi * Estar/Rstar * abs(cont_pres) )
        ContactForm = -dot(jump(tvq),
                        conditional(ge(overlap,0.0), -cont_pres,0.0)*jump(xr)/dist) \
                        *dc(0, metadata={"num_cells": 2,"special":"contact"})
        ThermalContact = -dot(
            jump(tT),
            conditional(ge(overlap,0.0), contact_temp*a_hertz,0.0)*jump(T)
            )*dc(0,metadata={"num_cells": 2,"special":"contact"})
        VoltageContact = -dot(
            jump(tVol),
            conditional(ge(overlap,0.0), -contact_em*a_hertz,0.0)*jump(Vol)
            )*dc(0,metadata={"num_cells": 2,"special":"contact"})

        FMechTot += ContactForm
        FTherTot += ThermalContact
        FElecTot += VoltageContact


        stat_fields = {
            'p_t0_0':Form(p_t0_0),
            'p_t1_0':Form(p_t1_0),
            'p_t2_0':Form(p_t2_0),
            'p_t0_1':Form(p_t0_1),
            'p_t1_1':Form(p_t1_1),
            'p_t2_1':Form(p_t2_1),
            'p_J_0':Form(p_J_0),
            'p_J_1':Form(p_J_1),

            'f_I':f_I,
            'f_Pi1':f_Pi1
            }

        fem_forms = self.fem_forms( MassMech,FMechTot,MassTher,FTherTot,FElecTot, wx,wv,T,Vol, Delw,DelT,DelVol)
        fem_forms.update(stat_fields)
        return fem_forms

    def split_for_io(self):
        pass
