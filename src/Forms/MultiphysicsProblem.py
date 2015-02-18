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
    'dissipation':1.0e-1
}
class MultiphysicsProblem(ProblemDescription):
    """ This creates a multiphysics problem with a monolithic space """
    def __init__(self,mesh,properties, buildform=True, orientation=0, order=(1,1)):
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
        
        ProblemDescription.__init__(self,mesh,p,False)
        
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
            'W': MixedFunctionSpace([V,V,V,S,S])
        }
        return d

    def Declare_Fields(self):
        self.space_key = {'wx':'W',
                'wv':'W'}
        return {'wx':Function(self.spaces['W']),
                'wv':Function(self.spaces['W'])}

    def Build_Forms(self):
        # Set up the functions, trials, and tests.
        W = self.spaces['W']
        wx = self.fields['wx']
        wv = self.fields['wv']
        
        vq,vh1,vh2,T,Vol = split(wv)
        q,h1,h2,Tnull,Vnull = split(wx)

        tw = TestFunction(W)
        tvq,tvh1,tvh2, tT, tVol = split(tw)
        Delw = TrialFunction(W)

        dvdtW = TrialFunction(W)
        dvqdt,dvh1dt,dvh2dt, dTdt, dVdtNull = split(dvdtW)

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
        
        orientation = self.orientation
        Ez = PROP['Ez']
        E1 = PROP['E1']
        E2 = PROP['E2']

        X0 = PROP['X0']
        I = Identity(W.cell().geometric_dimension())

        # Perform the cross section integration
        FTot = None
        Mass = None
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

            # Thermal Form
            Gradv = outer(v.dx(orientation),Ez) + outer(vh1,E1) + outer(vh2,E2)
            S = (mu_pt)*I+(-mu_pt+lmbda*ln(J))*inv(C).T
            T_FLoc = -(tT.dx(orientation) * T_k * T.dx(orientation)) + tT*1.0

            # Electrical Potential
            V_FLoc = -inner(tVol.dx(orientation), em_sig*Vol.dx(orientation)) \
              - inner(tVol.dx(orientation)*Ez,em_sig*(F.T*cross(v,em_B)))
            # Current force
            em_I = em_sig*Vol.dx(orientation)
            ey = (Ez+q.dx(orientation)) /sqrt( inner(Ez+q.dx(orientation),Ez+q.dx(orientation)) )
            
            FExt = -em_I*inner(tv,cross(ey,em_B)) + inner(tv,-dissipation*v) + inner(tv, f_dens_ext)
            
            # Finalize
            FLoc = weight*J0*( FInt + FExt + T_FLoc + V_FLoc )*dx
            MLoc = weight*J0*(inner(tv,rho*dvdt)+inner(tT,rho*T_cp*dTdt))*dx
            FTot = FLoc if FTot is None else FTot + FLoc
            Mass = MLoc if Mass is None else Mass + MLoc
            
        # Contact forms
        xr = X0 + q
        dist = sqrt(dot(jump(xr),jump(xr)))
        overlap = (2.0*Constant(radius)-dist)
        ContactForm = -dot(jump(tvq),
                        conditional(ge(overlap,0.0), -40.0*overlap,0.0)*jump(xr)/dist)*dc(0, metadata={"num_cells": 2,"special":"contact"})

        
        # Take the functional derivatives of everything
        FTot += ContactForm
        AX = derivative(FTot,wx,Delw)
        AV = derivative(FTot,wv,Delw)
        
        # Dictionize and return
        return {
            'F' : Form(FTot),
            'AX': Form(AX),
            'AV': Form(AV),
            'M' : Form(Mass)
            }

    def split_for_io(self):
        q,h1,h2, Tnull, Vnull    = self.fields['wx'].split()
        vq,vh1,vh2,T,Vol = self.fields['wv'].split()
        
        g1 = project(self.properties['radius']*self.properties['E1']+h1,
                     self.spaces['V'])
        g2 = project(self.properties['radius']*self.properties['E2']+h2,
                     self.spaces['V'])     
        return {
                'q':q, 'h1':h1, 'h2':h2,
                'vq':vq, 'vh1':vh1, 'vh2':vh2,
                'g1':g1, 'g2':g2,
                'T':T, 'V':Vol
            }
