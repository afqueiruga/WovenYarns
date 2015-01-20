"""

The form for the beam element with thermal and EM problems in it.
This returns three different forms with two mass matrices and four tangents:

M,F,AX,AV
MT,FT,AT
FV,AV

The forms are coupled, but the tangents are only for the fields themselves.
"""

from dolfin import *
from QuadraturePoints import RectOuterProd,CircCart2D


def IterativeForm(W,V,S,wx,wv,T,Vol,X0, orientation=0,radius=1.0):
    vr,vg1,vg2 = split(wv)
    r,g1,g2 = split(wx)

    dw = TestFunction(W)
    dvr,dvg1,dvg2 = split(dw)
    dT = TestFunction(S)
    dVol = TestFunction(S)

    Delw = TrialFunction(W)
    DelT = TrialFunction(S)
    DelV = TrialFunction(S)

    
    dvdtW = TrialFunction(W)
    dvrdt,dvg1dt,dvg2dt = split(dvdtW)
    dTdt = TrialFunction(S)

    
    #
    # Material Properties
    #
    E, nu = 1.0, 0.0
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
    mu_alpha = Constant(-0.03)
    rho = Constant(1.0)
    thermalcond = Constant(1.0)

    em_B = Constant((0.0,0.0,0.0)) #Constant((0.0,1.0,0.0))
    # em_I = Constant(1.0/Height/Width)
    em_sig = Constant(10.0)

    print orientation
    if orientation == 0:
        Ez,E1,E2 = \
        Constant((1.0,0.0,0.0)),Constant((0.0,1.0,0.0)),Constant((0.0,0.0,1.0))
    elif orientation == 1:
        Ez,E1,E2 = \
       Constant((0.0,1.0,0.0)),Constant((0.0,0.0,1.0)), Constant((1.0,0.0,0.0))
    else:
        Ez,E1,E2 = \
        Constant((0.0,0.0,1.0)), Constant((1.0,0.0,0.0)),Constant((0.0,1.0,0.0))

    #
    # Gauss points
    #
    
    GPS2D = CircCart2D[4]
    
    Psi = None
    FExt = None
    AXTot = None
    AVTot = None
    Mass = None
    
    ThermalMass = None
    FT = None

    FV = None
    
    J0 = Constant(radius*radius/4.0 )
    for z1,z2,weight in GPS2D:
        u = r + radius*Constant(z1)*g1 + radius*Constant(z2)*g2
        v = vr + radius*Constant(z1)*vg1 + radius*Constant(z2)*vg2
        
        # dxdt = drdt + Height/2.0*Constant(z1)*dg1dt + Width/2.0*Constant(z1)*dg2dt
        dvdt = dvrdt + radius*Constant(z1)*dvg1dt + radius*Constant(z2)*dvg2dt
        du = derivative(u,wx,dw)
        dv = derivative(v,wv,dw)
                
        I = Identity(V.cell().geometric_dimension())
        F = I + outer(u.dx(orientation),Ez) + outer(g1,E1) + outer(g2, E2 )
        C = F.T*F
        Ic = tr(C)
        J = det(F)
        
        MassCont = inner(dv,rho*dvdt)

        # Stored Energy at this point
        mu_pt = mu+mu_alpha*T
        PsiCont = Constant(weight)*((mu_pt/2)*(Ic - 3) - mu_pt*ln(J) + (lmbda/2)*(ln(J))**2)

        # Thermal Form
        Gradv = outer(v.dx(orientation),Ez) + outer(vg1,E1) + outer(vg2,E2)
        S = (mu_pt)*I+(-mu_pt+lmbda*ln(J))*inv(C).T
        ThermalFLoc = -weight*(dT.dx(orientation) * thermalcond * T.dx(orientation))*dx + weight*dT*1.0*dx

        # Electrical Potential
        VoltageFLoc = weight*( -inner(dVol.dx(orientation), em_sig*Vol.dx(orientation)) - inner(dVol.dx(orientation)*Ez,em_sig*(F.T*cross(v,em_B))) )*dx
        # Current force
        em_I = em_sig*Vol.dx(0)
        ey = (Ez+r.dx(orientation)) /sqrt( inner(Ez+r.dx(orientation),Ez+r.dx(orientation)) )

        FExtCont = -em_I*inner(dv,cross(ey,em_B))
        FLocCont = weight*derivative(-PsiCont,wx,dw)*dx+weight*FExtCont*dx+weight*inner(dv,-1.0e-1*v)*dx
        
        # Add up the forms
        Psi = PsiCont if Psi is None else Psi + PsiCont
        Mass = MassCont if Mass is None else Mass + MassCont
        FExt = FLocCont if FExt is None else FExt + FLocCont
        FT = ThermalFLoc if FT is None else FT + ThermalFLoc
        FV = VoltageFLoc if FV is None else FV + VoltageFLoc
        
        # Velocity = VelocityCont if Velocity is None else Velocity + VelocityCont


    # Contact forms
    xr = X0 + r
    dist = sqrt(dot(jump(xr),jump(xr)))
    overlap = (2.0*Constant(radius)-dist)
    ContactForm = -dot(jump(dvr),
                      conditional(ge(overlap,0.0), -40.0*overlap,0.0)*jump(xr)/dist)*dc(0, metadata={"num_cells": 2,"special":"contact"})

    # Mechanical forms and derivatives
    Fform =  FExt + ContactForm #-derivative(Psi,wx,dw)*dx - Mass*dx
    Mform = Mass*dx
    AXform = derivative(Fform,wx,Delw)
    AVform = derivative(Fform,wv,Delw)

    # Thermal forms 
    # Thermal mass doesn't need to be integrated
    MTform = inner(dT,rho*dTdt)*dx
    ATform = derivative(FT,T,DelT)

    # Voltage Forms
    AVolform = derivative(FV,Vol,DelV)
    
    return Form(Fform),Form(Mform),Form(AXform),Form(AVform), \
      Form(FT),Form(MTform),Form(ATform), \
      Form(FV),Form(AVolform), \
      Ez,E1,E2
