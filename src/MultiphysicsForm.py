"""

The form for the beam element with thermal and EM problems in it.

"""

from dolfin import *

def MultiphysicsForm(W,V,S,wx,wv,X0):
    vr,vg1,vg2,T,Vol = split(wv)
    r,g1,g2,Tnull,Vnull = split(wx)

    dw = TestFunction(W)
    Delw = TrialFunction(W)

    dvdtW = TrialFunction(W)
    dvrdt,dvg1dt,dvg2dt, dTdt, dVdtNull = split(dvdtW)

    #
    # Material Properties
    #
    E, nu = 50.0, 0.0
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
    mu_alpha = Constant(-0.03)
    rho = Constant(1.0)
    thermalcond = Constant(1.0)
    Height = 2.5
    Width = 1.0

    em_B = Constant((0.1,0.0,1.0))
    # em_I = Constant(1.0/Height/Width)
    em_sig = Constant(10.0)

    E1,E2,E3 = \
        Constant((1.0,0.0,0.0)),Constant((0.0,1.0,0.0)),Constant((0.0,0.0,1.0))

    #
    # Gauss points
    #
    # GPS1D = [ [ 0.0,       8.0/9.0 ],
    #           [ sqrt(3.0/5.0), 5.0/9.0 ],
    #           [-sqrt(3.0/5.0), 5.0/9.0 ] ]
    GPS1D = [ [ sqrt(1.0/3.0), 1.0],
              [-sqrt(1.0/3.0), 1.0] ]
    GPS2D = []
    for z1,w1 in GPS1D:
        for z2,w2 in GPS1D:
            GPS2D.append([z1,z2,w1*w2])

    
    Psi = None
    FExt = None
    FTot = None
    AXTot = None
    AVTot = None
    Mass = None
    ThermalMass = None

    J0 = Constant(Width*Height/4.0 )
    for z1,z2,weight in GPS:
        u = r + Height/2.0*Constant(z1)*g1 + Width/2.0*Constant(z2)*g2
        v = vr + Height/2.0*Constant(z1)*vg1 + Width/2.0*Constant(z2)*vg2
        
        # dxdt = drdt + Height/2.0*Constant(z1)*dg1dt + Width/2.0*Constant(z1)*dg2dt
        dvdt = dvrdt + Height/2.0*Constant(z1)*dvg1dt + Width/2.0*Constant(z2)*dvg2dt
        du = derivative(u,wx,dw)
        dv = derivative(v,wv,dw)
        dT = derivative(T,wv,dw)
        dVol = derivative(Vol,wv,dw)
        
        I = Identity(V.cell().geometric_dimension())
        F = I + outer(u.dx(0),Constant((1.0,0.0,0.0))) + outer(g1,Constant((0.0,1.0,0.0))) + outer(g2,Constant( (0.0,0.0,1.0) ))
        C = F.T*F
        Ic = tr(C)
        J = det(F)
        
        MassCont = inner(dv,rho*dvdt)
        PsiCont = Constant(weight)*((mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2)

        # Thermal Form
        Gradv = outer(v.dx(2),E3) + outer(vg1,E1) + outer(vg2,E2)
        S = (mu_pt)*I+(-mu_pt+lmbda*ln(J))*inv(C).T
        ThermalFLoc = -weight*(dT.dx(2) * thermalcond * T.dx(2))*dx + weight*dT*1.0*dx

        # Electrical Potential
        VoltageFLoc = weight*( -inner(dVol.dx(2), em_sig*Vol.dx(2)) - inner(dVol.dx(2)*E3,em_sig*(F.T*cross(v,em_B))) )*dx
        # Current force
        em_I = em_sig*Vol.dx(2)
        ey = (E3+r.dx(2)) /sqrt( inner(E3+r.dx(2),E3+r.dx(2)))

        FLocCont = weight*derivative(-PsiCont,wx,dw)*dx+weight*FExtCont*dx+weight*inner(dv,-1.0e-2*v)*dx + ThermalFLoc + VoltageFLoc
        
        # Add up the forms
        Psi = PsiCont if Psi is None else Psi + PsiCont
        Mass = MassCont if Mass is None else Mass + MassCont
        FExt = FLocCont if FExt is None else FExt + FLocCont
        # Velocity = VelocityCont if Velocity is None else Velocity + VelocityCont

    # Thermal mass doesn't need to be integrated
    ThermalMass = inner(dT,rho*dTdt)*dx

    # Contact forms
    xr = X0 + r
    dist = sqrt(dot(jump(xr),jump(xr)))
    overlap = (Constant(0.05)-dist)
    ContactForm = -dot(jump(dvr),
                      conditional(ge(overlap,0.0),-200.0*overlap,0.0)*jump(xr)/dist)*dc(0, metadata={"num_cells": 2,"special":"contact"})

    # Finalize and make derivatives
    Fform = -derivative(Psi,wx,dw)*dx + FExt*dx + ContactForm #- Mass*dx
    Mform = Mass*dx + ThermalMass
    AXform = derivative(Fform,wx,Delw)
    AVform = derivative(Fform,wv,Delw)
    return Form(Fform),Form(Mform),Form(AXform),Form(AVform)
