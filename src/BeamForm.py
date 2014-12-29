"""

The form for the beam element.

"""

from dolfin import *

def BeamForm(W,V,wx,wv):
    # wx = Function(W)
    r,g1,g2  = split(wx)
    # wv = Function(W)
    vr,vg1,vg2 = split(wv)

    dvdtW = TrialFunction(W)
    dvrdt,dvg1dt,dvg2dt = split(dvdtW)

    dw = TestFunction(W)
    Delw = TrialFunction(W)
    Deldwdt = TrialFunction(W)
    dvr,dvg1,dvg2 = split(dw)
    
    # Material and geometry parameters
    E, nu = 5.0, 0.4
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
    CrossA = 0.1
    Height = 1.0
    Width = 0.5
    rho = 1.0
    
    GPS = [ [ 0.0,           0.0,           8.0/9.0 ],
            [ sqrt(3.0/5.0), sqrt(3.0/5.0), 5.0/9.0 ],
            [-sqrt(3.0/5.0), sqrt(3.0/5.0), 5.0/9.0 ],
            [-sqrt(3.0/5.0),-sqrt(3.0/5.0), 5.0/9.0 ],
            [ sqrt(3.0/5.0),-sqrt(3.0/5.0), 5.0/9.0 ] ]
    
    #
    # Build the form by integrating over the cross section
    #
    Psi = None #Constant(0.0)
    Mass = None #Constant(0.0)
    # Velocity = None #Constant(0.0)
    FExt = None #Constant(0.0)
    for z1,z2,weight in GPS:
        u = r + Height/2.0*Constant(z1)*g1 + Width/2.0*Constant(z2)*g2
        v = vr + Height/2.0*Constant(z1)*vg1 + Width/2.0*Constant(z2)*vg2
        
        # dxdt = drdt + Height/2.0*Constant(z1)*dg1dt + Width/2.0*Constant(z1)*dg2dt
        dvdt = dvrdt + Height/2.0*Constant(z1)*dvg1dt + Width/2.0*Constant(z2)*dvg2dt
        du = derivative(u,wx,dw)
        dv = derivative(v,wv,dw)
        
        I = Identity(V.cell().geometric_dimension())
        F = I + outer(u.dx(0),Constant((1.0,0.0,0.0))) + outer(g1,Constant((0.0,1.0,0.0))) + outer(g2,Constant( (0.0,0.0,1.0) ))
        C = F.T*F
        Ic = tr(C)
        J = det(F)
        
        MassCont = inner(dv,rho*dvdt)
        PsiCont = Constant(weight)*((mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2)
        # VelocityCont = inner(du,v)
        FExtCont = inner(dv,Constant((0.00,0.0,0.0)))
        FExtCont += inner(dv,-0.000001*v)
        
        # Psi += PsiCont
        # Mass += MassCont
        # Velocity += VelocityCont
        # FExt += FExtCont
        # construct = lambda f,fcont:
        Psi = PsiCont if Psi is None else Psi + PsiCont
        Mass = MassCont if Mass is None else Mass + MassCont
        FExt = FExtCont if FExt is None else FExt + FExtCont
        # Velocity = VelocityCont if Velocity is None else Velocity + VelocityCont
    xr = SpatialCoordinate(W.mesh())
    ContactForm = dot(jump(dvr),(Constant(0.0)-dot(jump(r),jump(r)))*jump(r))*dc(0, metadata={"num_cells": 2,"special":"contact"})

    
    Fform = -derivative(Psi,wx,dw)*dx + FExt*dx #+ ContactForm#- Mass*dx
    Mform = Mass*dx #derivative(F,dwdt,Deldwdt)
    AXform = derivative(Fform,wx,Delw)
    AVform = derivative(Fform,wv,Delw)
    
    return Form(Fform),Form(Mform),Form(AXform),Form(AVform)
