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
    
