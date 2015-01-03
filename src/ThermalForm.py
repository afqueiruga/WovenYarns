"""

A purely thermal form for the beam element.

"""

from dolfin import *

def ThermalForm(S,T,X0):
    dTdt = TrialFunction(S)
    dT = TestFunction(S)
    DelT = TrialFunction(S)

    CrossA = 0.1
    thermalCond = Constant(1.0)
    rho = Constant(1.0)
    Mass = inner(dT, rho*dTdt)*dx
    AT = inner( dT.dx(0), thermalCond * DelT.dx(0) ) * dx
    FT = inner( dT, 1.0)*dx 

    return Form(FT), Form(Mass), Form(AT)
