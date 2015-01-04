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
    dist = sqrt(dot(jump(X0),jump(X0)))
    overlap = (Constant(0.09)-dist)
    contactform = inner( jump(dT), conditional(le(overlap,0.0), Constant(4.1),Constant(0.0))* jump(DelT) )*dc(0, metadata={"num_cells": 2,"special":"contact"})
    AT = inner( dT.dx(0), thermalCond * DelT.dx(0) )* dx \
      + contactform
      
    FT = inner( dT, Constant(0.0))*dx 

    return Form(FT), Form(Mass), Form(AT)
