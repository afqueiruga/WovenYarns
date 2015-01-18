"""

A purely thermal form for the beam element.

"""

from dolfin import *

def ThermalForm(S,T,X0):
    dTdt = TrialFunction(S)
    dT = TestFunction(S)
    DelT = TrialFunction(S)

    CrossA = 0.1
    thermalCond = Constant(3.0)
    rho = Constant(5.0)
    Mass = inner(dT, rho*dTdt)*dx

    # DistVec = X0('+')-X0('-')
    # dist = dot(DistVec,DistVec)
    # overlap = (Constant(0.095)-dist)
    # contactform = inner( jump(dT), conditional(le(dist,0.11*0.11), Constant(4.1),Constant(0.0))* jump(DelT) )*dc(0, metadata={"num_cells": 2,"special":"contact"})
    
    dist = sqrt(dot(jump(X0),jump(X0)))
    overlap = (Constant(0.095)-dist)
    contactform = inner( jump(dT), conditional(le(sqrt(dot(jump(X0),jump(X0))),0.11), Constant(40.1),Constant(0.0))* jump(DelT) )*dc(0, metadata={"num_cells": 2,"special":"contact"})

    # contactform = inner( jump(dT), Constant(40.0)* jump(DelT) )*dc(0, metadata={"num_cells": 2,"special":"contact"})
    
    AT = inner( grad(dT), thermalCond * grad(DelT) )* dx \
      + contactform
      
    FT = inner( dT, Constant(0.0))*dx 

    return Form(FT), Form(Mass), Form(AT)
