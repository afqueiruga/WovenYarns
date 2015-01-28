from ProblemDescription import *
from QuadraturePoints import RectOuterProd,CircCart2D

class MultiphysicsProblem(ProblemDescription):
    """ This creates a multiphysics problem with a monolithic space """
    def Declare_Spaces(self):
        V = VectorFunctionSpace(self.mesh,"CG",1)
        S = FunctionSpace(self.mesh,"CG",1)
        d = {
            'S': S,
            'V': V,
            'W': MixedFunctionSpace([V,V,V,S,S])
        }
        return d

    def Declare_Fields(self):
        return {}

    def Build_Forms(self):
        return {}
