from ProblemDescription import *
from QuadraturePoints import RectOuterProd,CircCart2D

class MultiphysicsProblem(ProblemDescription):
    """ This creates a multiphysics problem with a monolithic space """
    def Declare_Spaces(self):
        V = VectorFunctionSpace(mesh,"CG",1)
        S = FunctionSpace(mesh,"CG",1)
        return ProblemDescription.Declare_Spaces().update({
            'S': S,
            'V': V,
            'W': MixedFunctionSpace([self.V,self.V,self.V,self.S,self.S])
        })

    def Declare_Fields(self):
        return {}

    def Build_Forms(self):
        return {}
