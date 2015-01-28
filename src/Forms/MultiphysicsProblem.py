from ProblemDescription import *
from QuadraturePoints import RectOuterProd,CircCart2D

class MultiphysicsProblem(ProblemDescription):
    """ This creates a multiphysics problem with a monolithic space """
    def __init__(self,mesh,properties,orientation=0):
        if properties.has_key("orientation"):
            orientation = properties["orientation"]
            properties.pop("orientation",None)
        ProblemDescription.__init__(self,mesh,properties)

        if orientation == 0:
            self.properties["Ez"] = Constant((1.0,0.0,0.0))
            self.properties["E1"] = Constant((0.0,1.0,0.0))
            self.properties["E2"] = Constant((0.0,0.0,1.0))
        elif orientation == 1:
            self.properties["Ez"] = Constant((0.0,1.0,0.0))
            self.properties["E1"] = Constant((0.0,0.0,1.0))
            self.properties["E2"] = Constant((1.0,0.0,0.0))
        else:
            self.properties["Ez"] = Constant((0.0,0.0,1.0))
            self.properties["E1"] = Constant((1.0,0.0,0.0))
            self.properties["E2"] = Constant((0.0,1.0,0.0))

        
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
        return {'wx':Function(self.spaces['W']),
                'wv':Function(self.spaces['W'])}

    def Build_Forms(self):
        return {}
