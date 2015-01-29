from dolfin import *

class ProblemDescription():
    """
    This is a prototype class representing a problem as
    a collection FunctionSpaces and FEniCS form, enumerating
    inputs, outputs, etc.
    """
    def __init__(self, mesh, properties, buildform = True):
        """"
        Properties should be a dictionary of double values or of FEM
        spaces on the mesh. If the properties is a constant, it will
        be turned into a space of type Real. 
        """
        self.mesh = mesh
        self.spaces = self.Declare_Spaces()
        if not self.spaces.has_key("R"):
            self.spaces["R"] = FunctionSpace(self.mesh,"Real",0)
        self.properties = {}
        for name,val in properties.iteritems():
            if isinstance(val, Coefficient):
                self.properties[name] = val
            else:
                self.properties[name] = Function(self.spaces['R'])
                self.properties[name].interpolate(Constant(val))

        self.fields = self.Declare_Fields()
        if buildform:
            self.forms = self.Build_Forms()

    def Declare_Spaces(self):
        " Tabulate all function spaces used. By default need the space of real "
        return {"R" : FunctionSpace(self.mesh,"Real",0)}
    
    def Declare_Fields(self):
        " Tabulate all of the functions that are to be solved. "
        return {}
    
    def Build_Forms(self):
        " Build the variational problem "
        return {}
