from dolfin import *



class ProblemDescription():
    """
    This is a prototype class representing a problem as
    a collection FunctionSpaces and FEniCS form, enumerating
    inputs, outputs, etc.
    """
    def __init__(self, mesh, properties):
        """"
        Properties should be a dictionary of double values or of FEM
        spaces on the mesh. If the properties is a constant, it will
        be turned into a space of type Real. 
        """
        self.mesh = mesh

        self.R = FunctionSpace(mesh,"Real")
        self.properties = {}
        for name,val in properties.iteritems():
            if val is Function:
                self.properties[name] = val
            else:
                self.properties[name] = Function(R)
                self.properties[name].interpolate(Constant(val))

        self.spaces = self.Declare_Spaces()
        self.forms  = self.Build_Forms()

    def Declare_Spaces():
        return {}
    
    def Build_Form():
        " Build the variational problem "
        return None
