import numpy as np

from src import ContactGroup
from src.ProximityTree import create_line as create_line

meshes = [ 
    create_line(np.array([0.0,0.0,0.0]),np.array([1.0,0.0,0.0]), 10),
    create_line(np.array([0.0,0.1,0.0]),np.array([1.0,0.1,0.0]), 10),
    create_line(np.array([0.0,0.2,0.0]),np.array([1.0,0.2,0.0]), 10)
    ]

cg = ContactGroup.ContactGroup(meshes)

from IPython import embed
embed()
