from dolfin import *
import numpy as np

from src.DIRK import *
from src import Warp
from src.Forms import MultiphysicsProblem
# TODO: Pass in M
# Make a warp data structure and march it one step

endpts = [ [[0.0,0.0,0.0],[1.0,0.0,0.0]] ,
           [[0.0,0.1,0.0],[1.0,0.1,0.0]] ,
           [[0.0,0.4,0.0],[1.0,0.4,0.0]] ]

defaults = { 'mu' : 10.0,
             'radius':0.2 }

props = [ { 'radius' : 0.1 },
            { 'mu' : 1.0 },
            {} ]

warp = Warp(endpts, props, defaults, MultiphysicsProblem)

warp.update()

def sys(time):
    return warp.assemble_forms(['F','AX','AV'],'W')

def bcapp(K,R,time,hold):
    print "Meh. Save this for an engineering test", time
    pass

dirk = DIRK_Monolithic(0.1,LDIRK[2], sys,warp.update,bcapp,
                       warp.fields['wx'].vector(),warp.fields['wv'].vector(),
                       warp.assemble_form('M','W'))
dirk.march()
