#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from scipy import optimize as scop
from IPython import embed

"""
Properties
"""

Tmax = 100.0
L = 1.0
Delta = 0.5*L
VBound = (L-Delta)/Tmax
endpts = [ [-L,0.0,0.0],[L,0.0,0.0] ]

E = 10.0
nu = 0.0
B = 1.0
J = 0.001
radius = 0.002

k = J*B/(E) # Using current density
Phi = np.pi/4.0
P = lambda x: np.sin(x*L)*np.sqrt( 1.0/((x-k)**2.0) - ((Delta*np.sin(Phi))**2.0)/((x*L)**2.0) ) - Delta*np.cos(Phi)
alpha = Delta*np.sin(Phi)

# oo =  np.linspace(0.0,5.0*np.pi,100)
# plt.plot(oo,P( oo ),label='$\Delta$ = {0}'.format(Delta))
# plt.grid()
# plt.show()

# Find all of the roots CHECK TOLERANCE!!!!!!!!!! Nope that wasn't it
omega=scop.fsolve(P, np.pi/2.0)[0] #np.arange(np.pi,3.0*np.pi,np.pi))

ranal = np.sqrt( 1.0/(omega-k)**2 - (Delta*np.cos(Phi))**2/omega**2)

sol = lambda s: np.array([alpha*s,  ranal*np.sin(omega*s), -ranal * np.cos(omega)+ranal*np.cos(omega*s),])
truesol = sol(0.0)[2]
print omega
print sol(0.0)
print sol(-L)
print sol(L)
print truesol
print omega/(omega-k)

exit()
def solveit(Nelem,p, Rad):
   
    
    props =  {
    'mu':E/(2*(1 + nu)),
    'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
    'radius':Rad,
    'em_B':Constant((B*np.cos(Phi),B*np.sin(Phi),0.0)),
    'em_I':-J }

    fib = Fibril.Fibril( endpts, Nelem, props,
                        CurrentBeamProblem,order=(p,1))

    """
    Boundary conditions on the velocity updates
    """
    zero = Constant((0.0,0.0,0.0)) #, 0.0,0.0,0.0, 0.0,0.0,0.0))
    velocity = Expression(("x[0]*V","0.0","0.0"),V=-VBound/L)
    
    bound = CompiledSubDomain("on_boundary")
    bczero = DirichletBC(fib.problem.spaces['W'].sub(0), zero, bound)
    bcmove = DirichletBC(fib.problem.spaces['W'].sub(0), velocity, bound)
    fib.problem.fields['wx'].interpolate(Expression(("0.0","0.0","0.0",#"cos(x[0]*period)","sin(-x[0]*period)",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0"),period=np.pi/10.0))
    fib.problem.fields['wv'].interpolate(Expression(("x[0]*V","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0"),V=-VBound/L))
    """
    DIRK's assembling routine
    """
    
    
    def sys(time):
        R = assemble(fib.problem.forms['F'])
        K = assemble(fib.problem.forms['AX'])
        AV = assemble(fib.problem.forms['AV'])
        return R,K,AV
    def apply_BCs(K,R,time,hold):
        bczero.apply(K,R)
        # if hold:
        #     bczero.apply(K,R)
        # else:
        #     bcmove.apply(K,R)

    # Do a bit of dynamic relaxation first
    NT = 400
    h = Tmax/NT
    dirk = DIRK_Monolithic(h,LDIRK[1], sys, lambda : None,apply_BCs,
                       fib.problem.fields['wx'].vector(),fib.problem.fields['wv'].vector(),
                       assemble(fib.problem.forms['M']))
    for t in xrange(NT):
        dirk.march()
        # fib.WriteFile("post/wire_test_"+str(t)+".pvd")
        # fib.WriteSurface("post/wire_mesh_test_"+str(t)+".pvd")


    # And the do a newton iteration at the end
    DelW = Function(fib.problem.spaces['W'])

    eps = 1.0
    tol = 1.0e-11
    maxiter = 10
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        R,AX,AV=sys(0.0)
        apply_BCs(AX,R,0.0,True)
        solve(AX,DelW.vector(),R)
        fib.problem.fields['wx'].vector()[:] -= DelW.vector()[:]
        eps=np.linalg.norm(DelW.vector().array(), ord=np.Inf)
        print "  ",itcnt," Norm:", eps

        # fib.WriteFile("post/wire_test_"+str(t+itcnt+1)+".pvd")
        # fib.WriteSurface("post/wire_mesh_test_"+str(t+itcnt+1)+".pvd")
        itcnt += 1
    
    
    weval = np.zeros(9)
    fib.problem.fields['wx'].eval(weval,np.array([0.0,0.0,0.0],dtype=np.double))
    # print weval
    # print fib.problem.fields['wx'].compute_vertex_values()
    fib.WriteFile("post/wire_done_"+str(Rad)+".pvd")
    fib.WriteSurface("post/wire_mesh_done_"+str(Rad)+".pvd")
    return weval[2]

# RADS = np.linspace(0.005,0.01,10) #[ 0.02, 0.01, 0.005 ] #, 0.001 ] #, 0.0001, 0.00001 ]
# points = map(lambda x:solveit(100,1,x), RADS)
# print points
# print truesol
# plt.plot([x for x in RADS], [y for y in points] ,'+-')
Nelems = [100] #range(25,201,25) #[4,6,8,10,12,14,16] #,18,20] #,30,40,50] #,100 ,200] #,300,400,600] #,800,1000]
ps = [ 2 ]
for p in ps:
    points = map(lambda x:solveit(x,p,0.005),Nelems)
    print points
    # plt.plot([L/x for x in Nelems], [y for y in points]
    #           ,'+-',label='p='+str(p))
    plt.loglog([2.0*L/x for x in Nelems],
               [np.abs( (y - truesol)) for y in points]
               ,'+-',label='p='+str(p))
print omega
print sol(0.0)
print sol(-L)
print sol(L)
print truesol
plt.show()

# SOME DATA: [0.7115796857321488, 0.71156571173037519, 0.71156431018881172, 0.71156399897569222, 0.71156387764328732, 0.7115638328981837, 0.71156380904687244, 0.711563798320615]
