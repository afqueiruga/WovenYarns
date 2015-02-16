#!/usr/bin/python

from src import *
from matplotlib import pylab as plt
from scipy import optimize as scop
from IPython import embed

"""
Properties
"""

Tmax = 2.0
L = 1.0
Delta = 0.5*L
VBound = (L-Delta)/Tmax
endpts = [ [-L,0.0,0.0],[L,0.0,0.0] ]

E = 10.0
nu = 0.0
B = 1.0
J = 1.0
radius = 0.02

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

def solveit(Nelem,p, Rad):
   
    
    props =  {
    'mu':E/(2*(1 + nu)),
    'lambda': E*nu/((1 + nu)*(1 - 2*nu)),
    'radius':Rad,
    'em_B':Constant((B*np.cos(Phi),B*np.sin(Phi),0.0)),
    'em_I':-J,
    'dissipation':2.5}

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
    fib.problem.fields['wx'].interpolate(Expression((
        "x[0]*sq",
"0",
"A1*cos(x[0]*p)",
"0",
"-1 + (A1*pow(p,2)*(1 + sq)*cos(x[0]*p))/sqrt(pow(A1,2)*pow(p,4)*pow(1 + sq,2)*pow(cos(x[0]*p),2))",
"0",
"(pow(A1,2)*pow(p,3)*(1 + sq)*sin(2*x[0]*p))/(2.*sqrt(pow(A1,2)*pow(p,4)*pow(1 + sq,2)*pow(cos(x[0]*p),2))*sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(sin(x[0]*p),2)))",
"0",
"(A1*pow(p,2)*pow(1 + sq,2)*cos(x[0]*p) - sqrt(pow(A1,2)*pow(p,4)*pow(1 + sq,2)*pow(cos(x[0]*p),2))*sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(sin(x[0]*p),2)))/(sqrt(pow(A1,2)*pow(p,4)*pow(1 + sq,2)*pow(cos(x[0]*p),2))*sqrt(pow(1 + sq,2) + pow(A1,2)*pow(p,2)*pow(sin(x[0]*p),2)))"
),
sq=0.0,A1=0.8,p=0.5*np.pi/L))
        # "x[0]*D","0.0","0.5*cos(-x[0]*period)", #"0.1*cos(-x[0]*period)","0.1*sin(x[0]*period)",
        #                                "0.0"," 0.0","0.0",
        #                                "0.0","0.0","0.0"),period=0.5*np.pi/L,D=-(L-Delta)/L))
    fib.problem.fields['wv'].interpolate(Expression(("x[0]*V","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0"),V=-VBound/L,period=0.5*np.pi/L))
    
    fib.WriteFile("post/coil/wire_test_"+str(0)+".pvd")
    fib.WriteSurface("post/coil/wire_mesh_test_"+str(0)+".pvd")

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
    NT = 25
    h = Tmax/NT
    dirk = DIRK_Monolithic(h,LDIRK[1], sys, lambda : None,apply_BCs,
                       fib.problem.fields['wx'].vector(),fib.problem.fields['wv'].vector(),
                       assemble(fib.problem.forms['M']))
    for t in xrange(NT):
        dirk.march()
        fib.WriteFile("post/coil/wire_test_"+str(t+1)+".pvd")
        fib.WriteSurface("post/coil/wire_mesh_test_"+str(t+1)+".pvd")


    # And the do a newton iteration at the end
    DelW = Function(fib.problem.spaces['W'])
    fib.problem.fields['wv'].interpolate(Expression(("0.0","0.0","0.0",
                                       "0.0"," 0.0","0.0",
                                       "0.0","0.0","0.0")))
    eps = 1.0
    tol = 1.0e-11
    maxiter = 20
    itcnt = 0
    while eps>tol and itcnt < maxiter:
        R,AX,AV=sys(0.0)
        apply_BCs(AX,R,0.0,True)
        solve(AX,DelW.vector(),R)
        fib.problem.fields['wx'].vector()[:] -= DelW.vector()[:]
        eps=np.linalg.norm(DelW.vector().array(), ord=np.Inf)
        print "  ",itcnt," Norm:", eps

        fib.WriteFile("post/coil/wire_test_"+str(t+itcnt+2)+".pvd")
        fib.WriteSurface("post/coil/wire_mesh_test_"+str(t+itcnt+2)+".pvd")
        itcnt += 1
    
    
    weval = np.zeros(9)
    fib.problem.fields['wx'].eval(weval,np.array([0.0,0.0,0.0],dtype=np.double))
    # print weval
    # print fib.problem.fields['wx'].compute_vertex_values()
    fib.WriteFile("post/coil/wire_done_"+str(p)+"_"+str(Nelem)+".pvd")
    fib.WriteSurface("post/wire_mesh_done_"+str(Rad)+".pvd")
    return weval[2]
solveit(25,2,radius)
exit()
# RADS = np.linspace(0.005,0.01,10) #[ 0.02, 0.01, 0.005 ] #, 0.001 ] #, 0.0001, 0.00001 ]
Nelems = [[ ]]
Nelems = [[100,150,200,250,300,350,400,450] ,
          [25,30,35,40,45,50,55,60],
          [6,8,10,12,14 ],#, 15,20,25,30,35,40],
          [4,6,8,10, 35]] #,15,25,30,35] ]
ps = [ 1,2,3,4 ] #,2,3,4 ]
# points = []
# for NS,p in zip(Nelems,ps):
#     points.append( map(lambda x:solveit(x,p,radius),NS) )
#     print points
# print points
points = [[0.71450240790324038, 0.71426425051315212, 0.71418036030742982, 0.71414143470542768, 0.71412026408309592, 0.71410749002193785, 0.71409919559305846, 0.71409350733195531],
          [0.71409255197806032, 0.71408389443911324, 0.71407811556602052, 0.71407612851948976, 0.71407442889027961, 0.71407380959542643, 0.71407316925195918, 0.71407293499895552], 
          [0.71429410619736811, 0.71414614513931729, 0.7141007485729457, 0.71408442098228808, 0.71407786341532009],#, 0.71407192094934024, 0.71407295364028001, 0.71407193822666348, 0.71407216751662395, 0.7140720318354028, 0.71407209678614281], 
          [0.71419439590525435, 0.71408873925240124, 0.71407502516226495,0.71407268295160187 #, 0.71407208150236023, 0.71407208030437042  , 0.71407208007532452, 0.71407208065841965]]
           , 0.71407208070930606]
          ]

bestsol = points[-1][-1]

def make_plots(Nelems,points):
    plt.xlabel("log(h)")
    plt.ylabel("log(e)")
    for p,NS,pts in zip(ps,Nelems,points):
        plt.loglog([L/x for x in NS],
                   [np.abs( (y - bestsol)) for y in pts]
                   ,'+-',label='p='+str(p))
        
    plt.legend()
    plt.figure()
    plt.xlabel("N")
    plt.ylabel("y(0)")
    for p,NS,pts in zip(ps,Nelems,points):
        plt.plot([x for x in NS],
                 [y for y in pts]
                 ,'+-',label='p='+str(p))
        plt.axhline(truesol)
    plt.legend()
    plt.show()


def compute_convergence(Nelems,points):
    import scipy.stats
    best = points[-1][-1]
     
    for ix,(NS,pts) in enumerate(zip(Nelems,points)):
        hs = [L/x for x in NS]
        print (scipy.stats.linregress([ np.log(x) for x in hs[:(-1 if ix==len(points)-1 else -2)] ],
                                      [ np.log(np.abs((y-best))) for y in pts[:(-1 if ix==len(points)-1 else -2)] ]))[0],
        print ""
print omega
print sol(0.0)
print sol(-L)
print sol(L)
print truesol

compute_convergence(Nelems, points)
make_plots(Nelems,points)

# SOME DATA: [0.7115796857321488, 0.71156571173037519, 0.71156431018881172, 0.71156399897569222, 0.71156387764328732, 0.7115638328981837, 0.71156380904687244, 0.711563798320615]


# THIS IS ALL SHIT! FORGOT TO SET V=0
# SOME BETTER DATA
#[[0.57747890857010731, 0.57759131049374468, 0.57762868427088676, 0.57764414489974003, 0.57765183458822944, 0.57765618635332494, 0.57765888515427111, 0.57766067498959772], [0.57731632848794778, 0.57748931375381762, 0.57756670562052914, 0.57760686026026664, 0.57762840825033179, 0.57764138837356382, 0.57764916623764728, 0.57765433271371636], [0.57758358849675517, 0.57765237173456607, 0.5776625602988289, 0.57766551139662337, 0.57766628932920583, 0.5776666715576011], [0.57763223991411661, 0.57766541293237839, 0.57766671125291003, 0.57766689409655347, 0.57766692802464958, 0.57766693566394089]]

# REALLY GOOD DATA
#p=1: [[0.71251339335081798, 0.71211348740788705, 0.71196530625502796, 0.71189416428249241, 0.7118544915025935, 0.71183009333019376, 0.71181401530239252, 0.71180286369002277]]
# p=2: [[0.7126949366314308, 0.71267261207215715, 0.71265019053243805, 0.712639090167831, 0.71263144227180031, 0.71262107540323005, 0.71261431581206303, 0.71260839246529595]]

# BETTER DATA
#[[0.71450240790324038, 0.71426425051315212, 0.71418036030742982, 0.71414143470542768, 0.71412026408309592, 0.71410749002193785, 0.71409919559305846, 0.71409350733195531], [0.71409255197806032, 0.71408389443911324, 0.71407811556602052, 0.71407612851948976, 0.71407442889027961, 0.71407380959542643, 0.71407316925195918, 0.71407293499895552], [0.71407192094934024, 0.71407295364028001, 0.71407193822666348, 0.71407216751662395, 0.7140720318354028, 0.71407209678614281], [0.71419439590525435, 0.71408873925240124, 0.71407502516226495,0.71407268295160187, 0.71407208150236023, 0.71407208030437042, 0.71407208007532452, 0.71407208065841965, 0.71407208070930606]]
