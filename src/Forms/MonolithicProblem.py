from MultiphysicsBaseProblem import *
from QuadraturePoints import RectOuterProd,CircCart2D

class MonolithicProblem(MultiphysicsBaseProblem):
    def Declare_Spaces(self):
        V = VectorFunctionSpace(self.mesh,"CG",self.order[0])
        S = FunctionSpace(self.mesh,"CG",self.order[1])
        d = {
            'S': S,
            'V': V,
            'W': MixedFunctionSpace([V,V,V,S,S])
        }
        return d

    def Declare_Fields(self):
        self.space_key = {'wx':'W',
                'wv':'W'}
        return {'wx':Function(self.spaces['W']),
                'wv':Function(self.spaces['W'])}
    
    def fields_and_tests(self):
        # Set up the functions, trials, and tests.
        W = self.spaces['W']
        wx = self.fields['wx']
        wv = self.fields['wv']
        
        vq,vh1,vh2,T,Vol = split(wv)
        q,h1,h2,Tnull,Vnull = split(wx)

        tw = TestFunction(W)
        tvq,tvh1,tvh2, tT, tVol = split(tw)
        Delw = TrialFunction(W)
        Delq,Delh1,Delh2,DelT,DelVol = split(Delw)
        
        dvdtW = TrialFunction(W)
        dvqdt,dvh1dt,dvh2dt, dTdt, dVdtNull = split(dvdtW)
        return wx, q,h1,h2, wv, vq,vh1,vh2, T, Vol, \
          tw, tvq,tvh1,tvh2, tT,tVol, \
          Delw,DelT,DelVol, \
          dvdtW,dvqdt,dvh1dt,dvh2dt,dTdt
          
    def fem_forms(self, MassMech,FMechTot,MassTher,FTherTot,FElecTot, 
                        wx,wv,T,Vol, Delw,DelT,DelVol):
        FTot = FMechTot + FTherTot + FElecTot
        Mass = MassMech + MassTher
        AX = derivative(FTot,wx,Delw)
        AV = derivative(FTot,wv,Delw)
        return {
            'F':Form(FTot),
            'AX':Form(AX),
            'AV': Form(AV),
            'M' : Form(Mass)
            }

    def split_for_io(self):
        q,h1,h2, Tnull, Vnull    = self.fields['wx'].split()
        vq,vh1,vh2,T,Vol = self.fields['wv'].split()
        
        g1 = project(self.properties['radius']*(self.properties['E1']+h1),
                     self.spaces['V'])
        g2 = project(self.properties['radius']*(self.properties['E2']+h2),
                     self.spaces['V'])
        I = project(self.forms['f_I'], self.spaces['V'])
        Pi1 = project(self.forms['f_Pi1'], self.spaces['V'])
        
        return {
                'q':q, 'h1':h1, 'h2':h2,
                'vq':vq, 'vh1':vh1, 'vh2':vh2,
                'g1':g1, 'g2':g2,
                'T':T, 'Vol':Vol,
                'I':I, 'Pi1':Pi1
            }
