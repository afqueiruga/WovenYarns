from MultiphysicsBaseProblem import *
from QuadraturePoints import RectOuterProd,CircCart2D

class DecoupledProblem(MultiphysicsBaseProblem):
    def Declare_Spaces(self):
        V = VectorFunctionSpace(self.mesh,"CG",self.order[0])
        S = FunctionSpace(self.mesh,"CG",self.order[1])
        d = {
            'S': S,
            'V': V,
            'W': MixedFunctionSpace([V,V,V])
        }
        return d

    def Declare_Fields(self):
        self.space_key = {'wx':'W',
                'wv':'W',
                'T':'S',
                'Vol':'S'}
        return {'wx':Function(self.spaces['W']),
                'wv':Function(self.spaces['W']),
                'T' :Function(self.spaces['S']),
                'Vol':Function(self.spaces['S'])}

    def fields_and_tests(self):
        # Set up the functions, trials, and tests.
        W = self.spaces['W']
        space_S = self.spaces['S']
        space_V = self.spaces['V']
        
        wx = self.fields['wx']
        wv = self.fields['wv']
        T = self.fields['T']
        Vol = self.fields['Vol']
        
        vq,vh1,vh2 = split(wv)
        q,h1,h2 = split(wx)

        tw = TestFunction(W)
        tvq,tvh1,tvh2 = split(tw)
        tT = TestFunction(space_S)
        tVol = TestFunction(space_S)
        
        Delw = TrialFunction(W)
        DelT = TrialFunction(space_S)
        DelVol = TrialFunction(space_S)

        dvdtW = TrialFunction(W)
        dvqdt,dvh1dt,dvh2dt = split(dvdtW)
        dTdt = TrialFunction(space_S)
        
        return wx, q,h1,h2, wv, vq,vh1,vh2, T, Vol, \
          tw, tvq,tvh1,tvh2, tT,tVol, \
          Delw,DelT,DelVol, \
          dvdtW,dvqdt,dvh1dt,dvh2dt,dTdt
          
    def fem_forms(self, MassMech,FMechTot,MassTher,FTherTot,FElecTot, wx,wv,T,Vol, Delw,DelT,DelVol):
        # Functional derivatives
        AX = derivative(FMechTot,wx,Delw)
        AV = derivative(FMechTot,wv,Delw)
        AT = derivative(FTherTot,T,DelT)
        AE = derivative(FElecTot,Vol,DelVol)

        return {
            'F' : Form(FMechTot),
            'AX': Form(AX),
            'AV': Form(AV),
            'M' : Form(MassMech),
            'FT': Form(FTherTot),
            'AT': Form(AT),
            'MT': Form(MassTher),
            'FE': Form(FElecTot),
            'AE': Form(AE)
            }
    
    def split_for_io(self):
        q,h1,h2,     = self.fields['wx'].split()
        vq,vh1,vh2 = self.fields['wv'].split()
        
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
                'T':self.fields['T'], 'Vol':self.fields['Vol'],
                'I':I, 'Pi1':Pi1
            }
