from __future__ import division
import numpy as np
import scipy.integrate as spin
from scipy import optimize
import pylab as pl
import sympy
import sys
import collections
import os

# Define the initial conditions for each of the four ODEs
inic = [5.,0.,1.,0.,0.,0.,0.,0.]

t = np.linspace(0, 20, 10000)

Modelname = 'MODEL_Halfreaction'

Parameters = {'k1':1/10,'k1m':1/20,
              'k2':1/20,'k2m':1/20,
              'k3':1/200,'k3m':1/175,
              'k4':1/200,'k4m':1/165}
              
Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))
Parameters.names = Parameters.keys()
Parameters.vals = Parameters.values()
              
System =    {'dE':'k1m*Es*P + k4*EP + k2*Es*B - k1*E*A - k4*E*P - k2m*E*Q',
             'dEs':'- k1m*Es*P + k3*EsQ - k2*Es*B + k1*E*A - k3*Es + k2m*E*Q',
             'dA':'- k1*E*A + k1m*Es*P',
             'dB':'- k2*Es*B + k2m*E*Q',
             'dP':'k1*E*A - k1m*Es*P - k4*E*P + k4m*EP',
             'dQ':'k2*E*B - k2m*E*Q - k3*Es*Q + k3m*EsQ',
             'dEsQ':'k3*Es*Q - k3m*EsQ',
             'dEP':'k4*E*P - k4m*EP'}
             
System = collections.OrderedDict(sorted(System.items(), key=lambda t: t[0]))
System.names = System.keys()
System.vals = System.values()

Measurable_States = {'E':0,'Es':0,'A':1,'B':1,'P':1,'Q':1,'EsQ':0,'EP':0}
Measurable_States = collections.OrderedDict(sorted(Measurable_States.items(), key=lambda t: t[0]))

## Automatic conversion of variables to symbols
def Analytic_local_sensitivities(System):
    '''
    Analytic derivation of the local sensitivities
    '''
    symbol_list = []
    for i in range(len(System.keys())):
        ## Deleting all spaces
        system_intern = System.values()[i].replace(" ", "")
        ## Deleting braces, only interested in symbol generation not in solution
        system_intern = system_intern.replace("(","")
        system_intern = system_intern.replace(")","")
        ## Adding ';' so the algorithm knows when to stop
        system_intern = system_intern+";"
        if system_intern[0]=='-':
            j_start=1
        else:
            j_start=0
        j = j_start
        while j < len(system_intern):
            print i
            print j
            print system_intern
            print j_start
            if system_intern[j:j+2]=='**':
                print system_intern[j_start:j]
                if not system_intern[j_start:j].isdigit():
                    addto = "= sympy.symbols('"+system_intern[j_start:j]+"')"
                    #print addto
                    exec(system_intern[j_start:j] + addto)
                    symbol_list.append(system_intern[j_start:j])
                j_start = j+2
                j +=2
                continue
            elif system_intern[j] == '*' or system_intern[j] == '+' or system_intern[j] == '-' or system_intern[j] == '/' or system_intern[j]==';':
                print system_intern[j_start:j]
                if not system_intern[j_start:j].isdigit():
                    addto = "= sympy.symbols('"+system_intern[j_start:j]+"')"
                    #print addto
                    exec(system_intern[j_start:j] + addto)
                    symbol_list.append(system_intern[j_start:j])
                j_start = j+1
            j+=1
        del system_intern

    symbol_list.sort()    
    
    C=len(symbol_list);
    i = 0
    nmax = len(symbol_list);
    n = 0
    while i < C-1 and n < nmax:
        n +=1
        if symbol_list[i] == symbol_list[i+1]:
            del symbol_list[i+1]
            C -=1
        else:
            i +=1
    
    parameter_list = []
    for i in range(len(symbol_list)):
        print i
        print symbol_list[i][0].istitle()
        if symbol_list[i][0].istitle() is False:
            parameter_list.append(symbol_list[i])
            
    symbol_list = parameter_list
            
    F = collections.defaultdict(dict)
    
    for i in range(System.__len__()):
        for j in range(len(symbol_list)):
            ## Evaluation of the system
            temp = eval(System.values()[i])
            ## Symbolic derivative of the system of interest
            temp = sympy.diff(temp,eval(symbol_list[j]))
            temp = temp*eval(symbol_list[j])/eval(System.keys()[i].replace("d","")+'+1e-6')
            F[System.keys()[i]][symbol_list[j]] = str(temp)
    return F,symbol_list

Out, symbol_list = Analytic_local_sensitivities(System)

def MakeCanonical(System,Parameters,Measurable_States):
    State_len = len(Measurable_States) 
    A = np.eye(State_len)
    U_len = sum(Measurable_States.values())
    B = np.zeros([State_len,U_len])
    j=0
    for i in range(State_len):
        if Measurable_States.values()[i] == 1:
            B[i,j]=1
            j+=1
    C = np.transpose(B)
    D = np.zeros([U_len,U_len])
    return A,B,C,D
    
#A,B,C,D = MakeCanonical(System,Parameters,Measurable_States)
#
#print A
#print B
#print C
#print D



def MakeModel(Modelname,System,Parameters,Out,symbol_list):
    """
    MakeFile(file_name): makes a file.
    """
    temp_path = os.path.dirname(os.path.realpath(__file__))+'/'+Modelname+'.py'
    print temp_path
    file = open(temp_path, 'w+')
    file.seek(0,0)
    
    file.write('#'+Modelname+'\n')
    
#    file.write('\n#Parameters\n\n')

    #for i in range(len(Parameters)):
        #file.write(str(Parameters.keys()[i]) + ' = ' + str(Parameters.values()[i])+'\n')
        
    #file.write('\nParameters = '+str(Parameters.keys()).replace("'","")+'\n')
        
    file.write('\n#System definition\n\n')
    
#    file.write('States = '+str(System.keys()).replace("'d","").replace("'","")+'\n\n')
    
    file.write('def system(States,t,Parameters):\n')
    for i in range(len(Parameters)):
        #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
        file.write('    '+str(Parameters.keys()[i]) + " = Parameters['"+Parameters.keys()[i]+"']\n")
    file.write('\n')
    for i in range(len(System)):
        file.write('    '+str(System.keys()[i]).replace("d","") + ' = States['+str(i)+']\n')
    file.write('\n')    
    for i in range(len(System)):
        file.write('    '+str(System.keys()[i]) + ' = ' + str(System.values()[i])+'\n')
    file.write('    return '+str(System.keys()).replace("'","")+'\n')
    
    file.write('\n#Sensitivities\n\n')
    
    file.write('def sensitivities(States,Parameters):\n')
    file.write('\n')
    for i in range(len(System)):
        file.write('    '+str(System.keys()[i]).replace("d","") + ' = States['+str(i)+']\n')
    file.write('\n') 
    for i in range(len(Parameters)):
        file.write('    '+str(Parameters.keys()[i]) + " = Parameters['"+Parameters.keys()[i]+"']\n")
    file.write('\n') 
    r = []
    for i in range(len(System.keys())):
        for j in range(len(symbol_list)):
            r.append(System.keys()[i]+'d'+symbol_list[j])
            file.write('    '+ r[-1] +' = ')
            file.write(Out[System.keys()[i]][symbol_list[j]]+'\n')
            #file.write(Out[System.keys()[i]][symbol_list[j]]+'*'+symbol_list[j]+'/'+str(System.keys()[i]).replace("d","")+'\n')
    file.write('    return '+str(r).replace("'",""))
            
    
    file.close()
    print 'Execution completed.'
    
    return r

r = MakeModel(Modelname,System,Parameters,Out,symbol_list)





sys.path.append('/media/DATA/Dropbox/Transaminase')
import MODEL_Halfreaction

#extra def solve_ode
#res = spin.odeint(MODEL_Halfreaction.system,inic,t,args=(Parameters,), full_output=1,hmax=0.0001)
res = spin.odeint(MODEL_Halfreaction.system,inic,t,args=(Parameters,), full_output=1)

#extra def plot... ->reform to matplotlib
pl.plot(t,res[0])
#pl.ylim([-1,10])
pl.legend(['E','Es','A','B','P','Q','EsQ','EP'])
pl.show()

fitfunc = lambda p, x: p[0]*x + p[1] # Target function
errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
p0 = [-0.5,1] # Initial guess for the parameters
output = []
bereik = range(2,1000)
for i in bereik:
    p1, success = optimize.leastsq(errfunc, p0[:], args=(t[0:i],res[0][0:i,0]),xtol=1e-9)
    output.append(p1[0]/p1[1])
    
#pl.plot(bereik,output)
#pl.show()


#berekenen van de 
test=[]
for i in range(len(t)):
    test.append(MODEL_Halfreaction.sensitivities(res[0][i,:],Parameters))

for h in range(len(System.keys())):
    pl.figure(h) #->reform to matplotlib
    print System.keys()[h].replace("d","")
    pl.suptitle(System.keys()[h].replace("d",""))
    for i in range(len(symbol_list)):
        sens = []
        for j in range(len(t)):
            sens.append(test[j][i+8*h])
        pl.subplot(4,2,i+1)
        k = pl.plot(t,sens)
        pl.legend((k),(symbol_list[i],))
        #pl.legend(r[8*i:7+8*i])
<<<<<<< HEAD
#    pl.savefig('/media/DATA/Dropbox/Transaminase/Sensitivity/Sensitivity_'+System.keys()[h]+'.pdf')
=======
    #pl.savefig('/media/DATA/Dropbox/Transaminase/Sensitivity/Sensitivity_'+System.keys()[h]+'.pdf')
>>>>>>> upstream/master
