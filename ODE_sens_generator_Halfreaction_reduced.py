from __future__ import division
import numpy as np
import scipy.integrate as spin
from scipy import optimize
import pylab as pl
import sympy
import sys
import collections
import os
import pprint

# Define the initial conditions for each of the four ODEs
#inic = [0,1]
inic = {'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.}

t = np.linspace(0, 20, 10000)

Modelname = 'MODEL_Halfreaction'

#Parameters = {'p1':0,'p2':0,'p3':0}
Parameters = {'k1':1/10,'k1m':1/20,
              'k2':1/20,'k2m':1/20,
              'k3':1/200,'k3m':1/175,
              'k4':1/200,'k4m':1/165}
              
Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))
Parameters.names = Parameters.keys()
Parameters.vals = Parameters.values()

  
#System = {'dX1':'-p1*X1-p2*(1-p3*X2)*X1',
#        'dX2':'p2*(1-p3*X2)*X1-p4*X2'}
#System = {'dX1':'-(p1+p2)*X1+p3*X2',
#          'dX2':'p1*X1-p3*X2'}
System =    {'dEn':'k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ',
             'dEs':'- k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ',
             'dSA':'- k1*En*SA + k1m*Es*PP',
             'dSB':'- k2*Es*SB + k2m*En*PQ',
             'dPP':'k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP',
             'dPQ':'k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ',
             'dEsQ':'k3*Es*PQ - k3m*EsQ',
             'dEP':'k4*En*PP - k4m*EP'}
             
System = collections.OrderedDict(sorted(System.items(), key=lambda t: t[0]))
System.names = System.keys()
System.vals = System.values()

Measurable_States = {'En':0,'Es':0,'SA':1,'SB':1,'PP':1,'PQ':1,'EsQ':0,'EP':0}
#Measurable_States = {'X1':0,'X2':1}
Measurable_States = collections.OrderedDict(sorted(Measurable_States.items(), key=lambda t: t[0]))

#def Symbolify(dictionary):
#    for i in range(len(dictionary.keys())):
#        exec("global "+dictionary.keys()[i] /
#        dictionary.keys()[i]+" = sympy.symbols('"+dictionary.keys()[i]+"')")
#        print eval(dictionary.keys()[i])
#    return dictionary.keys()
#
#Symbolify(Parameters)

def Symbolify(Parameters,Measurable_states):
    for i in range(len(Parameters.keys())):
        exec(Parameters.keys()[i]+" = sympy.symbols('"+Parameters.keys()[i]+"')")
    for i in range(len(Measurable_states.keys())):
        exec(Measurable_states.keys()[i]+" = sympy.symbols('"+Measurable_states.keys()[i]+"')")        
    return dictionary.keys()

Symbolify(Parameters)
    

## Automatic conversion of variables to symbols
def Analytic_local_sensitivities(System,Parameters):
    '''
    Analytic derivation of the local sensitivities
    '''
    # Symbolify system states
    for i in range(len(System)):
        exec(System.keys()[i][1:]+" = sympy.symbols('"+System.keys()[i][1:]+"')")
    # Symbolify parameters
    for i in range(len(Parameters)):
        exec(Parameters.keys()[i]+" = sympy.symbols('"+Parameters.keys()[i]+"')")
    
    # Make empty list for saving symbolic sensitivity formulas
    Sensitivity_list = []
    # Make empty list for saving combined symbols corresponding with sensitivity
    Sensitivity_symbols = []
    
    # 
    for j in range(System.__len__()+1):
        for i in range(System.__len__()):
            for k in range(len(Parameters)):
                ##Evaluation of the system
                Sensitivity_list.append(str(eval(System.values()[i])))
                # Symbolic derivative of the system to a certain parameter
                if j ==0:
                    Sensitivity_symbols.append(System.keys()[i]+'d'+Parameters.keys()[k])
                    Sensitivity_list[-1] = sympy.diff(Sensitivity_list[-1],eval(Parameters.keys()[k]))
                # First replace certain state by its derivative and then perform partial derivative to specific parameter
                else:
                    Sensitivity_symbols.append(System.keys()[i]+System.keys()[j-1]+'X'+System.keys()[j-1]+'d'+Parameters.keys()[k])
                    exec(System.keys()[j-1][1:]+" = sympy.symbols('("+System.values()[j-1].replace(" ","")+")')")
                    Sensitivity_list[-1] = eval(str(Sensitivity_list[-1]))
                    #temp = sympy.diff(temp,eval(System.keys()[j-1].replace("d","")))*sympy.diff(eval(System[System.keys()[j-1]]),eval(parameter_list[k]))
                    exec(System.keys()[j-1][1:]+" = sympy.symbols('"+System.keys()[j-1][1:]+"')")
                    Sensitivity_list[-1] = sympy.diff(Sensitivity_list[-1],eval(Parameters.keys()[k]))
                   
                    print Sensitivity_list[-1]
                Sensitivity_list[-1] = Sensitivity_list[-1]*eval(Parameters.keys()[k])#/eval(symbol_list[i]+'+1e-6')
                #F[System.keys()[i]][parameter_list[k]] = str(temp)
#                Sensivitivity_list.append(temp)
#    
    file = open(os.getcwd()+'/Sensitivity_Out.py','w+')
    file.seek(0,0)
    file.write('def Sensitivity_direct(States,Parameters):\n')
    temp = []
    for i in range(len(System)):
        file.write('    '+System.keys()[i][1:]+" = States['"+System.keys()[i][1:]+"']\n")
    file.write('\n')
    for i in range(len(Parameters)):
        file.write('    '+Parameters.keys()[i]+" = Parameters['"+Parameters.keys()[i]+"']\n")
    file.write('\n')
    for i in range(len(System)*len(Parameters)):
        file.write('    '+Sensitivity_symbols[i]+' = '+str(Sensitivity_list[i])+'\n')
        exec(Sensitivity_symbols[i]+" = sympy.symbols('"+Sensitivity_symbols[i]+"')")
        temp.append(eval(Sensitivity_symbols[i]))
    file.write('    Output = {}\n')
    for i in range(System.__len__()):
        for j in range(len(Parameters)):
            file.write("    Output['"+'d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+"'] = "+'d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+'\n')
   
    file.write('    return Output\n')
#    pprint.pprint(temp,file)
    file.write('\n')
    temp = []
    test = []
    file.write('def Sensitivity_indirect(States,Parameters):\n')
    for i in range(len(System)):
        file.write('    '+System.keys()[i][1:]+" = States['"+System.keys()[i][1:]+"']\n")
    file.write('\n')
    for i in range(len(Parameters)):
        file.write('    '+Parameters.keys()[i]+" = Parameters['"+Parameters.keys()[i]+"']\n")
    file.write('\n')
    for i in range(len(System)*len(Parameters),len(Sensitivity_symbols)):
        file.write('    '+Sensitivity_symbols[i]+' = '+str(Sensitivity_list[i])+'\n')
        temp.append(Sensitivity_symbols[i])
    file.write('    Output = {}\n')
    for i in range(System.__len__()):
        for j in range(len(Parameters)):
            file.write('    d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+' = ')
            for k in range(System.__len__()):
                file.write('d'+System.keys()[i][1:]+'d'+System.keys()[k][1:]+'Xd'+System.keys()[k][1:]+'d'+Parameters.keys()[j]+' + ') 
            file.seek(-3,2)
            file.write('\n')
#            for k in range(System.__len__()):
#                file.write("    Output['"+'d'+state_list[i]+'d'+state_list[k]+'Xd'+state_list[k]+'d'+parameter_list[j]+"'] = " + 'd'+state_list[i]+'d'+state_list[k]+'Xd'+state_list[k]+'d'+parameter_list[j]+'\n')
                
    
            exec('d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+" = sympy.symbols('"+'d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+"')")
            test.append(eval('d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]))
    
    for i in range(System.__len__()):
        for j in range(len(Parameters)):
            file.write("    Output['"+'d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+"'] = "+'d'+System.keys()[i][1:]+'d'+Parameters.keys()[j]+'\n')
            
    
    file.write('    return Output\n')
    file.close()
            
    return Sensitivity_list
    

Out = Analytic_local_sensitivities(System)

#def MakeCanonical(System,Parameters,Measurable_States,parameter_list,state_list):
#    #for i in range(len(System.values())):
#    for i in range(len(parameter_list)):
#        addto = "= sympy.symbols('"+parameter_list[i]+"')"
#        exec(parameter_list[i] + addto)
##    for i in range(len(state_list)):
###        addto = "= sympy.symbols('"+state_list[i]+"')"
###        exec(state_list[i] + addto)
##        addto = "= sympy.symbols('"+state_list[i]+"c')"
##        exec(state_list[i] + addto)
#    temp_path = os.path.dirname(os.path.realpath(__file__))+'/'+'Canonical_system'+'.py'
#    print temp_path
#    file = open(temp_path, 'w+')
#    file.seek(0,0)
#    file.write('# Canonical_system\n\n')
#    file.write('import numpy as np\n')
#    file.write('import sympy\n')
#    file.write('def Linear_A(States_eq,Parameters):\n')
#    file.write('\n')
#    for i in range(len(state_list)):
#        file.write('    '+state_list[i]+"_eq = sympy.symbols('"+state_list[i]+"_eq')\n")
#    file.write('\n')
#    for i in range(len(parameter_list)):
#        file.write('    '+parameter_list[i]+" = sympy.symbols('"+Parameters.keys()[i]+"')\n")
#    file.write('\n')
#    
#    for i in range(len(state_list)):
#        for j in range(len(state_list)):
#            if i is not j:
#                addto = "= sympy.symbols('"+state_list[j]+"_eq')"
#                exec(state_list[j] + addto)
#            else:
#                addto = "= sympy.symbols('"+state_list[j]+"')"
#                exec(state_list[j] + addto)
#        for j in range(len(System.values())):
#            file.write('    '+System.keys()[j]+'d')
#            file.write(state_list[i]+' = ')
#            file.write(str(sympy.diff(eval(System.values()[j]),eval(state_list[i])))+'\n')
#    Out_A = [0]*len(state_list)
#    file.write('    '+'return np.matrix([')
#    for i in range(len(state_list)):
#        for j in range(len(state_list)):
#            Out_A[j] = System.keys()[i]+'d'+state_list[j]
#        file.write(str(Out_A).replace("'","")+',')
#    file.seek(-1,2)
#    file.write('])\n')
#
#            
#    State_len = len(Measurable_States) 
#    U_len = sum(Measurable_States.values())
#    B = np.zeros([State_len,U_len])
#    j=0
#    for i in range(State_len):
#        if Measurable_States.values()[i] == 1:
#            B[i,j]=1
#            j+=1
#    file.write('\ndef Linear_B():\n   return np.')
#    pprint.pprint(B,file)
#    file.write('\ndef Linear_C():\n   return np.')
#    C = np.transpose(B)
#    pprint.pprint(C,file)
#    file.write('\ndef Linear_D():\n   return np.')
#    D = np.zeros([U_len,U_len])
#    pprint.pprint(D,file)
#    file.close()
#    
#MakeCanonical(System,Parameters,Measurable_States,symbol_list,state_list)

#sys.path.append('/media/DATA/Dropbox/Transaminase/biointense')
#
#temp_path = os.path.dirname(os.path.realpath(__file__))+'/'+'Test'+'.py'
#print temp_path
#file = open(temp_path, 'w+')
#file.seek(0,0)
#
#def IdentifiabilityCheck(inic,Parameters):
#    import Canonical_system
#    A = Canonical_system.Linear_A(inic,Parameters)
#    B = Canonical_system.Linear_B()
#    C = Canonical_system.Linear_C()
#    D = Canonical_system.Linear_D()
#    
#    s = sympy.symbols('s')
#    
#    pprint.pprint((s*sympy.eye(len(A))-A).adjugate(),file)
#    
##    H2 = (s*sympy.eye(len(A))-A).inv()
##    H1 = C*H2*B+D
##    return H1,H2
#    
##IdentifiabilityCheck(inic,Parameters)
##print H1
##print H2
##
##
##pprint.pprint(H1,file)
##pprint.pprint(H2,file)
#
#file.close()

iterations = 2

def TaylorSeriesApproach(System,state_list,parameter_list,Measurable_States,inic,iterations):
    '''
    Identifiability: TaylorSeriesApproach
    '''
    intern_system = {}
    # Convert all parameters to symbols
    for i in range(len(parameter_list)):
        exec(parameter_list[i]+" = sympy.symbols('"+parameter_list[i]+"')")
    # Add (t) to the different states in order to calculate the derivative to the time   
    for i in range(len(state_list)):
        exec(state_list[i]+" = sympy.symbols('"+state_list[i]+"(t)')")
    # Replace states without time by states WITH time
    for i in range(len(System)):
        intern_system[System.keys()[i]] = str(eval(System.values()[i]))
    # Sort internal system
    intern_system = collections.OrderedDict(sorted(intern_system.items(), key=lambda t: t[0]))
    # Symbolify t
    t = sympy.symbols('t')
    # Delete state symbols (only looking to time dependence)
    for i in range(len(state_list)):
        exec('del '+state_list[i])
    # Construct empty identification matrix
    Ident_matrix = np.zeros([sum(Measurable_States.values()),iterations,len(parameter_list),len(parameter_list)])+10
    # For all measurable states run TaylorSeriesApproac
    for h in range(sum(Measurable_States.values())):
        # Only perform identifiability analysis for measurable outputs
        h_measurable = np.where(np.array(Measurable_States.values())==1)[0][h]
        # Make list for measurable output derivatives
        Measurable_Output_Derivatives = []
        Measurable_Output_Derivatives_numerical_values = []
        # Number of iterations = nth order-derivatives
        for i in range(iterations):
            if len(Measurable_Output_Derivatives) == 0:
                # Copy original system in dict
                Measurable_Output_Derivatives.append(str(intern_system['d'+state_list[h_measurable]]))
            else:
                # Take derivative of previous element fo list
                Measurable_Output_Derivatives.append(str(sympy.diff(Measurable_Output_Derivatives[-1],t)))
            for j in range(len(state_list)):
                # Replace 'Derivative(X(t),t)' by dX(t) from system
                Measurable_Output_Derivatives[-1] = Measurable_Output_Derivatives[-1].replace("Derivative("+state_list[j]+"(t), t)",'('+intern_system['d'+state_list[j]]+')')
            #Measurable_Output_Derivatives_numerical_values.append(Measurable_Output_Derivatives[-1])
            for j in range(len(state_list)):
                # Replace symbols by the corresponding numerical values
                Measurable_Output_Derivatives_numerical_values[-1] = Measurable_Output_Derivatives_numerical_values[-1].replace(state_list[j]+"(t)",str(inic[state_list[j]]))
                # Keep the symbolic values (still testing mode)                
                #AAA[-1] = AAA[-1].replace(state_list[j]+"(t)",str(state_list[j]))
            # Simplify sympy expression
            Measurable_Output_Derivatives[-1] = str(sympy.simplify(Measurable_Output_Derivatives[-1]))
            for j in range(len(parameter_list)):
                for k in range(j+1,len(parameter_list)):
                    # Exchange two symbols with each other
                    exec(parameter_list[j]+" = sympy.symbols('"+parameter_list[k]+"')")
                    exec(parameter_list[k]+" = sympy.symbols('"+parameter_list[j]+"')")
                    # Evaluate 'symbolic' expression
                    Measurable_Output_Derivatives_temp_plus = str(eval(Measurable_Output_Derivatives_numerical_values[i]))
                    # Reset symbols to their original values                    
                    exec(parameter_list[k]+" = sympy.symbols('"+parameter_list[k]+"')")
                    exec(parameter_list[j]+" = sympy.symbols('"+parameter_list[j]+"')")
                    # If answer is the same then these parameters are not identifiable
                    Ident_matrix[h,i,k,j] = eval(Measurable_Output_Derivatives_numerical_values[i]+' != '+Measurable_Output_Derivatives_temp_plus)      
    return Ident_matrix

Ident_matrix = TaylorSeriesApproach(System,state_list,symbol_list,Measurable_States,inic,iterations)

print Ident_matrix
#
#def MakeModel(Modelname,System,Parameters,Out,symbol_list):
#    """
#    MakeFile(file_name): makes a file.
#    """
#    temp_path = os.path.dirname(os.path.realpath(__file__))+'/'+Modelname+'.py'
#    print temp_path
#    file = open(temp_path, 'w+')
#    file.seek(0,0)
#    
#    file.write('#'+Modelname+'\n')
#    
##    file.write('\n#Parameters\n\n')
#
#    #for i in range(len(Parameters)):
#        #file.write(str(Parameters.keys()[i]) + ' = ' + str(Parameters.values()[i])+'\n')
#        
#    #file.write('\nParameters = '+str(Parameters.keys()).replace("'","")+'\n')
#        
#    file.write('\n#System definition\n\n')
#    
##    file.write('States = '+str(System.keys()).replace("'d","").replace("'","")+'\n\n')
#    
#    file.write('def system(States,t,Parameters):\n')
#    for i in range(len(Parameters)):
#        #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
#        file.write('    '+str(Parameters.keys()[i]) + " = Parameters['"+Parameters.keys()[i]+"']\n")
#    file.write('\n')
#    for i in range(len(System)):
#        file.write('    '+str(System.keys()[i]).replace("d","") + ' = States['+str(i)+']\n')
#    file.write('\n')    
#    for i in range(len(System)):
#        file.write('    '+str(System.keys()[i]) + ' = ' + str(System.values()[i])+'\n')
#    file.write('    return '+str(System.keys()).replace("'","")+'\n')
#    
#    file.write('\n#Sensitivities\n\n')
#    
#    file.write('def sensitivities(States,Parameters):\n')
#    file.write('\n')
#    for i in range(len(System)):
#        file.write('    '+str(System.keys()[i]).replace("d","") + ' = States['+str(i)+']\n')
#    file.write('\n') 
#    for i in range(len(Parameters)):
#        file.write('    '+str(Parameters.keys()[i]) + " = Parameters['"+Parameters.keys()[i]+"']\n")
#    file.write('\n') 
#    r = []
#    for i in range(len(System.keys())):
#        for j in range(len(symbol_list)):
#            r.append(System.keys()[i]+'d'+symbol_list[j])
#            file.write('    '+ r[-1] +' = ')
#            file.write(Out[System.keys()[i]][symbol_list[j]]+'\n')
#            #file.write(Out[System.keys()[i]][symbol_list[j]]+'*'+symbol_list[j]+'/'+str(System.keys()[i]).replace("d","")+'\n')
#    file.write('    return '+str(r).replace("'",""))
#            
#    
#    file.close()
#    print 'Execution completed.'
#    
#    return r
#
#r = MakeModel(Modelname,System,Parameters,Out,symbol_list)
#
#
#
#
#
#sys.path.append('/media/DATA/Dropbox/Transaminase')
#import MODEL_Halfreaction
#
##extra def solve_ode
##res = spin.odeint(MODEL_Halfreaction.system,inic,t,args=(Parameters,), full_output=1,hmax=0.0001)
#res = spin.odeint(MODEL_Halfreaction.system,inic,t,args=(Parameters,), full_output=1)
#
##extra def plot... ->reform to matplotlib
#pl.plot(t,res[0])
##pl.ylim([-1,10])
#pl.legend(['E','Es','A','B','P','Q','EsQ','EP'])
#pl.show()
#
#fitfunc = lambda p, x: p[0]*x + p[1] # Target function
#errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
#p0 = [-0.5,1] # Initial guess for the parameters
#output = []
#bereik = range(2,1000)
#for i in bereik:
#    p1, success = optimize.leastsq(errfunc, p0[:], args=(t[0:i],res[0][0:i,0]),xtol=1e-9)
#    output.append(p1[0]/p1[1])
#    
##pl.plot(bereik,output)
##pl.show()
#
#
##berekenen van de 
#test=[]
#for i in range(len(t)):
#    test.append(MODEL_Halfreaction.sensitivities(res[0][i,:],Parameters))
#
#for h in range(len(System.keys())):
#    pl.figure(h) #->reform to matplotlib
#    print System.keys()[h].replace("d","")
#    pl.suptitle(System.keys()[h].replace("d",""))
#    for i in range(len(symbol_list)):
#        sens = []
#        for j in range(len(t)):
#            sens.append(test[j][i+8*h])
#        pl.subplot(4,2,i+1)
#        k = pl.plot(t,sens)
#        pl.legend((k),(symbol_list[i],))
#        #pl.legend(r[8*i:7+8*i])
#    #pl.savefig('/media/DATA/Dropbox/Transaminase/Sensitivity/Sensitivity_'+System.keys()[h]+'.pdf')
