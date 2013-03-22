r"""
ODE GENERATOR
*************
"""

from __future__ import division
import numpy as np
import scipy.integrate as spin
from scipy import optimize
import pylab as pl
import sympy
import sys
import ordereddict as collections
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

## Automatic conversion of variables to symbols
def AnalyticLocalSens(System,Parameters):
    r"""Analytic derivation of the local sensitivities
    """
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
    
    # Calculate direct and indirect sensitivities
    for j in range(System.__len__()+1):
        for i in range(System.__len__()):
            for k in range(len(Parameters)):
                ##Evaluation of the system
                Sensitivity_list.append(str(eval(System.values()[i])))
                # Symbolic derivative of the system to a certain parameter
                if j ==0:
                    # Make symbol for pythonscript
                    Sensitivity_symbols.append(System.keys()[i]+'d'+Parameters.keys()[k])
                    # Perform partial derivation to certian parameter
                    Sensitivity_list[-1] = sympy.diff(Sensitivity_list[-1],eval(Parameters.keys()[k]))
                # First replace certain state by its derivative and then perform partial derivative to specific parameter
                else:
                    # Make symbol for pythonscript
                    Sensitivity_symbols.append(System.keys()[i]+System.keys()[j-1]+'X'+System.keys()[j-1]+'d'+Parameters.keys()[k])
                    # Replace state by its derivative
                    exec(System.keys()[j-1][1:]+" = sympy.symbols('("+System.values()[j-1].replace(" ","")+")')")
                    # Evaluate                    
                    Sensitivity_list[-1] = eval(str(Sensitivity_list[-1]))
                    #temp = sympy.diff(temp,eval(System.keys()[j-1].replace("d","")))*sympy.diff(eval(System[System.keys()[j-1]]),eval(parameter_list[k]))
                    # Reset state to its original symbolic representation                    
                    exec(System.keys()[j-1][1:]+" = sympy.symbols('"+System.keys()[j-1][1:]+"')")
                    # Perform partial derivation to certian parameter
                    Sensitivity_list[-1] = sympy.diff(Sensitivity_list[-1],eval(Parameters.keys()[k]))
                   
                # Multiply sensitivity with the value of the parameter
                Sensitivity_list[-1] = Sensitivity_list[-1]*eval(Parameters.keys()[k])#/eval(symbol_list[i]+'+1e-6')
    
    return Sensitivity_symbols, Sensitivity_list
    

Sensitivity_symbols, Sensitivity_list = AnalyticLocalSens(System,Parameters)

def MakeCanonical(System,Parameters,Measurable_States,inic):
    print System.keys()
    # Symbolify parameters
    for i in range(len(Parameters)):
        exec(Parameters.keys()[i] + " = sympy.symbols('"+Parameters.keys()[i]+"')")
    # Symbolify states
    A = np.zeros([len(System),len(System)])
    A_list = []
    for i in range(len(System)):
        for j in range(len(System)):
            if i is not j:
                exec(System.keys()[j][1:]+"= sympy.symbols('"+System.keys()[j][1:]+"_eq')")
            else:
                exec(System.keys()[j][1:] +" = sympy.symbols('"+System.keys()[j][1:]+"')")
        for j in range(len(System)):
           A_list.append(sympy.integrate(sympy.diff(eval(System.values()[j]),eval(System.keys()[i][1:])),eval(System.keys()[i][1:]))/eval(System.keys()[i][1:]))
  
    
    for i in range(len(Parameters)):
        exec(Parameters.keys()[i]+' = '+str(Parameters.values()[i]))
    for i in range(len(System)):
        exec(inic.keys()[i]+'_eq = '+str(inic.values()[i]))
    
    for i in range(len(System)):
        for j in range(len(System)):
            A[i,j] = eval(str(A_list[i*len(System)+j]))

    B = np.zeros([len(Measurable_States) ,sum(Measurable_States.values())])
    j=0
    for i in range(len(Measurable_States)):
        if Measurable_States.values()[i] == 1:
            B[i,j]=1
            j+=1
    C = np.transpose(B)
    D = np.zeros([sum(Measurable_States.values()),sum(Measurable_States.values())])
    
    return A,B,C,D
    
A,B,C,D = MakeCanonical(System,Parameters,Measurable_States,inic)


def IdentifiabilityCheck(A,B,C,D):
    s = sympy.symbols('s')
   
    H2 = C*((s*sympy.eye(len(A))-A).inv())
    H1 = H2*B+D
    return H1,H2
    
H1,H2 = IdentifiabilityCheck(A,B,C,D)

#print H1
#print H2


iterations = 4

def TaylorSeriesApproach(System,Parameters,Measurable_States,inic,iterations):
    r"""
Identifiability: TaylorSeriesApproach
=====================================

Parameters
----------
System: OrderedDict
    Ordered dict with the keys as the derivative of a state (written as 'd'+State),
    the values of the dictionary is the ODE system written as a string
Parameters: OrderedDict
    Ordered dict with parameter names as keys, parameter values are the values
    of the dictionary
Measurable_States: OrderedDict
    Contains all the states, with key 'State' for all states. Values of 
    the orderedDict are 1 if output is measurable and 0 if output is not 
    measurable.
inic: OrderedDict
    Contains initial conditions for all states
iterations: int
    Number of derivatives the algorithm has to calculate (TaylorSeries!)

Returns
-------
Identifiability_Pairwise: array
    Contains for every Measurable state and every iteration, an array
    (number parameters x number parameters), values of 1 show that this
    parameter pair is not interchangeable. Values of 0 shows that this pair is 
    interchangeable. Values of 10 can be ignored.

Identifiability_Ghostparameter: array
    Contains for every Measurable state and every iteration, an array
    (1 x number parameters), a value of 1 show that this parameter is unique. 
    A value of 0 states that this parameter is not uniquely identifiable.
    

References
----------
.. [1] E. Walter and L. Pronzato, "Identification of parametric models from experimental data.", 1997.

Examples
--------
These are written in doctest format, and should illustrate how to
use the function.

>>> inic = {'X1':0,'X2':1}
>>> Parameters = {'p1':0,'p2':0,'p3':0}
>>> System = {'dX1':'-(p1+p2)*X1+p3*X2','dX2':'p1*X1-p3*X2'}
>>> Measurable_States = {'X1':0,'X2':1}
>>> iterations = 2

>>> Identifiability_Pairwise, Identifiability_Ghostparameter = 
TaylorSeriesApproach(System,Parameters,Measurable_States,inic,iterations)
>>> print Identifiability_Pairwise
>>> print Identifiability_Ghostparameter
 
"""
    intern_system = {}
    # Convert all parameters to symbols
    for i in range(len(Parameters)):
        exec(Parameters.keys()[i]+" = sympy.symbols('"+Parameters.keys()[i]+"')")
    # Add (t) to the different states in order to calculate the derivative to the time   
    for i in range(len(System)):
        exec(System.keys()[i][1:]+" = sympy.symbols('"+System.keys()[i][1:]+"(t)')")
    # Replace states without time by states WITH time
    for i in range(len(System)):
        intern_system[System.keys()[i]] = str(eval(System.values()[i]))
    # Sort internal system
    intern_system = collections.OrderedDict(sorted(intern_system.items(), key=lambda t: t[0]))
    # Symbolify t
    t = sympy.symbols('t')
    # Delete state symbols (only looking to time dependence)
    for i in range(len(System)):
        exec('del '+System.keys()[i][1:])
    # Construct empty identification matrix
    Identifiability_Pairwise = np.zeros([sum(Measurable_States.values()),iterations,len(Parameters),len(Parameters)])+10
    Identifiability_Ghostparameter = np.zeros([sum(Measurable_States.values()),iterations,len(Parameters)])+10
    # For all measurable states run TaylorSeriesApproac
    for h in range(sum(Measurable_States.values())):
        # Only perform identifiability analysis for measurable outputs
        h_measurable = np.where(np.array(Measurable_States.values())==1)[0][h]
        # Make list for measurable output derivatives
        Measurable_Output_Derivatives = []
        Measurable_Output_Derivatives_numerical_values = []
        # Make ghost parameter
        P_P_ghost = sympy.symbols('P_P_ghost')
        # Number of iterations = nth order-derivatives
        for i in range(iterations):
            if len(Measurable_Output_Derivatives) == 0:
                # Copy original system in dict
                Measurable_Output_Derivatives.append(str(intern_system['d'+System.keys()[h_measurable][1:]]))
            else:
                # Take derivative of previous element fo list
                Measurable_Output_Derivatives.append(str(sympy.diff(Measurable_Output_Derivatives[-1],t)))
            for j in range(len(System)):
                # Replace 'Derivative(X(t),t)' by dX(t) from system
                Measurable_Output_Derivatives[-1] = Measurable_Output_Derivatives[-1].replace("Derivative("+System.keys()[j][1:]+"(t), t)",'('+intern_system['d'+System.keys()[j][1:]]+')')
            Measurable_Output_Derivatives_numerical_values.append(Measurable_Output_Derivatives[-1])
            for j in range(len(System)):
                # Replace symbols by the corresponding numerical values
                Measurable_Output_Derivatives_numerical_values[-1] = Measurable_Output_Derivatives_numerical_values[-1].replace(System.keys()[j][1:]+"(t)",str(inic[System.keys()[j][1:]]))
                # Keep the symbolic values (still testing mode)                
                #AAA[-1] = AAA[-1].replace(state_list[j]+"(t)",str(state_list[j]))
            # Simplify sympy expression
            Measurable_Output_Derivatives[-1] = str(sympy.simplify(Measurable_Output_Derivatives[-1]))
            for j in range(len(Parameters)):
                for k in range(j+1,len(Parameters)):
                    # Exchange two symbols with each other
                    exec(Parameters.keys()[j]+" = sympy.symbols('"+Parameters.keys()[k]+"')")
                    exec(Parameters.keys()[k]+" = sympy.symbols('"+Parameters.keys()[j]+"')")
                    # Evaluate 'symbolic' expression
                    Measurable_Output_Derivatives_temp_plus = str(eval(Measurable_Output_Derivatives_numerical_values[i]))
                    # Reset symbols to their original values                    
                    exec(Parameters.keys()[k]+" = sympy.symbols('"+Parameters.keys()[k]+"')")
                    exec(Parameters.keys()[j]+" = sympy.symbols('"+Parameters.keys()[j]+"')")
                    # If answer is the same then these parameters are not identifiable
                    Identifiability_Pairwise[h,i,k,j] = eval(Measurable_Output_Derivatives_numerical_values[i]+' != '+Measurable_Output_Derivatives_temp_plus)
            for j in range(len(Parameters)):
                # Replace parameter by ghostparameter
                exec(Parameters.keys()[j]+" = sympy.symbols('P_P_ghost')")
                # Evaluate 'symbolic' expression
                Measurable_Output_Derivatives_temp_plus = str(eval(Measurable_Output_Derivatives_numerical_values[i]))
                # Reset parameter to its original value                   
                exec(Parameters.keys()[j]+" = sympy.symbols('"+Parameters.keys()[j]+"')")
                # If answer is the same then this parameter is not unique identifiable
                Identifiability_Ghostparameter[h,i,j] = eval(Measurable_Output_Derivatives_numerical_values[i]+' != '+Measurable_Output_Derivatives_temp_plus)
    return Identifiability_Pairwise, Identifiability_Ghostparameter

Identifiability_Pairwise, Identifiability_Ghostparameter = TaylorSeriesApproach(System,Parameters,Measurable_States,inic,iterations)

print Identifiability_Pairwise
print Identifiability_Ghostparameter

def MakeModel(Modelname,System,Parameters,Sensitivity_symbols,Sensitivity_list):
    """
    MakeFile(file_name): makes a file.
    """
    temp_path = os.getcwd()+'/'+Modelname+'.py'
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

MakeModel(Modelname,System,Parameters,Sensitivity_symbols,Sensitivity_list)
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
