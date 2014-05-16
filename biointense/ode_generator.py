"""
Created on Mon Mar 25 12:04:03 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator by Tvandaele
"""
from __future__ import division
import numpy as np
from numpy import *
import scipy.integrate as spin
import scipy.interpolate as spint
import sympy
import shutil
import sys
if sys.hexversion > 0x02070000:
    import collections
else:
    import ordereddict as collections
import os
import pandas as pd
import pprint

from matplotlib import colors
import matplotlib.pyplot as plt

from plotfunctions import *

np.set_printoptions(threshold=np.nan) # Necessary for printing big matrices in a proper way!

class DAErunner(object):
    '''
    Class to generated an ODE model based on a Parameter and Variables
    dictionary. Includes ODE-model run, identifiability analysis with Tatlor
    apporach and Laplace and numerical/analytical sensitivity analysis
    
    Parameters
    ------------
    System : OrderedDict
        Ordered dict with the keys as the derivative of a state (written as 
        'd'+State), the values of the dictionary is the ODE system written 
        as a string
    Parameters : OrderedDict
        Ordered dict with parameter names as keys, parameter values are the 
        values of the dictionary    
    Modelname : string
        String name to define the model
        
    
    Examples
    ----------
    >>> Parameters = {'k1':1/10, 'k1m':1/20,
                      'k2':1/20, 'k2m':1/20,
                      'k3':1/200,'k3m':1/175,
                      'k4':1/200,'k4m':1/165}
    >>> System = {'dEn':'k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ',
                  'dEs':'- k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ',
                  'dSA':'- k1*En*SA + k1m*Es*PP',
                  'dSB':'- k2*Es*SB + k2m*En*PQ',
                  'dPP':'k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP',
                  'dPQ':'k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ',
                  'dEsQ':'k3*Es*PQ - k3m*EsQ',
                  'dEP':'k4*En*PP - k4m*EP'}                            
    >>> Modelname = 'MODEL_Halfreaction'
    >>> #INITIATE MODEL
    >>> M1 = odegenerator(System, Parameters, Modelname = Modelname)
    >>> M1.set_measured_states(['SA', 'SB', 'PP', 'PQ'])
    >>> M1.set_initial_conditions({'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.})
    >>> M1.set_time({'start':1,'end':20,'nsteps':10000})
    >>> #run the model
    >>> modeloutput = M1.solve_ode(plotit=False)
    >>> modeloutput.plot(subplots=True) 
    >>> #run the taylor approach for identifiability, till second order
    >>>  M1.taylor_series_approach(2)
    '''
    
    def __init__(self,*args,**kwargs):
        '''

        '''
        
        if kwargs.get('print_on') == None:
            self._print_on = True
	else:
	    self._print_on = kwargs.get('print_on')

        self._has_def_algebraic = False
        self._has_def_ODE = False
        self.Parameters = collections.OrderedDict(sorted(kwargs.get('Parameters').items(), key=lambda t: t[0]))  
        try:        
            self.System = collections.OrderedDict(sorted(kwargs.get('ODE').items(), key=lambda t: t[0]))    
            self._Variables = [i[1:] for i in self.System.keys()]  
            self._has_ODE = True
            try:
                self._has_def_ODE = kwargs.get('has_def_ODE')
                if self._has_def_ODE and self._print_on:
                    print('ODE(s) were defined by user')
            except:
                pass
            
            if self._print_on:
                print(str(len(self.System)) + ' ODE(s) was/were defined. Continuing...')
        except:
            if self._print_on:
                print('No ODES defined. Continuing...')
            self._has_ODE = False
        
        
        self.modelname = kwargs.get('Modelname')
        
        try:
            Algebraic = kwargs.get('Algebraic')
            if self._print_on:
                print(str(len(Algebraic)) + ' Algebraic equation(s) was/were defined. Continuing...')
        except:
            try:
                self._has_def_algebraic = kwargs.get('has_def_algebraic')
                if self._has_def_algebraic and self._print_on:
                    print('You defined your own function, please make sure this function is provided in\
                        '+self.modelname+'.py as Algebraic_outputs'+'(self._Time, self.Parameters)')
            except:
                pass
            if self._has_ODE:
                Algebraic = {}
                for i in self._Variables:
                    Algebraic[i] = i
                if self._print_on:
                    print('No Algebraic equations defined. All ODEs are regarded as outputs...')
                self._has_algebraic = True
            else:
                raise Exception('A model without ODEs or output cannot be regarded as a model!')
        
        self.Algebraic =  collections.OrderedDict(sorted(Algebraic.items(),key=lambda t: t[0]))
        self._has_algebraic = True
        self._Outputs = self.Algebraic.keys()
        if self._has_ODE and not self._has_def_ODE:
            self._analytic_local_sensitivity()
        self._alg_LSA()
        self.solve_fast_way = True
        
        self._has_externalfunction = False     
        #self._write_model_to_file()

        self._ode_procedure = "odeint"
        self._generate_model()
        self._has_generated_model = False

        #Sensitivity stuff
        self.LSA_type = None
        
    def _reset_parameters(self, Parameters):
        '''Parameter stuff
        
        '''
        self.Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))

    def set_time(self,timedict):
        '''define time to calculate model
        
        The start, end and number of steps (nsteps) for the ode-calculation
        
        Parameters
        -----------
        timedict : dict
            three elments dictionary; start, end and nsteps. The latter is the
            number of timesteps between start and end to get output from
        
        '''
        if timedict['start'] > timedict['end']:
            raise Exception('End timestep must be smaller then start!')
        #if timedict['nsteps'] < (timedict['end'] - timedict['start']):
        if timedict['nsteps'] < 10:
            raise Exception('Step too small')
        
        self._TimeDict = timedict
        self._Time = np.linspace(timedict['start'], timedict['end'], timedict['nsteps'])

    def set_initial_conditions(self, inic):
        '''define initial conditions to calculate model
        
        The initial condition of the different variables
        
        Parameters
        -----------
        timedict : dict
            For every variable, an initial value (float) is needed
            
        Examples
        ----------
        >>> inic = {'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.}
        >>> set_initial_conditions(inic)
        '''
        self.Initial_Conditions = collections.OrderedDict(sorted(inic.items(), key=lambda t: t[0]))

    def set_measured_states(self, Measurable_States):
        '''define the measured variables
        
        Define which variables can be measured

        Parameters
        -----------
        Measurable_States : list
            string names of the variables that can be measured
            
        Examples
        ----------
        >>> set_measured_states(['SA', 'SB', 'PP', 'PQ'])        
        
        '''
        Measured_temp = {}
        if self._has_ODE:
            for key in self.System:
                Measured_temp[key] = 0
        if self._has_algebraic:
            for key in self.Algebraic:
                Measured_temp[key] = 0
        self._MeasuredList=[]
        
        Measurable_States.sort()
        
        for measured in Measurable_States:
            dmeasured = 'd' + measured
            if dmeasured in Measured_temp:
                Measured_temp[dmeasured] = 1
                self._MeasuredList.append(measured)
            elif self._has_algebraic and measured in Measured_temp:
                Measured_temp[measured] = 1
                self._MeasuredList.append(measured)    
            else:
                raise Exception('The variable',measured,'is not part of the current model.')

        self.Measurable_States = collections.OrderedDict(sorted(Measured_temp.items(), key=lambda t: t[0]))

    def get_variables(self):
        '''get a list of all model variables
        
        '''
        return self._Variables

    def get_measured_outputs(self):
        '''get a list of all measurable variables
        
        Help function for getting the values one can measure in the lab
        '''
        try:
            self._MeasuredList
        except:
            if self._print_on:
                print('All variables are assumed measured, since no selection was set')
            self.set_measured_states(self._Outputs)
        return self._MeasuredList
    
    def get_time(self):
        '''get the model-time characteristics
        
        '''
        try:
            self._TimeDict['start']
            if self._print_on:
                print('start timestep is '+ str(self._TimeDict['start']))
                print('end timestep is '+ str(self._TimeDict['end']))
                print('number of timesteps for printing is '+ str(self._TimeDict['nsteps']))
        except:
            raise Exception('Please add a time-dictionary containing start, end and nsteps')
            
    def _analytic_local_sensitivity(self):
        '''Analytic derivation of the local sensitivities
        
        Sympy based implementation to get the analytic derivation of the
        model sensitivities. Algebraic variables in the ODE equations are replaced
        by its equations to perform the analytical derivation.
                
        Notes
        ------
        
        The output is best viewed by the write_model_to_file method        
        
        See Also
        ---------
        _write_model_to_file
        
        '''    
        
        # Set up symbolic matrix of system states
        system_matrix = sympy.Matrix(sympy.sympify(self.System.values()))
        # Set up symbolic matrix of variables
        states_matrix = sympy.Matrix(sympy.sympify(self._Variables))
        # Set up symbolic matrix of parameters
        parameter_matrix = sympy.Matrix(sympy.sympify(self.Parameters.keys()))
        
        # Replace algebraic stuff in system_matrix to perform LSA
        if self._has_algebraic:
            # Set up symbolic matrix of algebraic
            algebraic_matrix = sympy.Matrix(sympy.sympify(self.Algebraic.keys()))
            
            h = 0
            while (np.sum(np.abs(system_matrix.jacobian(algebraic_matrix))) != 0) and (h <= len(self.Algebraic.keys())):
                for i, alg in enumerate(self._Outputs):
                    system_matrix = system_matrix.replace(alg, self.Algebraic.values()[i])
                h += 1
                         
        # Initialize and calculate matrices for analytic sensitivity calculation
        # dfdtheta
        dfdtheta = system_matrix.jacobian(parameter_matrix)
        self.dfdtheta = np.array(dfdtheta)
        # dfdx
        dfdx = system_matrix.jacobian(states_matrix)
        self.dfdx = np.array(dfdx)
        # dxdtheta
        dxdtheta = np.zeros([len(states_matrix),len(self.Parameters)])
        self.dxdtheta = np.asmatrix(dxdtheta)
        
#        #dgdtheta
#        dgdtheta = np.zeros([sum(self.Measurable_States.values()),len(self.Parameters)])
#        self.dgdtheta = np.array(dgdtheta)
#        #dgdx
#        dgdx = np.eye(len(self.states_matrix))*self.Measurable_States.values()
#        #Remove zero rows
#        self.dgdx = np.array(dgdx[~np.all(dgdx == 0, axis=1)])
        
    def _alg_swap(self):
        
        try:
            self.Algebraic_swapped
        except:        
            h = 0
            algebraic_matrix = sympy.Matrix(sympy.sympify(self.Algebraic.values()))
            algebraic_keys = sympy.Matrix(sympy.sympify(self.Algebraic.keys()))
            while (np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys))) != 0) and (h <= len(self.Algebraic.keys())):
                for i, alg in enumerate(self.Algebraic.keys()):
                    algebraic_matrix = algebraic_matrix.replace(alg, self.Algebraic.values()[i])
                h += 1
            
            self.Algebraic_swapped = algebraic_matrix
        
        return self.Algebraic_swapped

    def _alg_LSA(self):
        '''Analytic derivation of the local sensitivities
        
        Sympy based implementation to get the analytic derivation of the
        model sensitivities. Algebraic variables in the ODE equations are replaced
        by its equations to perform the analytical derivation.
                
        Notes
        ------
        
        The output is best viewed by the write_model_to_file method        
        
        See Also
        ---------
        _write_model_to_file
        
        '''    
        
        # Set up symbolic matrix of system states
        #algebraic_matrix = sympy.Matrix(sympy.sympify(self.Algebraic.values()))
        #algebraic_keys = sympy.Matrix(sympy.sympify(self.Algebraic.keys()))
        # Set up symbolic matrix of variables
        if self._has_ODE:
            states_matrix = sympy.Matrix(sympy.sympify(self._Variables))
        # Set up symbolic matrix of parameters
        parameter_matrix = sympy.Matrix(sympy.sympify(self.Parameters.keys()))
 
        algebraic_matrix = self._alg_swap()             
#        h = 0
#        while (np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys))) != 0) and (h <= len(self.Algebraic.keys())):
#            for i, alg in enumerate(self.Algebraic.keys()):
#                algebraic_matrix = algebraic_matrix.replace(alg, self.Algebraic.values()[i])
#            h += 1
                                
        # Initialize and calculate matrices for analytic sensitivity calculation
        # dgdtheta
        dgdtheta = algebraic_matrix.jacobian(parameter_matrix)
        self.dgdtheta = np.array(dgdtheta)
        # dgdx
        if self._has_ODE:
            dgdx = algebraic_matrix.jacobian(states_matrix)
            self.dgdx = np.array(dgdx)
        
    def _check_for_meas(self, Measurable_States):
        '''verify presence of measured states
        
        '''        
        #CHECK FOR THE MEASURABLE VARIBALES
        if Measurable_States == False:
            try:
                self._MeasuredList
                if self._print_on:
                    print('Used measured variables: '+str(self._MeasuredList))
            except:
                raise Exception('No measurable states are provided for the current model')            
        else:
            self.set_measured_states(Measurable_States)
            if self._print_on:
                print('Updated measured states are used')

    def _check_for_init(self, Initial_Conditions):
        '''verify presence of initial conditions
        
        '''
        #CHECK FOR THE INITIAL CONDITIONS
        if Initial_Conditions == False:
            try:
#                print 'Used initial conditions: ', self.Initial_Conditions
                self.Initial_Conditions
            except:
                raise Exception('No initial conditions are provided for the current model')            
        else:
            self.set_initial_conditions(Initial_Conditions)   
            if self._print_on:          
                print('Updated initial conditions are used')

    def _check_for_time(self, Timesteps):
        '''verify presence of model time information
        
        '''
        #CHECK FOR THE INITIAL CONDITIONS
        if Timesteps == False:
            try:
#                print 'Used timesteps: ', self._TimeDict
                self._TimeDict
            except:
                raise Exception('No time step information is provided for the current model')            
        else:
            self.set_time(Timesteps)
            if self._print_on:             
                print('Updated initial conditions are used')
              
    def makeStepFunction(self,array_list, accuracy=0.001):
        '''makeStepFunction
        
        A function for making multiple steps or pulses, the function can be used
        as an Algebraic equation. Just call stepfunction(t) for using the stepfunction 
        functionality. At this moment only one stepfunction can be used at a time, but
        can be included in multiple variables.
        
        Parameters
        -----------
        array_list: list
            Contains list of arrays with 2 columns and an undefined number of rows. 
            In the first column of the individual arrays the time at which a step
            is made. In the second column the value the function should go to.
        accuracy: float
            What is the maximal timestep for going from one value to another. By 
            increasing this value, less problems are expected with the solver. However
            accuracy will be decreases. The standard value is 0.001, but depending 
            on the system dynamics this can be altered.

        Returns
        -------
        stepfunction: function
            Function which automatically interpolates in between the steps which
            were given. Can also be called as self.stepfunction
       
        '''
        stepfunction = []        
        
        for n,array in enumerate(array_list):
            if array.shape[1] != 2:
                raise Exception("The input array should have 2 columns!")
            array_len = array.shape[0]
            x = np.zeros(2*array_len)
            y = np.zeros(2*array_len)
            if array[0,0] != 0:
                raise Exception("The first value of the stepfunction should be given at time 0!")
            x[0] = array[0,0]
            y[0] = array[0,1]
            for i in range(array_len-1):
                x[2*i+1] = array[i+1,0]-accuracy/2
                y[2*i+1] = array[i,1]
                x[2*(i+1)] = array[i+1,0]+accuracy/2
                y[2*(i+1)] = array[i+1,1]
            x[-1] = 1e15
            y[-1] = array[-1,1]
            
            if n is 0:
                stepfunction = [spint.interp1d(x, y)]
            else:
                stepfunction.append(spint.interp1d(x, y))
                
        return stepfunction
        
    def addExternalFunction(self, function):
        '''
        Add external function to the biointense module
        
        Comparable to stepfunction but more general now!
        '''
        
        self.externalfunction = function
        self._has_externalfunction = True  
        self._has_generated_model = False
        
    def _generate_model(self, procedure = 'ode'):
        '''Write derivative of model as definition in file
        
        Writes a file with a derivative definition to run the model and
        use it for other applications
        
        Parameters
        -----------
        
        '''
        temp_path = os.path.join(os.getcwd(),self.modelname+'.py') 
        temp_path_user = os.path.join(os.getcwd(),self.modelname+'_user.py')
        if self._has_def_algebraic and self._has_def_ODE:
            if self._print_on:
                print('Functions are defined by user, no model to be generated!')
            shutil.copyfile(temp_path_user,temp_path)
            return 0
                
        if self._has_ODE:
            try:
                self.dfdtheta
            except:
                if self._print_on:
                    print('Running symbolic calculation of analytic sensitivity ...')
                self._analytic_local_sensitivity()
                if self._print_on:
                    print('... Done!')       
       
        # Write function for solving ODEs only
        system = ""
        LSA_analytical = ""
        algebraic = ""
        algebraic_sens = ""
        if self._has_ODE and not self._has_def_ODE:
            if self._has_externalfunction:
                if self._ode_procedure == "ode":
                    system +='def _fun_ODE(t,ODES,Parameters,input):\n'
                else:
                    system +='def _fun_ODE(ODES,t,Parameters,input):\n'
            else:
                if self._ode_procedure == "ode":
                    system +='def _fun_ODE(t,ODES,Parameters):\n'
                else:
                    system +='def _fun_ODE(ODES,t,Parameters):\n'
        
            for i in range(len(self.Parameters)):
                #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                system +='    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
            system +='\n'
            
            for i in range(len(self.System)):
                system +='    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n'
            system +='\n'
            
            if self._has_externalfunction:
                for i, step in enumerate(self.externalfunction):
                    system +='    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                system +='\n'
                
            if self._has_algebraic:
                for i in range(len(self.Algebraic)):
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                    system +='    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n'
                system +='\n'
            
            for i in range(len(self.System)):
                system +='    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n'
        
            system +='    return '+str(self.System.keys()).replace("'","")+'\n\n\n'

            # Write function for solving ODEs of both system and analytical sensitivities
            if self._has_externalfunction:
                if self._ode_procedure == "ode":
                    LSA_analytical += 'def _fun_ODE_LSA(t,ODES,Parameters,input):\n'
                else:
                    LSA_analytical += 'def _fun_ODE_LSA(ODES,t,Parameters,input):\n'
            else:
                if self._ode_procedure == "ode":
                    LSA_analytical += 'def _fun_ODE_LSA(t,ODES,Parameters):\n'
                else:
                    LSA_analytical += 'def _fun_ODE_LSA(ODES,t,Parameters):\n'
            for i in range(len(self.Parameters)):
                #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                LSA_analytical += '    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
            LSA_analytical += '\n'
            for i in range(len(self.System)):
                LSA_analytical += '    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n'
            LSA_analytical += '\n'
            if self._has_externalfunction:
                for i, step in enumerate(self.externalfunction):
                    LSA_analytical += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                LSA_analytical += '\n'
            if self._has_algebraic:
                for i in range(len(self.Algebraic)):
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                    LSA_analytical += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n'
                LSA_analytical += '\n'
            for i in range(len(self.System)):
                LSA_analytical += '    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n'
            
            if self._print_on:
                print('ODE sensitivities are printed to string....')
            LSA_analytical += '\n    #Sensitivities\n\n'
            
            # Calculate number of states by using inputs
            LSA_analytical += '    state_len = len(ODES)/(len(Parameters)+1)\n'
            # Reshape ODES input to array with right dimensions in order to perform matrix multiplication
            LSA_analytical += '    dxdtheta = array(ODES[state_len:].reshape(state_len,len(Parameters)))\n\n'
            
            # Write dfdtheta as symbolic array
            LSA_analytical += '    dfdtheta = '
            LSA_analytical += pprint.pformat(self.dfdtheta)
            # Write dfdx as symbolic array
            LSA_analytical += '\n    dfdx = '
            LSA_analytical += pprint.pformat(self.dfdx)
            # Calculate derivative in order to integrate this
            LSA_analytical += '\n    dxdtheta = dfdtheta + dot(dfdx,dxdtheta)\n'
    
            LSA_analytical += '    return '+str(self.System.keys()).replace("'","")+'+ list(dxdtheta.reshape(-1,))'+'\n\n\n'
        
        if self._has_algebraic and not self._has_def_algebraic:
            if self._has_ODE:
                if self._has_externalfunction:
                    algebraic += '\ndef _fun_alg(ODES,t,Parameters, input):\n'
                else:
                    algebraic += '\ndef _fun_alg(ODES,t,Parameters):\n'
            elif self._has_externalfunction:
                algebraic += '\ndef _fun_alg(t,Parameters, input):\n'
            else:
                algebraic += '\ndef _fun_alg(t,Parameters):\n'
            for i in range(len(self.Parameters)):
                #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                algebraic += '    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
            algebraic += '\n'
            if self._has_ODE:
                for i in range(len(self.System)):
                    if self.solve_fast_way:
                        algebraic += '    '+str(self.System.keys()[i])[1:] + ' = ODES[:,'+str(i)+']\n'
                    else:
                        algebraic += '    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n'
                algebraic += '\n'
            if self._has_externalfunction:
                for i, step in enumerate(self.externalfunction):
                    algebraic += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                algebraic += '\n'
            for i in range(len(self.Algebraic)):
                #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+' + zeros(len(t))\n')
                algebraic += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])
                if self.solve_fast_way:
                    algebraic += ' + np.zeros(len(t))\n'
                else:
                    algebraic += '\n'
            algebraic += '\n'
            algebraic += '    algebraic = array('+str(self.Algebraic.keys()).replace("'","")+').T\n\n'
            algebraic += '    return algebraic\n\n\n'
            
            #

            if self._has_ODE and not self._has_def_ODE:
                #Algebraic sens
                if self._has_externalfunction:
                    algebraic_sens += '\ndef _fun_alg_LSA(ODES,t,Parameters, input, dxdtheta):\n'
                else:
                    algebraic_sens += '\ndef _fun_alg_LSA(ODES,t,Parameters, dxdtheta):\n'
                for i in range(len(self.Parameters)):
                    #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                    algebraic_sens += '    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
                algebraic_sens += '\n'
                for i in range(len(self.System)):
                    if self.solve_fast_way:
                        algebraic_sens += '    '+str(self.System.keys()[i])[1:] + ' = ODES[:,'+str(i)+']\n'
                    else:
                        algebraic_sens += '    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n'
                algebraic_sens += '\n'
                if self._has_externalfunction:
                    for i, step in enumerate(self.externalfunction):
                        algebraic_sens += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                    algebraic_sens += '\n'
                for i in range(len(self.Algebraic)):
                    algebraic_sens += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i]) +'\n'
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                if self._print_on:
                    print('Algebraic sensitivities are printed to string....')
                algebraic_sens += '\n    #Sensitivities\n\n'
                           
                # Write dgdtheta as symbolic array
                algebraic_sens += '    dgdtheta = np.zeros([len(t), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.Parameters.keys())) + '])\n'
                for i, alg in enumerate(self.Algebraic.keys()):
                    for j, par in enumerate(self.Parameters.keys()):
                        algebraic_sens += '    dgdtheta[:,' + str(i) + ',' + str(j) +'] = ' + str(self.dgdtheta[i,j])+'\n'

                # Write dgdx as symbolic array
                algebraic_sens += '    dgdx = np.zeros([len(t), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.System.keys())) + '])\n'
                for i, alg in enumerate(self.Algebraic.keys()):
                    for j, par in enumerate(self.System.keys()):
                        algebraic_sens += '    dgdx[:,' + str(i) + ',' + str(j) +'] = ' + str(self.dgdx[i,j])+'\n'

                # The two time-dependent 2D matrices should be multiplied with each other (dot product).
                # In order to yield a time-dependent 2D matrix, this is possible using the einsum function.
                algebraic_sens += "\n    dgdxdxdtheta = np.einsum('ijk,ikl->ijl',dgdx,dxdtheta)\n"
        
                algebraic_sens += '\n    dydtheta = dgdtheta + dgdxdxdtheta\n'
                
            else:
                #Algebraic sens
                if self._has_externalfunction:
                    algebraic_sens += '\ndef _fun_alg_LSA(t,Parameters, input):\n'
                else:
                    algebraic_sens += '\ndef _fun_alg_LSA(t,Parameters):\n'
                for i in range(len(self.Parameters)):
                    #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                    algebraic_sens += '    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
                algebraic_sens += '\n'
                if self._has_externalfunction:
                    for i, step in enumerate(self.externalfunction):
                        algebraic_sens += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                    algebraic_sens += '\n'
                for i in range(len(self.Algebraic)):
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                    algebraic_sens += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n'
                algebraic_sens += '\n'
                if self._print_on:
                    print('Algebraic sensitivities are printed to the file....')
                algebraic_sens += '\n    #Sensitivities\n\n'
                           
                # Write dgdtheta as symbolic array
                algebraic_sens += '    dgdtheta = '   
                algebraic_sens += pprint.pformat(self.dgdtheta)

                algebraic_sens += '\n\n    dydtheta = dgdtheta\n'
    
            algebraic_sens += '    return dydtheta'+'\n\n\n'
        
        if self._has_ODE:
            exec(system)
            exec(LSA_analytical)
            self._fun_ODE_str = system
            self._fun_ODE_LSA_str = LSA_analytical
            self._fun_ODE = _fun_ODE
            self._fun_ODE_LSA = _fun_ODE_LSA
        else:
            self._fun_ODE_str = None
            self._fun_ODE_LSA_str = None
            self._fun_ODE = None
            self._fun_ODE_LSA = None

        if self._has_algebraic:
            exec(algebraic)
            exec(algebraic_sens)
            self._fun_alg_str = algebraic
            self._fun_alg_LSA_str = algebraic_sens
            self._fun_alg = _fun_alg
            self._fun_alg_LSA = _fun_alg_LSA
        else:
            self._fun_alg_str = None
            self._fun_alg_LSA_str = None
            self._fun_alg = None
            self._fun_alg_LSA = None
        if self._print_on:
            print('...All functions are generated!')

        
    def solve_algebraic(self, plotit = True):
        """
        """
        if not self._has_algebraic:
            raise Exception('This model has no algebraic equations!')
        if not (self._has_ODE or self._has_generated_model):
            self._generate_model()
            self._has_generated_model = True
        if self._has_ODE:
            try:
                self.ode_solved
            except:
                raise Exception('First run self.solve_ode()!')
            
        if self.solve_fast_way:
            #try:
                if self._has_ODE:
                    if self._has_externalfunction:
                        algeb_out = self._fun_alg(np.array(self.ode_solved), self._Time, self.Parameters, self.externalfunction)
                    else:
                        algeb_out =self._fun_alg(np.array(self.ode_solved), self._Time, self.Parameters)             
                elif self._has_externalfunction:
                    algeb_out = self._fun_alg(np.array(self.ode_solved), self._Time, self.Parameters)
                else:
                    algeb_out = self._fun_alg(self._Time, self.Parameters)  
            #except:
             #   raise Exception('Maybe you should take the slow for loop way to results or adapt your functions? (self.solve_fast_way = False)')
        else:
            #try:
                algeb_out = np.zeros([len(self._Time),len(self.Algebraic)])
                if self._has_ODE:
                    if self._has_externalfunction:
                        for i, timestep in enumerate(self._Time):
                            algeb_out[i,:] = self._fun_alg(np.array(self.ode_solved.ix[timestep]), timestep, self.Parameters, self.externalfunction)
                    else:
                        for i, timestep in enumerate(self._Time):
                            algeb_out[i,:] = self._fun_alg(np.array(self.ode_solved.ix[timestep]), timestep, self.Parameters)
                elif self._has_externalfunction:
                    for i, timestep in enumerate(self._Time):
                        algeb_out[i,:] = self._fun_alg(np.array(self.ode_solved.ix[timestep]), timestep, self.Parameters)
                else:
                    for i, timestep in enumerate(self._Time):
                        algeb_out[i,:] = self._fun_alg(timestep, self.Parameters)
            #except:
             #   raise Exception('Maybe you should take the fast for loop way to results or adapt your functions? (self.solve_fast_way = True)')
       
        
        self.algeb_solved = pd.DataFrame(algeb_out, columns=self.Algebraic.keys(), 
                                 index = self._Time)
        
        #plotfunction
        if plotit == True:
            if len(self.Algebraic) == 1:
                self.algeb_solved.plot(subplots = False)
            else:
                self.algeb_solved.plot(subplots = True)
                                 
    def calcAlgLSA(self, Sensitivity = 'CAS'):
        """
        """
        try:
            self.algeb_solved
        except:
            if self._print_on:
                print('First running self.solve_algebraic()')
            self.solve_algebraic(plotit = False)
            
        if not self._has_algebraic:
            raise Exception('This model has no algebraic equations!')
        if not self._has_ODE or self._has_generated_model:
            self._generate_model()
            self._has_generated_model = True
        if self._has_ODE:
            try:
                self._ana_sens_matrix
            except:
                if self._print_on:
                    print('Running ODE LSA...')
                self.calcOdeLSA()
                if self._print_on:
                    print('...Done! Going to Algebraic LSA...')

               
        algeb_out = np.empty((self._Time.size, len(self.Algebraic.keys()) ,len(self.Parameters)))         
        
        if self._has_ODE:
            if self._has_externalfunction:
                algeb_out = self._fun_alg_LSA(np.array(self.ode_solved), self._Time, self.Parameters, self.externalfunction, self._ana_sens_matrix)
            else:
                algeb_out = self._fun_alg_LSA(np.array(self.ode_solved), self._Time, self.Parameters, self._ana_sens_matrix) 
        elif self._has_externalfunction:
            algeb_out = self._fun_alg_LSA(self._Time, self.Parameters, self.externalfunction)
        else:
            algeb_out = self._fun_alg_LSA(self._Time, self.Parameters)           
               
        alg_dict = {}
        
        for i,key in enumerate(self.Algebraic.keys()):
            alg_dict[key] = pd.DataFrame(algeb_out[:,i,:], columns=self.Parameters.keys(), 
                                 index = self._Time)
       
        self.getAlgLSA = self._LSA_converter(self.algeb_solved, alg_dict, self.Algebraic.keys(), Sensitivity,'ALGSENS')
                
        return self.getAlgLSA

    def solve_ode(self, TimeStepsDict = False, Initial_Conditions = False, 
                  plotit = True, with_sens = False, procedure = "odeint"):
        '''Solve the differential equation
        
        Solves the ode model with the given properties and model configuration
        
        Parameters
        -----------
        TimeStepsDict : False|dict
            If False, the time-attribute is checked for and used. 
        Initial_Conditions : False|dict
            If False, the initial conditions  attribute is checked for and used. 
        
        Returns
        ---------
        df : pandas DataFrame
            DataFrame with in the columns the the different variable outputs
        
        See Also
        ---------
        set_initial_conditions, set_time
                
        '''
        if not self._has_ODE:
            raise Exception('This model has no ODE equations!')
        #        print 'Current parameters', self.Parameters.values()
        self._check_for_time(TimeStepsDict)
        self._check_for_init(Initial_Conditions)
        
        if (self._has_generated_model and self._ode_procedure is not procedure) or \
                (self._has_generated_model is False):
            if self._print_on:
                print("Writing model to file for '" + procedure + "' procedure...")
            self._ode_procedure = procedure
            self._generate_model(procedure = procedure)
            self._has_generated_model = True
            if self._print_on:
                print('...Finished writing to file!')
        elif self._print_on:
            print("Model was already written to file! We are using the '" + \
                procedure + "' procedure for solving ODEs. If you want to rewrite \
                the model to the file, please add 'write = True'.")
        
        	
        if with_sens == False:
            if procedure == "odeint":
                if self._print_on:
                    print("Going for odeint...")
                if self._has_externalfunction:
                    res = spin.odeint(self._fun_ODE,self.Initial_Conditions.values(), self._Time,args=(self.Parameters,self.externalfunction,))
                else:
                    res = spin.odeint(self._fun_ODE,self.Initial_Conditions.values(), self._Time,args=(self.Parameters,))
                #put output in pandas dataframe
                df = pd.DataFrame(res, index=self._Time, columns = self._Variables)                
    
    
            elif procedure == "ode":
                if self._print_on:
                    print("Going for generic methodology...")
                #ode procedure-generic
                #                f = eval(self.modelname+'.system')
                r = spin.ode(self._fun_ODE).set_integrator('lsoda') #   method='bdf', with_jacobian = False
                    #                                                    nsteps = 300000)
                r.set_initial_value(self.Initial_Conditions.values(), 0).set_f_params(self.Parameters)
                dt = (self._TimeDict['end']-self._TimeDict['start'])/self._TimeDict['nsteps'] #needs genereic version
                moutput = []
                toutput = []
                end_val = self._Time[-1]-self._TimeDict['end']/(10*self._TimeDict['nsteps'])             
                if self._print_on:                
                    print("Starting ODE loop...")
                while r.successful() and r.t < end_val:          
                    r.integrate(r.t + dt)
                    #print "This integration worked well?", r.successful()
                    #print("%g %g" % (r.t, r.y[0]))
                    #                    print "t: ", r.t
                    #                    print "y: ", r.y
    
                    moutput.append(r.y)
                    toutput.append(r.t)
                if self._print_on:
                    print("...Done!")
                
                #make df
                df = pd.DataFrame(moutput, index = toutput, 
                                  columns = self._Variables)          
        else:
            #odeint procedure
            if procedure == "odeint":
                if self._print_on:
                    print("Going for odeint...")
                if self._has_externalfunction:
                    res = spin.odeint(self._fun_ODE_LSA, np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), self._Time,args=(self.Parameters,self.externalfunction,))
                else:
                    res = spin.odeint(self._fun_ODE_LSA, np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), self._Time,args=(self.Parameters,))
                #put output in pandas dataframe
                df = pd.DataFrame(res[:,0:len(self._Variables)], index=self._Time,columns = self._Variables)
                if self._has_algebraic:               
                    self._ana_sens_matrix = res[:,len(self._Variables):].reshape(len(self._Time),len(self._Variables),len(self.Parameters))
                    #self._ana_sens_matrix = np.rollaxis(np.rollaxis(self._ana_sens_matrix,1,0),2,1)
                analytical_sens = {}
                for i in range(len(self._Variables)):
                    #Comment was bug!
                    #analytical_sens[self._Variables[i]] = pd.DataFrame(res[:,len(self._Variables)*(1+i):len(self._Variables)*(1+i)+len(self.Parameters)], index=self._Time,columns = self.Parameters.keys())
                    analytical_sens[self._Variables[i]] = pd.DataFrame(res[:,len(self._Variables)+len(self.Parameters)*(i):len(self._Variables)+len(self.Parameters)*(1+i)], index=self._Time,columns = self.Parameters.keys())
                
            elif procedure == "ode":
                if self._print_on:                
                    print("Going for generic methodology...")
                #ode procedure-generic
                r = spin.ode(self._fun_ODE_LSA)
                r.set_integrator('vode', method='bdf', with_jacobian = False)
                if self._has_externalfunction:
                    r.set_initial_value(np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), 0)
                    r.set_f_params(self.Parameters,self.stepfunction)                    
                else:
                    r.set_initial_value(np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), 0)
                    r.set_f_params(self.Parameters)
                dt = (self._TimeDict['end']-self._TimeDict['start'])/self._TimeDict['nsteps']
                moutput = []
                toutput = []
                end_val = self._Time[-1]-self._TimeDict['end']/(10*self._TimeDict['nsteps'])
                if self._print_on:                
                    print("Starting ODE loop...")
                while r.successful() and r.t < end_val:
                    r.integrate(r.t+dt)
#                    print("%g %g" % (r.t, r.y))
                    moutput.append(r.y)
                    toutput.append(r.t)
                if self._print_on:
                    print("...Done!")
                
                moutput = np.array(moutput)
                df = pd.DataFrame(moutput[:,0:len(self._Variables)], index=toutput,columns = self._Variables)
                
                if self._has_algebraic:               
                    self._ana_sens_matrix = moutput[:,len(self._Variables):].reshape(len(toutput),len(self._Variables),len(self.Parameters))
                    #self._ana_sens_matrix = np.rollaxis(np.rollaxis(self._ana_sens_matrix,1,0),2,1)
                analytical_sens = {}
                for i in range(len(self._Variables)):
                    #Comment was bug!
                    #analytical_sens[self._Variables[i]] = pd.DataFrame(res[:,len(self._Variables)*(1+i):len(self._Variables)*(1+i)+len(self.Parameters)], index=self._Time,columns = self.Parameters.keys())
                    analytical_sens[self._Variables[i]] = pd.DataFrame(moutput[:,len(self._Variables)+len(self.Parameters)*(i):len(self._Variables)+len(self.Parameters)*(1+i)], index=toutput,columns = self.Parameters.keys())
                
                #make df
#                df = pd.DataFrame(moutput, index = toutput, 
#                                  columns = self._Variables)
                #                print "df is: ", df

        #plotfunction
        if plotit == True:
            if len(self._Variables) == 1:
                df.plot(subplots = False)
            else:
                df.plot(subplots = True)
               
        self.ode_solved = df
        if self._has_algebraic:
            #print "If you want the algebraic equations also, please rerun manually\
             #by using the 'self.solve_algebraic()' function!"
        
            self.solve_algebraic(plotit = plotit)

        if with_sens == False:
            return df
        else:
            return df, analytical_sens
        
    def collinearity_check(self,variable):
        '''
        
        Collinearity check calculates whether variables show collinear behaviour or not.
        Collinearity is only useful when using Total Relative Sensitivity, because it is
        the relative change which is important. One should bear in mind that collinearity
        measures vary between 0 and infinity. At the internet I found that for values
        between 15-30 you need to watch out. Above 30 you are in trouble and above 100 is 
        a real disaster :-)
        
        Parameters
        -----------
        variable : string
            Give the variable for which the collinearity check has to be performed. 
        
        Returns
        ---------
        df : pandas DataFrame
            DataFrame with in the columns and rows the different parameters and there
            corresponding collinearity values.
        
        See Also
        ---------
        analytical_sensitivity
        '''
        try:
            self.analytical_sensitivity
        except:
            if self._print_on:            
                print('Running Analytical sensitivity analysis')
            self.analytic_local_sensitivity(Sensitivity = 'CTRS')
            if self._print_on:
                print('... Done!')

        if self.LSA_type != 'CTRS':
            raise Exception('The collinearity_check function is only useful for Total Relative Sensitivity!')
     
        # Make matrix for storing collinearities per two parameters
        Collinearity_pairwise = np.zeros([len(self.Parameters),len(self.Parameters)])
        for i,parname1 in enumerate(self.Parameters):
            for j,parname2 in enumerate(self.Parameters.keys()[i:]):
                # Transpose is performed on second array because by selecting only one column, python automatically converts the column to a row!
                # Klopt enkel voor genormaliseerde sensitiviteiten
                # collinearity = |X.X'|
                X = np.matrix(np.vstack([self.analytical_sensitivity[variable][parname1],self.analytical_sensitivity[variable][parname2]]))
                Collinearity_pairwise[i,i+j] = np.sqrt(1/min(np.linalg.eigvals(np.array(X*X.transpose()))))

        x = pd.DataFrame(Collinearity_pairwise, index=self.Parameters.keys(), columns = self.Parameters.keys())
        
        try:
            self.Collinearity_Pairwise[variable] = x
        except:
            self.Collinearity_Pairwise = {}
            self.Collinearity_Pairwise[variable] = x
            
    def calcOdeLSA(self, approach = 'analytical', **kwargs):
        '''Calculate Local Sensitivity
        
        For every parameter calculate the sensitivity of the output variables.
        
        Parameters
        -----------
        approach : string
            String should refer to one of the two possible sensitivity\
            approaches: 'analytical' or 'numerical'         
        
        Returns
        --------
        sens : dict
            each variable gets a t timesteps x k par DataFrame
            
        '''
        if approach == 'analytical':
            sens = self.analytic_local_sensitivity(**kwargs)
        elif approach == 'numerical':
            sens = self.numeric_local_sensitivity(**kwargs)
        else:
            raise Exception("Approach needs to be defined as 'analytical' or 'numerical'")
        return sens

    def _LSA_converter(self, df, sens_matrix, sens_variables, sens_method, procedure):
        '''
        '''
        if sens_method == 'CPRS':
            #CPRS = CAS*parameter
            for i in sens_variables:
                 sens_matrix[i] = sens_matrix[i]*self.Parameters.values()
        elif sens_method == 'CTRS':
            #CTRS
            if min(df.mean()) == 0 or max(df.mean()) == 0:
                self.LSA_type = None
                raise Exception(procedure+': It is not possible to use the CTRS method for\
                    calculating sensitivity, because one or more variables are\
                    fixed at zero. Try to use another method or to change the\
                    initial conditions!')
            elif min(df.min()) == 0 or max(df.max()) == 0:
                if self._print_on:                
                    print(procedure+' Using AVERAGE of output values')
                for i in sens_variables:
                     sens_matrix[i] = sens_matrix[i]*self.Parameters.values()/df[i].mean()
            else:
                if self._print_on:
                    print(procedure+'ANASENS: Using EVOLUTION of output values')
                for i in sens_variables:
                     sens_matrix[i] = sens_matrix[i]*self.Parameters.values()/np.tile(np.array(df[i]),(len(self.Parameters),1)).T
        elif sens_method != 'CAS':
            self.LSA_type = None
            raise Exception('You have to choose one of the sensitivity\
             methods which are available: CAS, CPRS or CTRS')
        
        print(procedure+': The ' + sens_method + ' sensitivity method is used, do not\
                forget to check whether outputs can be compared!')
        return sens_matrix
    
    def analytic_local_sensitivity(self, Sensitivity = 'CAS', procedure = 'odeint'):
        '''Calculates analytic based local sensitivity 
        
        For every parameter calculate the sensitivity of the output variables.
        
        Parameters
        -----------
        Sensitivity : string
            String should refer to one of the three possible sensitivity\
            measures: Absolute Sensitivity (CAS), Parameter Relative Sensitivity (CPRS) or
            Total Relative Sensitivity (CTRS)'         
        
        Returns
        --------
        analytical_sens : dict
            each variable gets a t timesteps x k par DataFrame
            
        '''
        if not self._has_ODE:
            raise Exception('This model has no ODE equations!')
        self.LSA_type = Sensitivity

        df, analytical_sens = self.solve_ode(with_sens = True, plotit = False, procedure = procedure)      
        
        self.analytical_sensitivity = self._LSA_converter(df, analytical_sens, self._Variables, Sensitivity,'ANASENS')
        
        return analytical_sens
        

    def numeric_local_sensitivity(self, perturbation_factor = 0.0001, 
                                  TimeStepsDict = False, 
                                  Initial_Conditions = False,
                                  Sensitivity = 'CAS'):
        '''Calculates numerical based local sensitivity 
        
        For every parameter calculate the sensitivity of the output variables.
        
        Parameters
        -----------
        perturbation_factor : float (default 0.0001)
            factor for perturbation of the parameter value to approximate 
            the derivative
        TimeStepsDict : False|dict
            If False, the time-attribute is checked for and used. 
        Initial_Conditions : False|dict
            If False, the initial conditions  attribute is checked for and used.
        Sensitivity : string
            String should refer to one of the three possible sensitivity\
            measures: Absolute Sensitivity (CAS), Parameter Relative Sensitivity (CPRS) or
            Total Relative Sensitivity (CTRS)'        
        
        Returns
        --------
        numerical_sens : dict
            each variable gets a t timesteps x k par DataFrame
            
        '''
        self._check_for_time(TimeStepsDict)
        self._check_for_init(Initial_Conditions)
        self.LSA_type = Sensitivity
         
        #create a dictionary with everye key the variable and the values a dataframe
        numerical_sens = {}
        for key in self._Variables:
            #belangrijk dat deze dummy in loop wordt geschreven!
            dummy = np.empty((self._Time.size,len(self.Parameters)))
            numerical_sens[key] = pd.DataFrame(dummy, index=self._Time, columns = self.Parameters.keys())
        
        for i,parameter in enumerate(self.Parameters):
            value2save = self.Parameters[parameter]
#            print 'sensitivity for parameter ', parameter
            #run model with parameter value plus perturbation 
            self.Parameters[parameter] = value2save + perturbation_factor*value2save
            modout_plus = self.solve_ode(plotit = False)
#            modout_plus = pd.DataFrame(modout, columns = self._Variables)
            #run model with parameter value minus perturbation 
            self.Parameters[parameter] = value2save - perturbation_factor*value2save
            modout_min = self.solve_ode(plotit = False)        
#            modout_min = pd.DataFrame(modout, columns = self._Variables)
            self.Parameters[parameter] = value2save
            modout = self.solve_ode(plotit = False) 
            
            #calculate sensitivity for this parameter, all outputs    
            #sensitivity indices:
#            CAS = (modout_plus-modout_min)/(2.*perturbation_factor*value2save) #dy/dp         
            #CAS
            sensitivity_out = (modout_plus-modout)/(perturbation_factor*value2save) #dy/dp

            #put on the rigth spot in the dictionary
            for var in self._Variables:
                numerical_sens[var][parameter] = sensitivity_out[var][:].copy()
               
            #put back original value
            self.Parameters[parameter] = value2save
               
        self.numerical_sensitivity = self._LSA_converter(df, numerical_sens, self._Variables, Sensitivity, 'NUMSENS')

        return self.numerical_sensitivity

    def visual_check_collinearity(self, output, analytic = False, layout = 'full', upperpane = 'pearson'):
        '''show scatterplot of sensitivities
        
        Check for linear dependence of the local sensitivity outputs for a 
        specific variable. If the sensitivity output of two parameters is 
        highly related, those parameters are probably interfering for 
        this model output
        
        Parameters
        ------------        
        output : str
            name of the variable to get the collinearity check from
        layout : full|half
            full doubles the visualisation, half only shows the lower half of 
            the scattermatrix
        upperpane : pearson|spearman|kendall|data
            Decision about the content of the upper pane of the graph if full
            layout is selected; implemented are pearson, spearman, kendall 
            correlation coefficients; when data is chosen, the data is plotted again            
            
        '''
        if analytic == False:
            try:
                self.numerical_sensitivity
            except:
                self.numeric_local_sensitivity()
            
            toanalyze = self.numerical_sensitivity[output].as_matrix().transpose()
        else:
            try:
                self.analytical_sensitivity
            except:
                self.analytic_local_sensitivity()
            
            toanalyze = self.analytical_sensitivity[output].as_matrix().transpose()
            
        fig, axes = scatterplot_matrix(toanalyze, plottext=self.Parameters.keys(), plothist = False,
                           layout = layout, upperpane = upperpane, marker='o', color='black', mfc='none')
        plt.draw()

    def plot_collinearity(self, ax1, redgreen = False):
        '''plot of calculated collinearity check 
        
        Make an overview plot of the collinearity calculation as decribed in
        literature
        
        Parameters
        -----------
        ax1 : matplotlib axis instance
            the axis will be updated by the function
        redgreen :  boolean
            if True, red/grees color is used instead of greyscale

        Returns
        ---------
        ax1 : matplotlib axis instance
            axis with the plotted output 
        
        Notes
        ------
        Collinearity check is implemented as described in [2]_, where
        a threshold is defined to identify the dependence between parameters.
        
        References
        -----------
        .. [2] Brun, R., Reichert, P., Kfinsch, H.R., Practical Identifiability
            Analysis of Large Environmental Simulation (2001), 37, 1015-1030
            
        TODO update this to useful purpose!
        '''
        
        mat_to_plot = np.zeros([len(self._Variables),len(self.Parameters),len(self.Parameters)])
        for i in range(len(self._Variables)):
            try:
                mat_to_plot[i,:,:] = self.Collinearity_Pairwise[self._Variables[i]]
            except:
                if self._print_on:                
                    print('Collinearity check of variable '+self._Variables[i]+' is now performed!') 
                mat_to_plot[i,:,:] = self.collinearity_check(self._Variables[i])
            
        textlist = {}
        #Plot the rankings in the matrix
        for i in range(len(self._Variables)):
            F = (mat_to_plot[i] <400.)*(mat_to_plot[i] >0.)
            #print F
            for j in range(len(self.Parameters)):
                for k in range(len(self.Parameters)):
                    if F[j,k] == True:
                        #print j,k
                        #print F[j,k]
                        try:
                            #print textlist[(self._Variables[i],self.Parameters.keys()[j])]
                            textlist[(self._Variables[i],self.Parameters.keys()[j])]=textlist[(self._Variables[i],self.Parameters.keys()[j])]+','+(self.Parameters.keys()[k])   
                        except:
                            textlist[(self._Variables[i],self.Parameters.keys()[j])]=self.Parameters.keys()[k]  
                        try:
                            #print textlist[(self._Variables[i],self.Parameters.keys()[k])]
                            textlist[(self._Variables[i],self.Parameters.keys()[k])]=textlist[(self._Variables[i],self.Parameters.keys()[k])]+','+(self.Parameters.keys()[j])   
                        except:
                            textlist[(self._Variables[i],self.Parameters.keys()[k])]=self.Parameters.keys()[j]  

        #print textlist
        #sorted(textlist[('EsQ','k1')].split(), key=str.lower)
        
#        if redgreen == True:
        cmap = colors.ListedColormap(['FireBrick','YellowGreen'])
#        else:
#            cmap = colors.ListedColormap(['.5','1.'])
#            
        bounds=[0,0.9,2.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
#        #plot tje colors for the frist tree parameters
        ax1.matshow(mat_to_plot[0],cmap = cmap, norm=norm)
        ax1.set_aspect('auto')
                
        for i in range(len(textlist)):
            yplace = np.where(np.core.defchararray.find(self._Variables,textlist.keys()[i][0])==0)[0][0]
            xplace = np.where(np.core.defchararray.find(self.Parameters.keys(),textlist.keys()[i][1])==0)[0][0]
            ax1.text(xplace, yplace, textlist.values()[i],fontsize=14, horizontalalignment='center', verticalalignment='center')   
        
        #place ticks and labels
        ax1.set_xticks(np.arange(8))
        ax1.set_xbound(-0.5,np.arange(8).size-0.5)
        ax1.set_xticklabels(self.Parameters.keys(), rotation = 30, ha='left')
        
        for i in range(7):
            ax1.hlines(i+0.5,-0.5,7.5)
            ax1.vlines(i+0.5,-0.5,7.5)
        
        ax1.set_yticks(np.arange(8))
        ax1.set_ybound(np.arange(8).size-0.5,-0.5)
        ax1.set_yticklabels(self._Variables)
        
        ax1.spines['bottom'].set_color('none')
        ax1.spines['right'].set_color('none')
        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        return ax1
        
    def _getCoefficients(self, enzyme):
        '''Filter enzyme equations and forms out of ODE system and convert 
            the filtered system to its canonical form.
        
        Parameters
        -----------
        enzyme : string
            All enzyme forms have to start with the same letters, e.g. 'En' or
            'E_'. This allows the algorithm to select the enzyme forms.

        Returns
        ---------
        coeff_matrix : sympy Matrix
            Contains the coefficients of the canonical system of enzyme_equations.
        enzyme_forms : sympy Matrix
            Contains all enzyme forms which are present in the system.
        enzyme_equations: sympy Matrix
            Contains the corresponding rate equation of the different enzyme
            forms.
        
        Notes
        ------
        The conncection between the three returns is the matrix multiplication:
        coeff_matrix*enzyme_forms = enzyme_equations
        
        '''
        
        enzyme_forms = []
        
        for var in self._Variables:
            if var.startswith(enzyme):
                enzyme_forms.append(var)

        # Set up symbolic matrix of enzyme states
        enzyme_equations = sympy.Matrix(sympy.sympify([self.System['d'+i] for i in enzyme_forms]))       
        # Set up symbolic matrix of enzymes
        enzyme_forms = sympy.Matrix(sympy.sympify(enzyme_forms))

        coeff_matrix = sympy.zeros(len(enzyme_equations),len(enzyme_forms))        
        
        for i,syst in enumerate(enzyme_equations):
            for j,state in enumerate(enzyme_forms):
                coeff_matrix[i,j] = syst.coeff(state)
                
        return coeff_matrix, enzyme_forms, enzyme_equations      
        
    def makeQSSA(self, enzyme = 'En' , variable = 'PP'):
        '''Calculate quasi steady-state equation out of ODE system 
        
        This function calculates the quasi steady-state equation for the 
        variable of interest
        
        Parameters
        -----------
        enzyme : string
            All enzyme forms have to start with the same letters, e.g. 'En' or
            'E_'. This allows the algorithm to select the enzyme forms, otherwise
            a reduction is not possible.
        variable: string
            Which rate equation has to be used to replace the enzyme forms with
            the QSSA.

        Returns
        ---------
        QSSA_var : sympy equation
            Symbolic sympy equation of variable which obeys the QSSA.
            
        QSSA_enz : sympy equation
            Symbolic sympy equation of all enzyme forms which obeys the QSSA.
        
        Notes
        ------
        The idea for the calculations is based on [1]_, where the system is
        first transformed in its canonical form.
        
        References
        -----------
        .. [1] Ishikawa, H., Maeda, T., Hikita, H., Miyatake, K., The 
            computerized derivation of rate equations for enzyme reactions on 
            the basis of the pseudo-steady-state assumption and the 
            rapid-equilibrium assumption (1988), Biochem J., 251, 175-181
        
        Examples
        ---------
        >>> System = {'dEn':'-k1*En*SA + k2*EnSA + kcat*EnSA',
                  'dEnSA':'k1*En*SA - k2*EnSA - kcat*EnSA',
                  'dSA':'-k1*En*SA + k2*EnSA',
                  'dPP':'kcat*EnSA'}
        >>> Parameters = {'k1':0,'k2':0,'kcat':0}
        >>> Modelname = 'QSSA_MM'
        >>> M1 = odegenerator(System, Parameters, Modelname = Modelname)
        >>> M1.makeQSSA(enzyme = 'En', variable = 'PP')        
        '''
        
        # Run _getCoefficients to get filtered rate equations
        coeff_matrix, enzyme_forms, enzyme_equations = self._getCoefficients(enzyme)
        
        # Add row with ones to set the sum of all enzymes equal to En0
        coeff_matrix = coeff_matrix.col_join(sympy.ones([1,len(enzyme_forms)]))      
        
        # Make row matrix with zeros (QSSA!), but replace last element with
        # En0 for fixing total som of enzymes
        QSSA_matrix = sympy.zeros([coeff_matrix.shape[0],1])
        QSSA_matrix[-1] = sympy.sympify('En0')
        
        # Add column with outputs to coeff_matrix
        linear_system = coeff_matrix.row_join(QSSA_matrix)
        
        # Find QSSE by using linear solver (throw away one line (system is closed!))
        # list should be passed as *args, therefore * before list
        QSSE_enz = sympy.solve_linear_system(linear_system[1:,:],*list(enzyme_forms))

        # Replace enzyme forms by its QSSE in rate equation of variable of interest
        QSSE_var = sympy.sympify(self.System['d' + variable])
        for enz in enzyme_forms:
            QSSE_var = QSSE_var.replace(enz,QSSE_enz[enz])
        
        # To simplify output expand all terms (=remove brackets) and afterwards
        # simplify the equation
        QSSE_var = sympy.simplify(sympy.expand(QSSE_var))
        
        self.QSSE_var = QSSE_var
        self.QSSE_enz = QSSE_enz
        
    def QSSAtoModel(self, substrates, products):
        system = {}
        for i in substrates:
            system['d'+i] = '-QSSE'
        for i in products:
            system['d'+i] = 'QSSE'
            
        algebraic = {}
        algebraic['QSSE'] = str(self.QSSE_var)
        return odegenerator(System = system, Parameters = self.Parameters, Modelname = self.modelname + '_QSSA', Algebraic = algebraic)
               
    def checkMassBalance(self, variables = 'En'):
        '''Check mass balance of enzyme forms
        
        This function checks whether the sum of all enzyme forms is equal to zero.
        
        Parameters
        -----------
        variables : string
            There are two possibilies: First one can give just the first letters of
            all enzyme forms, the algorithm is selecting all variables starting with
            this combination. Second, one can give the symbolic mass balance himself
            the algorithm will check the mass balance. See examples!
            
        Returns
        ---------
        massBalance : sympy symbolics
            If this is zero then mass balance is closed, otherwise the remaining
            terms are shown.
            
        Examples
        ----------
        >>>System = {'dEn':'-k1*En*SA + k2*EnSA + kcat*EnSA',
                     'dEnSA':'k1*En*SA - k2*EnSA - kcat*EnSA',
                     'dSA':'-k1*En*SA + k2*EnSA',
                     'dPP':'kcat*EnSA'}
        >>>Parameters = {'k1':0,'k2':0,'kcat':0}
        >>>Modelname = 'QSSA_MM'
        >>>#INITIATE MODEL
        >>>M1 = odegenerator(System, Parameters, Modelname = Modelname)
        >>>M1.checkMassBalance(variables='En')
        >>>#Or one could also write
        >>>M1.checkMassBalance(variables='En + EnSA')
        >>>#One could also make linear combination of mass balances, this is
        especially useful for systems like NO, NO2 and N2. In which the mass balance
        for N is equal to NO + NO2 + 2*N2 = 0.
        >>>M1.checkMassBalance(variables='En + 2*EnSA')
            
        '''
        
        variables = variables.replace(" ","")
        len_var = len(variables.split('+'))
        
        if len_var == 1:
            var_forms = []
            string = ''
            
            for var in self._Variables:
                if var.startswith(variables):
                    var_forms.append(var)
                    string = string + '+' + self.System['d'+var]
            massBalance = sympy.sympify(string)
        
        elif len_var > 1:
            var_sym = sympy.sympify(variables)
            # Set up symbolic matrix of system
            system_matrix = sympy.Matrix(sympy.sympify(self.System.values()))
            # Set up symbolic matrix of variables
            states_matrix = sympy.Matrix(sympy.sympify(self._Variables))
            
            massBalance = 0
            for i,var in enumerate(states_matrix):
                massBalance += var_sym.coeff(var)*system_matrix[i]
                
        else:
            raise Exception("The argument 'variables' need to be provided!")
        
        if self._print_on:
            if massBalance == 0:
                print("The mass balance is closed!")
            else:
                print("The mass balance is NOT closed for the enzyme forms of '" + variables +"'! \
                    The following term(s) cannot be striked out: " + str(massBalance))
        
        return massBalance
