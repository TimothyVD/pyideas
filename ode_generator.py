"""
Created on Mon Mar 25 12:04:03 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator by Tvandaele
"""
from __future__ import division
import numpy as np
from numpy import *
from sympy import *
import scipy.integrate as spin
import scipy.interpolate as spint
import sympy
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

class odegenerator(object):
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
    
    def __init__(self, System, Parameters, Modelname = 'MyModel',*args,**kwargs):
        '''

        '''
        
        self.Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))
        self.System = collections.OrderedDict(sorted(System.items(), key=lambda t: t[0]))    
        
        self.modelname = Modelname
        self._Variables = [i[1:] for i in self.System.keys()]
        
        try:
            self.Algebraic =  collections.OrderedDict(sorted(kwargs.get('Algebraic').items(),key=lambda t: t[0]))
            self._has_algebraic = True
            self._alg_swap()
            self._alg_LSA()
        except:
            print 'No Algebraic equations defined. Continuing...'
            self._has_algebraic = False
            
        self._has_stepfunction = False
        self._analytic_local_sensitivity()            
        #self._write_model_to_file()
                
        self._wrote_model_to_file = False
        self._ode_procedure = "ode"

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
            string names of the variables thta can be measured
            
        Examples
        ----------
        >>> set_measured_states(['SA', 'SB', 'PP', 'PQ'])        
        
        '''
        Measured_temp = {}
        for key in self.System:
            Measured_temp[key] = 0        
        self._MeasuredList=[]
        
        Measurable_States.sort()
        
        for measured in Measurable_States:
            dmeasured = 'd' + measured
            print dmeasured
            if dmeasured in Measured_temp:
                Measured_temp[dmeasured] = 1
                self._MeasuredList.append(measured)
            
            else:
                raise Exception('The variable',measured,'is not part of the current model.')

        self.Measurable_States = collections.OrderedDict(sorted(Measured_temp.items(), key=lambda t: t[0]))

    def get_variables(self):
        '''get a list of all model variables
        
        '''
        return self._Variables

    def get_measured_variables(self):
        '''get a list of all measurable variables
        
        Help function for getting the values one can measure in the lab
        '''
        try:
            self._MeasuredList
        except:
            print 'All variables are assumed measured, since no selection was set'
            self.set_measured_states(self._Variables)
        return self._MeasuredList
    
    def get_time(self):
        '''get the model-time characteristics
        
        '''
        try:
            self._TimeDict['start']
            print 'start timestep is ', self._TimeDict['start']
            print 'end timestep is ', self._TimeDict['end']
            print 'number of timesteps for printing is ', self._TimeDict['nsteps']
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
                for i, alg in enumerate(self.Algebraic.keys()):
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
        print algebraic_matrix
        dgdtheta = algebraic_matrix.jacobian(parameter_matrix)
        self.dgdtheta = np.array(dgdtheta)
        # dgdx
        dgdx = algebraic_matrix.jacobian(states_matrix)
        self.dgdx = np.array(dgdx)
        
        
    def _check_for_meas(self, Measurable_States):
        '''verify presence of measured states
        
        '''        
        #CHECK FOR THE MEASURABLE VARIBALES
        if Measurable_States == False:
            try:
                print 'Used measured variables: ',self._MeasuredList
            except:
                raise Exception('No measurable states are provided for the current model')            
        else:
            self.set_measured_states(Measurable_States)
            print 'Updated measured states are used'

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
            print 'Updated initial conditions are used'

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
            print 'Updated initial conditions are used'
              
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
                
        self.stepfunction = stepfunction
        self._has_stepfunction = True  
        self._wrote_model_to_file = False
               
        return stepfunction

    def taylor_series_approach(self, iterations, Measurable_States = False,
                             Initial_Conditions = False):
        '''Identifiability: TaylorSeriesApproach
        
        Taylor Series Approach for verification of identification.
        
        Parameters
        -----------
        iterations : int
            Number of derivatives the algorithm has to calculate (TaylorSeries!)
        Measurable_States : list or False
            if False, the previously set variables are used; otherwise this
            contains all the measured states in a list
        Initial_Conditions : Dict of False
            if False, the previously set conditions are used; otherwise this 
            dict contains initial conditions for all states

        Returns
        -------
        Identifiability_Pairwise : array
            Contains for every Measurable state and every iteration, an array
            (number parameters x number parameters), values of 1 show that this
            parameter pair is not interchangeable. Values of 0 shows that this pair is 
            interchangeable. Values of 10 can be ignored.
        
        Identifiability_Ghostparameter : array
            Contains for every Measurable state and every iteration, an array
            (1 x number parameters), a value of 1 show that this parameter is unique. 
            A value of 0 states that this parameter is not uniquely identifiable.
        
        Identifiability_Swapping : array

        Notes
        ------
        Identifiability is defined both by the so-called ghost-parameter and 
        the swap-parameter method. 
            
        References
        ----------
        .. [1] E. Walter and L. Pronzato, Identification of parametric models 
                from experimental data., 1997.
        
        See Also
        ---------
        taylor_compare_methods_check, plot_taylor_ghost
         
        '''  
        self._check_for_init(Initial_Conditions)
        self._check_for_meas(Measurable_States)

        intern_system = {}
        # Convert all parameters to symbols
        for i in range(len(self.Parameters)):
            exec(self.Parameters.keys()[i]+" = sympy.symbols('"+self.Parameters.keys()[i]+"')")
        # Add (t) to the different states in order to calculate the derivative to the time   
        for i in range(len(self.System)):
            exec(self.System.keys()[i][1:]+" = sympy.symbols('"+self.System.keys()[i][1:]+"(t)')")
        # Replace states without time by states WITH time
        for i in range(len(self.System)):
            intern_system[self.System.keys()[i]] = str(eval(self.System.values()[i]))
        # Sort internal system
        intern_system = collections.OrderedDict(sorted(intern_system.items(), key=lambda t: t[0]))
        # Symbolify t
        t = sympy.symbols('t')
        # Delete state symbols (only looking to time dependence)
        for i in range(len(self.System)):
            exec('del '+self.System.keys()[i][1:])
        # Construct empty identification matrix
        self.Identifiability_Pairwise = np.zeros([sum(self.Measurable_States.values()),iterations,len(self.Parameters),len(self.Parameters)])+10
        self.Identifiability_Ghostparameter = np.zeros([sum(self.Measurable_States.values()),iterations,len(self.Parameters)])+10
        # For all measurable states run TaylorSeriesApproac
        for h in range(sum(self.Measurable_States.values())):
            # Only perform identifiability analysis for measurable outputs
            h_measurable = np.where(np.array(self.Measurable_States.values())==1)[0][h]
            # Make list for measurable output derivatives
            Measurable_Output_Derivatives = []
            Measurable_Output_Derivatives_numerical_values = []
            # Make ghost parameter
            P_P_ghost = sympy.symbols('P_P_ghost')
            # Number of iterations = nth order-derivatives
            for i in range(iterations):
                if len(Measurable_Output_Derivatives) == 0:
                    # Copy original system in dict
                    Measurable_Output_Derivatives.append(str(intern_system['d'+self.System.keys()[h_measurable][1:]]))
                else:
                    # Take derivative of previous element of list
                    Measurable_Output_Derivatives.append(str(sympy.diff(Measurable_Output_Derivatives[-1],t)))
                for j in range(len(self.System)):
                    # Replace 'Derivative(X(t),t)' by dX(t) from system
                    Measurable_Output_Derivatives[-1] = Measurable_Output_Derivatives[-1].replace("Derivative("+self.System.keys()[j][1:]+"(t), t)",'('+intern_system['d'+self.System.keys()[j][1:]]+')')
                Measurable_Output_Derivatives_numerical_values.append(Measurable_Output_Derivatives[-1])
                for j in range(len(self.System)):
                    # Replace symbols by the corresponding numerical values
                    Measurable_Output_Derivatives_numerical_values[-1] = Measurable_Output_Derivatives_numerical_values[-1].replace(self.System.keys()[j][1:]+"(t)",str(self.Initial_Conditions[self.System.keys()[j][1:]]))
                    # Keep the symbolic values (still testing mode)                
                    #AAA[-1] = AAA[-1].replace(state_list[j]+"(t)",str(state_list[j]))
                # Simplify sympy expression
                Measurable_Output_Derivatives[-1] = str(sympy.simplify(Measurable_Output_Derivatives[-1]))
                for j in range(len(self.Parameters)):
                    for k in range(j+1,len(self.Parameters)):
                        # Exchange two symbols with each other
                        exec(self.Parameters.keys()[j]+" = sympy.symbols('"+self.Parameters.keys()[k]+"')")
                        exec(self.Parameters.keys()[k]+" = sympy.symbols('"+self.Parameters.keys()[j]+"')")
                        # Evaluate 'symbolic' expression
                        Measurable_Output_Derivatives_temp_plus = str(eval(Measurable_Output_Derivatives_numerical_values[i]))
                        # Reset symbols to their original values                    
                        exec(self.Parameters.keys()[k]+" = sympy.symbols('"+self.Parameters.keys()[k]+"')")
                        exec(self.Parameters.keys()[j]+" = sympy.symbols('"+self.Parameters.keys()[j]+"')")
                        # If answer is the same then these parameters are not identifiable
                        self.Identifiability_Pairwise[h,i,k,j] = eval(Measurable_Output_Derivatives_numerical_values[i]+' != '+Measurable_Output_Derivatives_temp_plus)
                for j in range(len(self.Parameters)):
                    # Replace parameter by ghostparameter
                    exec(self.Parameters.keys()[j]+" = sympy.symbols('P_P_ghost')")
                    # Evaluate 'symbolic' expression
                    Measurable_Output_Derivatives_temp_plus = str(eval(Measurable_Output_Derivatives_numerical_values[i]))
                    # Reset parameter to its original value                   
                    exec(self.Parameters.keys()[j]+" = sympy.symbols('"+self.Parameters.keys()[j]+"')")
                    # If answer is the same then this parameter is not unique identifiable
                    self.Identifiability_Ghostparameter[h,i,j] = eval(Measurable_Output_Derivatives_numerical_values[i]+' != '+Measurable_Output_Derivatives_temp_plus)
        self.Identifiability_Swapping = self._pairwise_to_ghoststyle(iterations)
        return self.Identifiability_Pairwise, self.Identifiability_Ghostparameter, self.Identifiability_Swapping

    def taylor_compare_methods_check(self):
        '''Taylor identifibility compare approaches
        
        Check if the ghost-parameter and swap-parameter methods are giving the 
        same result        
        '''
        check = ((self.Identifiability_Ghostparameter==self.Identifiability_Swapping)==0).sum()
        if check == 0:
            print 'Both approaches yield the same solution!'
        else:
            print 'There is an inconsistency between the Ghost and Swapping approach'
            print 'Ghostparameter'
            pprint.pprint(self.Identifiability_Ghostparameter)
            print 'Swapping'
            pprint.pprint(self.Identifiability_Swapping)

    def _pairwise_to_ghoststyle(self,iterations):
        '''Puts the output of both Taylor methods in similar output format
        
        '''
        self.Parameter_Identifiability = np.ones([sum(self.Measurable_States.values()),iterations,len(self.Parameters)])
        for h in range(sum(self.Measurable_States.values())):
            for i in range(iterations):
                for j in range(len(self.Parameters)):
                    self.Parameter_Identifiability[h,i,j] = min([min(self.Identifiability_Pairwise[h,i,j,:]),min(self.Identifiability_Pairwise[h,i,:,j])])
        return self.Parameter_Identifiability

    def plot_taylor_ghost(self, ax = 'none', order = 0, redgreen = False):
        '''Taylor identifiability plot
        
        Creates an overview plot of the identifiable parameters, given
        a certain order to show
        
        Parameters
        -----------
        ax1 : matplotlib axis instance
            the axis will be updated by the function
        order : int
            order of the taylor expansion to plot (starts with 0)
        redgreen : boolean True|False
            if True, identifibility is addressed by red/green colors, otherwise
            greyscale color is used

        Returns
        ---------
        ax1 : matplotlib axis instance
            axis with the plotted output 
                
        Examples
        ----------
        >>> M1 = odegenerator(System, Parameters, Modelname = Modelname)
        >>> fig = plt.figure()
        >>> fig.subplots_adjust(hspace=0.3)
        >>> ax1 = fig.add_subplot(211)
        >>> ax1 = M1.plot_taylor_ghost(ax1, order = 0, redgreen=True)
        >>> ax1.set_title('First order derivative')
        >>> ax2 = fig.add_subplot(212)
        >>> ax2 = M1.plot_taylor_ghost(ax2, order = 1, redgreen=True)
        >>> ax2.set_title('Second order derivative')
        
        '''
        if ax == 'none':
            fig, ax1 = plt.subplots()
        else:
            ax1 = ax
        
        
        mat_to_plot = self.Identifiability_Ghostparameter[:,order,:]
              
        xplaces=np.arange(0,mat_to_plot.shape[1],1)
        yplaces=np.arange(0,mat_to_plot.shape[0],1)
                
        if redgreen == True:
            cmap = colors.ListedColormap(['FireBrick','YellowGreen'])
        else:
            cmap = colors.ListedColormap(['.5','1.'])
            
        bounds=[0,0.9,2.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        #plot tje colors for the frist tree parameters
        ax1.matshow(mat_to_plot,cmap=cmap,norm=norm)
        
        #Plot the rankings in the matrix
        for i in range(mat_to_plot.shape[1]):
            for j in range(mat_to_plot.shape[0]):
                if mat_to_plot[j,i]== 0.:
                    ax1.text(xplaces[i], yplaces[j], '-', 
                             fontsize=14, horizontalalignment='center', 
                             verticalalignment='center')
                else:
                    ax1.text(xplaces[i], yplaces[j], '+', 
                             fontsize=14, horizontalalignment='center', 
                             verticalalignment='center')   
                             
        #place ticks and labels
        ax1.set_xticks(xplaces)
        ax1.set_xbound(-0.5,xplaces.size-0.5)
        ax1.set_xticklabels(self.Parameters.keys(), rotation = 30, ha='left')
        
        ax1.set_yticks(yplaces)
        ax1.set_ybound(yplaces.size-0.5,-0.5)
        ax1.set_yticklabels(self.get_measured_variables())
        
        ax1.spines['bottom'].set_color('none')
        ax1.spines['right'].set_color('none')
        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        
        return ax1
       
        
    def _make_canonical(self):
        '''transforms model in canonical shape
                
        '''
        print self.System.keys()
        # Symbolify parameters
        for i in range(len(self.Parameters)):
            exec(self.Parameters.keys()[i] + " = sympy.symbols('"+self.Parameters.keys()[i]+"')")
        # Symbolify states
        self._canon_A = np.zeros([len(self.System),len(self.System)])
        A_list = []
        for i in range(len(self.System)):
            for j in range(len(self.System)):
                if i is not j:
                    exec(self.System.keys()[j][1:]+"= sympy.symbols('"+self.System.keys()[j][1:]+"_eq')")
                else:
                    exec(self.System.keys()[j][1:] +" = sympy.symbols('"+self.System.keys()[j][1:]+"')")
            for j in range(len(System)):
               A_list.append(sympy.integrate(sympy.diff(eval(self.System.values()[j]),eval(self.System.keys()[i][1:])),eval(self.System.keys()[i][1:]))/eval(self.System.keys()[i][1:]))
      
        for i in range(len(self.Parameters)):
            exec(self.Parameters.keys()[i]+' = '+str(self.Parameters.values()[i]))
        for i in range(len(self.System)):
            exec(self.Initial_Conditions.keys()[i]+'_eq = '+str(self.Initial_Conditions.values()[i]))
        
        for i in range(len(self.System)):
            for j in range(len(self.System)):
                self._canon_A[i,j] = eval(str(A_list[i*len(self.System)+j]))
    
        self._canon_B = np.zeros([len(self.Measurable_States) ,sum(self.Measurable_States.values())])
        j=0
        for i in range(len(self.Measurable_States)):
            if self.Measurable_States.values()[i] == 1:
                self._canon_B[i,j]=1
                j+=1
        self._canon_C = np.transpose(self._canon_B)
        self._canon_D = np.zeros([sum(self.Measurable_States.values()),sum(self.Measurable_States.values())])
        
        return self._canon_A, self._canon_B, self._canon_C, self._canon_D


    def _identifiability_check_laplace_transform(self, Measurable_States = False, 
                              Initial_Conditions = False):
        '''Laplace transformation based identifiability test
        
        Checks the identifiability by Laplace transformation
        
        Parameters
        -----------
        Measurable_States : list or False
            if False, the previously set variables are used; otherwise this
            contains all the measured states in a list
        Initial_Conditions : Dict of False
            if False, the previously set conditions are used; otherwise this 
            dict contains initial conditions for all states
        
        Returns
        --------
        H1 : ndarray
            identifibaility array 1
        H2 : ndarray
            identifibaility array 2
        
        '''
        #check for presence of initial conditions and measured values
        self._check_for_init(Initial_Conditions)
        self._check_for_meas(Measurable_States)        

        #Make cannonical
        self._make_canonical()
        
        s = sympy.symbols('s')
        H2 = self._canon_C*((s*sympy.eye(len(self._canon_A))-self._canon_A).inv())
        H1 = H2*self._canon_B+self._canon_D
        
        return H1,H2

    def _write_model_to_file(self, procedure = 'ode'):
        '''Write derivative of model as definition in file
        
        Writes a file with a derivative definition to run the model and
        use it for other applications
        
        Parameters
        -----------
        
        '''
                
        try:
            self.dfdtheta
        except:
            print 'Running symbolic calculation of analytic sensitivity ...'
            self._analytic_local_sensitivity()
            print '... Done!'
            
        temp_path = os.path.join(os.getcwd(),self.modelname+'.py')
        print 'File is printed to: ', temp_path
        print 'Filename used is: ', self.modelname
        file = open(temp_path, 'w+')
        file.seek(0,0)
        
        file.write('#'+self.modelname+'\n')
        file.write('from __future__ import division\n')
        file.write('from numpy import *\n\n')
        
        # Write function for solving ODEs only
        if self._has_stepfunction:
            if self._ode_procedure == "ode":
                file.write('def system(t,ODES,Parameters,stepfunction):\n')
            else:
                file.write('def system(ODES,t,Parameters,stepfunction):\n')
        else:
            if self._ode_procedure == "ode":
                file.write('def system(t,ODES,Parameters):\n')
            else:
                file.write('def system(ODES,t,Parameters):\n')
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
        file.write('\n')
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
        file.write('\n')  
        try:
            self.stepfunction
            for i, step in enumerate(self.stepfunction):
                file.write('    step'+str(i) + ' = stepfunction['+str(i)+'](t)'+'\n')
            file.write('\n')
        except AttributeError:
            pass
        try:
            self.Algebraic
            for i in range(len(self.Algebraic)):
                #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n')
            file.write('\n')
        except AttributeError:
            pass
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n')
        
        file.write('    return '+str(self.System.keys()).replace("'","")+'\n\n\n')
        
        # Write function for solving ODEs of both system and analytical sensitivities
        if self._has_stepfunction:
            if self._ode_procedure == "ode":
                file.write('def system_with_sens(t,ODES,Parameters,stepfunction):\n')
            else:
                file.write('def system_with_sens(ODES,t,Parameters,stepfunction):\n')
        else:
            if self._ode_procedure == "ode":
                file.write('def system_with_sens(t,ODES,Parameters):\n')
            else:
                file.write('def system_with_sens(ODES,t,Parameters):\n')
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
        file.write('\n')
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
        file.write('\n')
        try:
            self.stepfunction
            for i, step in enumerate(self.stepfunction):
                file.write('    step'+str(i) + ' = stepfunction['+str(i)+'](t)'+'\n')
            file.write('\n')
        except AttributeError:
            pass
        try:
            self.Algebraic
            for i in range(len(self.Algebraic)):
                #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n')
            file.write('\n')
        except AttributeError:
            pass
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n')
        
        print 'Sensitivities are printed to the file....'
        file.write('\n    #Sensitivities\n\n')
        
        # Calculate number of states by using inputs
        file.write('    state_len = len(ODES)/(len(Parameters)+1)\n')
        # Reshape ODES input to array with right dimensions in order to perform matrix multiplication
        file.write('    dxdtheta = array(ODES[state_len:].reshape(state_len,len(Parameters)))\n\n')
        
        # Write dfdtheta as symbolic array
        file.write('    dfdtheta = ')
        pprint.pprint(self.dfdtheta,file)
        # Write dfdx as symbolic array
        file.write('\n    dfdx = ')
        pprint.pprint(self.dfdx,file)
        # Calculate derivative in order to integrate this
        file.write('\n    dxdtheta = dfdtheta + dot(dfdx,dxdtheta)\n')

        file.write('    return '+str(self.System.keys()).replace("'","")+'+ list(dxdtheta.reshape(-1,))'+'\n\n\n')
        
        try:
            self.Algebraic
            if self._has_stepfunction:
                file.write('\ndef Algebraic_outputs(ODES,t,Parameters, stepfunction):\n')
            else:
                file.write('\ndef Algebraic_outputs(ODES,t,Parameters):\n')
            for i in range(len(self.Parameters)):
                #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
            file.write('\n')
            for i in range(len(self.System)):
                #file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
                file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES[:,'+str(i)+']\n')
            file.write('\n')
            try:
                self.stepfunction
                for i, step in enumerate(self.stepfunction):
                    file.write('    step'+str(i) + ' = stepfunction['+str(i)+'](t)'+'\n')
                file.write('\n')
            except AttributeError:
                pass
            if self.Algebraic != None:
                for i in range(len(self.Algebraic)):
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+' + zeros(len(t))\n')
                    file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+' + zeros(len(t))\n')
            file.write('\n')
            file.write('    algebraic = array('+str(self.Algebraic.keys()).replace("'","")+').T\n\n')
            file.write('    return algebraic\n\n\n')
            #Algebraic sens
            if self._has_stepfunction:
                file.write('\ndef Algebraic_sens(ODES,t,Parameters, stepfunction, dxdtheta):\n')
            else:
                file.write('\ndef Algebraic_sens(ODES,t,Parameters, dxdtheta):\n')
            for i in range(len(self.Parameters)):
                #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
            file.write('\n')
            for i in range(len(self.System)):
                file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
            file.write('\n')
            try:
                self.stepfunction
                for i, step in enumerate(self.stepfunction):
                    file.write('    step'+str(i) + ' = stepfunction['+str(i)+'](t)'+'\n')
                file.write('\n')
            except AttributeError:
                pass
            if self.Algebraic != None:
                for i in range(len(self.Algebraic)):
                    #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                    file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n')
            file.write('\n')
            print 'Sensitivities are printed to the file....'
            file.write('\n    #Sensitivities\n\n')
                       
            # Write dgdtheta as symbolic array
            file.write('    dgdtheta = ')
            pprint.pprint(self.dgdtheta,file)
            # Write dgdx as symbolic array
            file.write('\n    dgdx = ')
            pprint.pprint(self.dgdx,file)
#            file.write('    dgdtheta = zeros('+str([self._TimeDict['nsteps'],self.dgdtheta.shape[0],self.dgdtheta.shape[1]])+')')
#            for i,dg in enumerate(self.dgdtheta):
#                for j,dg2 in enumerate(dg):
#                    file.write('\n    dgdtheta[:,'+str(i)+','+str(j)+'] = ' + str(self.dgdtheta[i,j]))
            # Write dgdx as symbolic array
            
#            file.write('\n\n    dgdx = zeros('+str([self._TimeDict['nsteps'],self.dgdx.shape[0],self.dgdx.shape[1]])+')')
#            for i,dg in enumerate(self.dgdx):
#                for j,dg2 in enumerate(dg):
#                    file.write('\n    dgdx[:,'+str(i)+','+str(j)+'] = ' + str(self.dgdx[i,j]))
            # Calculate derivative in order to integrate this
            file.write('\n\n    dydtheta = dgdtheta + dot(dgdx,dxdtheta)\n')
    
            file.write('    return dydtheta'+'\n\n\n')
        
        except AttributeError:
            pass
        file.close()
        print '...done!'
        
    def rerun_for_algebraic(self):
        """
        """
        try:
            os.remove(self.modelname + '.pyc')
        except:
            pass
        exec('import ' + self.modelname)
        exec(self.modelname+' = reload('+self.modelname+')')

        if self._has_stepfunction:
            algeb_out = eval(self.modelname+'.Algebraic_outputs'+'(np.array(self.ode_solved), self._Time, self.Parameters, self.stepfunction)')
        else:
            algeb_out = eval(self.modelname+'.Algebraic_outputs'+'(np.array(self.ode_solved), self._Time, self.Parameters)')
 
        self.algeb_solved = pd.DataFrame(algeb_out, columns=self.Algebraic.keys(), 
                                 index = self.ode_solved.index)
                                 
    def calcAlgLSA(self):
        """
        """
        try:
            self._ana_sens_matrix
        except:
            raise Exception('First run self.analytical_local_sensitivity()!')
        
        exec('import ' + self.modelname)
        
        algeb_out = np.empty((self.ode_solved.index.size, len(self.Algebraic.keys()) ,len(self.Parameters)))         
        
        if self._has_stepfunction:
            for i,timestep in enumerate(self.ode_solved.index):
                algeb_out[i,:,:] = eval(self.modelname+'.Algebraic_sens'+'(np.array(self.ode_solved.ix[timestep]), timestep, self.Parameters, self.stepfunction, self._ana_sens_matrix[i,:,:])')
        else:
            for i,timestep in enumerate(self.ode_solved.index):
                algeb_out[i,:,:] = eval(self.modelname+'.Algebraic_sens'+'(np.array(self.ode_solved.ix[timestep]), timestep, self.Parameters, self._ana_sens_matrix[i,:,:])')
                
        alg_dict = {}
        
        for i,key in enumerate(self.Algebraic.keys()):
            alg_dict[key] = pd.DataFrame(algeb_out[:,i,:], columns=self.Parameters.keys(), 
                                 index = self.ode_solved.index)
        
        self.getAlgLSA = alg_dict

    def solve_ode(self, TimeStepsDict = False, Initial_Conditions = False, 
                  plotit = True, with_sens = False, procedure = "odeint", write = False):
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
        #        print 'Current parameters', self.Parameters.values()
        self._check_for_time(TimeStepsDict)
        self._check_for_init(Initial_Conditions)
        
        if (self._wrote_model_to_file and self._ode_procedure is not procedure) or \
                (self._wrote_model_to_file is False) or (write):
            print "Writing model to file for '" + procedure + "' procedure..."
            self._ode_procedure = procedure
            self._write_model_to_file(procedure = procedure)
            self._wrote_model_to_file = True
            print '...Finished writing to file!'
        else:
            print "Model was already written to file! We are using the '" + \
                procedure + "' procedure for solving ODEs. If you want to rewrite \
                the model to the file, please add 'write = True'."
        
        try:
            os.remove(self.modelname + '.pyc')
        except:
            pass
        exec('import ' + self.modelname)
        exec('reload('+self.modelname+')')
        print(eval(self.modelname+'.system_with_sens'))
        	
        if with_sens == False:
            if procedure == "odeint": 
                print "Going for odeint..."
                if self._has_stepfunction:
                    res = spin.odeint(eval(self.modelname+'.system'),self.Initial_Conditions.values(), self._Time,args=(self.Parameters,self.stepfunction,))
                else:
                    res = spin.odeint(eval(self.modelname+'.system'),self.Initial_Conditions.values(), self._Time,args=(self.Parameters,))
                #put output in pandas dataframe
                df = pd.DataFrame(res, index=self._Time, columns = self._Variables)                
    
    
            elif procedure == "ode":
                print "Going for generic methodology..."
                #ode procedure-generic
                #                f = eval(self.modelname+'.system')
                r = spin.ode(eval(self.modelname+'.system')).set_integrator('lsoda') #   method='bdf', with_jacobian = False
                    #                                                    nsteps = 300000)
                r.set_initial_value(self.Initial_Conditions.values(), 0).set_f_params(self.Parameters)
                dt = (self._TimeDict['end']-self._TimeDict['start'])/self._TimeDict['nsteps'] #needs genereic version
                moutput = []
                toutput = []
                end_val = self._Time[-1]-self._TimeDict['end']/(10*self._TimeDict['nsteps'])             
                print "Starting ODE loop..."
                while r.successful() and r.t < end_val:          
                    r.integrate(r.t + dt)
                    #print "This integration worked well?", r.successful()
                    #print("%g %g" % (r.t, r.y[0]))
                    #                    print "t: ", r.t
                    #                    print "y: ", r.y
    
                    moutput.append(r.y)
                    toutput.append(r.t)
                print "...Done!"
                
                #make df
                df = pd.DataFrame(moutput, index = toutput, 
                                  columns = self._Variables)          
        else:
            #odeint procedure
            if procedure == "odeint":
                print "Going for odeint..."
                if self._has_stepfunction:
                    res = spin.odeint(eval(self.modelname+'.system_with_sens'), np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), self._Time,args=(self.Parameters,self.stepfunction,))
                else:
                    res = spin.odeint(eval(self.modelname+'.system_with_sens'), np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), self._Time,args=(self.Parameters,))
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
                print "Going for generic methodology..."
                #ode procedure-generic
                r = spin.ode(eval(self.modelname+'.system_with_sens'))
                r.set_integrator('vode', method='bdf', with_jacobian = False)
                if self._has_stepfunction:
                    r.set_initial_value(np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), 0)
                    r.set_f_params(self.Parameters,self.stepfunction)                    
                else:
                    r.set_initial_value(np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), 0)
                    r.set_f_params(self.Parameters)
                dt = (self._TimeDict['end']-self._TimeDict['start'])/self._TimeDict['nsteps']
                moutput = []
                toutput = []
                end_val = self._Time[-1]-self._TimeDict['end']/(10*self._TimeDict['nsteps'])
                print "Starting ODE loop..."
                while r.successful() and r.t < end_val:
                    r.integrate(r.t+dt)
#                    print("%g %g" % (r.t, r.y))
                    moutput.append(r.y)
                    toutput.append(r.t)
                print "...Done!"
                
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
            print "If you want the algebraic equations also, please rerun manually\
             by using the 'self.rerun_for_algebraic()' function!"
        
            #self.rerun_for_algebraic()

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
            print 'Running Analytical sensitivity analysis'
            self.analytic_local_sensitivity(Sensitivity = 'CTRS')
            print '... Done!'

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
        self.LSA_type = Sensitivity

        df, analytical_sens = self.solve_ode(with_sens = True, plotit = False, procedure = procedure)
        
        if Sensitivity == 'CPRS':
            #CPRS = CAS*parameter
            for i in self._Variables:
                 analytical_sens[i] = analytical_sens[i]*self.Parameters.values()
        elif Sensitivity == 'CTRS':
            #CTRS
            if min(df.mean()) == 0 or max(df.mean()) == 0:
                self.LSA_type = None
                raise Exception('ANASENS: It is not possible to use the CTRS method for\
                    calculating sensitivity, because one or more variables are\
                    fixed at zero. Try to use another method or to change the\
                    initial conditions!')
            elif min(df.min()) == 0 or max(df.max()) == 0:
                print 'ANASENS: Using AVERAGE of output values'
                for i in self._Variables:
                     analytical_sens[i] = analytical_sens[i]*self.Parameters.values()/df[i].mean()
            else:
                print 'ANASENS: Using EVOLUTION of output values'
                for i in self._Variables:
                     analytical_sens[i] = analytical_sens[i]*self.Parameters.values()/np.tile(np.array(df[i]),(len(self._Variables),1)).T
        elif Sensitivity != 'CAS':
            self.LSA_type = None
            raise Exception('You have to choose one of the sensitivity\
             methods which are available: CAS, CPRS or CTRS')
        
        self.analytical_sensitivity = analytical_sens

        print 'ANASENS: The ' + Sensitivity + ' sensitivity method is used, do not\
                forget to check whether outputs can be compared!'
        
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
            
            #we use now CPRS, but later on we'll adapt to CTRS
            if Sensitivity == 'CPRS':
                #CPRS = CAS*parameter
                sensitivity_out = sensitivity_out*value2save
            elif Sensitivity == 'CTRS':
                #CTRS
                average_out = (modout_plus+modout_min)/2.
                if min(abs(average_out.mean())) < 1e-10:
                    self.LSA_type = None
                    raise Exception('NUMSENS: It is not possible to use the CTRS method for\
                        calculating sensitivity, because one or more variables are\
                        fixed at zero. Try to use another method or to change the\
                        initial conditions!')
                elif min(average_out.abs().min()) < 1e-10:
                    if i==0:
                        print 'NUMSENS: Using AVERAGE of output values'
                    sensitivity_out = sensitivity_out*value2save/average_out.mean()
                else:
                    if i==0:
                        print 'NUMSENS: Using EVOLUTION of output values'
                    sensitivity_out = sensitivity_out*value2save/average_out
            elif Sensitivity != 'CAS':
                self.LSA_type = None
                raise Exception('You have to choose one of the sensitivity\
                 methods which are available: CAS, CPRS or CTRS')
            
            #put on the rigth spot in the dictionary
            for var in self._Variables:
                numerical_sens[var][parameter] = sensitivity_out[var][:].copy()
               
            #put back original value
            self.Parameters[parameter] = value2save
        print 'NUMSENS: The ' + Sensitivity + ' sensitivity method is used, do not\
                forget to check whether outputs can be compared!'
        self.numerical_sensitivity = numerical_sens

        return numerical_sens

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
                print 'Collinearity check of variable '+self._Variables[i]+' is now performed!' 
                mat_to_plot[i,:,:] = self.collinearity_check(self._Variables[i])
            
        textlist = {}
        #Plot the rankings in the matrix
        for i in range(len(self._Variables)):
            F = (mat_to_plot[i] <400.)*(mat_to_plot[i] >0.)
            print F
            for j in range(len(self.Parameters)):
                for k in range(len(self.Parameters)):
                    if F[j,k] == True:
                        print j,k
                        print F[j,k]
                        try:
                            print textlist[(self._Variables[i],self.Parameters.keys()[j])]
                            textlist[(self._Variables[i],self.Parameters.keys()[j])]=textlist[(self._Variables[i],self.Parameters.keys()[j])]+','+(self.Parameters.keys()[k])   
                        except:
                            textlist[(self._Variables[i],self.Parameters.keys()[j])]=self.Parameters.keys()[k]  
                        try:
                            print textlist[(self._Variables[i],self.Parameters.keys()[k])]
                            textlist[(self._Variables[i],self.Parameters.keys()[k])]=textlist[(self._Variables[i],self.Parameters.keys()[k])]+','+(self.Parameters.keys()[j])   
                        except:
                            textlist[(self._Variables[i],self.Parameters.keys()[k])]=self.Parameters.keys()[j]  

        print textlist
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
        QSSE_enz = self._solve_linear_system(linear_system[1:,:], list(enzyme_forms))   
        
        # Replace enzyme forms by its QSSE in rate equation of variable of interest
        QSSE_var = sympy.sympify(self.System['d' + variable])
        for enz in enzyme_forms:
            QSSE_var = QSSE_var.replace(enz,QSSE_enz[enz])
        
        # To simplify output expand all terms (=remove brackets) and afterwards
        # simplify the equation
        QSSE_var = sympy.simplify(sympy.expand(QSSE_var))
        
        self.QSSE_var = QSSE_var
        self.QSSE_enz = QSSE_enz
               
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
        
        if massBalance == 0:
            print "The mass balance is closed!"
        else:
            print "The mass balance is NOT closed for the enzyme forms of '" + variables +"'! \
                The following term(s) cannot be striked out: " + str(massBalance)
        
        return massBalance

    def _solve_linear_system(self, system, symbols, **flags):
        r"""
        Solve system of N linear equations with M variables, which means
        both under- and overdetermined systems are supported. The possible
        number of solutions is zero, one or infinite. Respectively, this
        procedure will return None or a dictionary with solutions. In the
        case of underdetermined systems, all arbitrary parameters are skipped.
        This may cause a situation in which an empty dictionary is returned.
        In that case, all symbols can be assigned arbitrary values.
    
        Input to this functions is a Nx(M+1) matrix, which means it has
        to be in augmented form. If you prefer to enter N equations and M
        unknowns then use `solve(Neqs, *Msymbols)` instead. Note: a local
        copy of the matrix is made by this routine so the matrix that is
        passed will not be modified.
    
        The algorithm used here is fraction-free Gaussian elimination,
        which results, after elimination, in an upper-triangular matrix.
        Then solutions are found using back-substitution. This approach
        is more efficient and compact than the Gauss-Jordan method.
    
        >>> from sympy import Matrix, solve_linear_system
        >>> from sympy.abc import x, y
    
        Solve the following system::
    
               x + 4 y ==  2
            -2 x +   y == 14
    
        >>> system = Matrix(( (1, 4, 2), (-2, 1, 14)))
        >>> solve_linear_system(system, x, y)
        {x: -6, y: 2}
    
        A degenerate system returns an empty dictionary.
    
        >>> system = Matrix(( (0,0,0), (0,0,0) ))
        >>> solve_linear_system(system, x, y)
        {}
    
        """
        matrix = system[:, :]
        syms = symbols
        i, m = 0, matrix.cols - 1  # don't count augmentation
    
        while i < matrix.rows:
            if i == m:
                # an overdetermined system
                if any(matrix[i:, m]):
                    return None   # no solutions
                else:
                    # remove trailing rows
                    matrix = matrix[:i, :]
                    break
    
            if not matrix[i, i]:
                # there is no pivot in current column
                # so try to find one in other columns
                for k in xrange(i + 1, m):
                    if matrix[i, k]:
                        break
                else:
                    if matrix[i, m]:
                        # we need to know if this is always zero or not. We
                        # assume that if there are free symbols that it is not
                        # identically zero (or that there is more than one way
                        # to make this zero. Otherwise, if there are none, this
                        # is a constant and we assume that it does not simplify
                        # to zero XXX are there better ways to test this?
                        if not matrix[i, m].free_symbols:
                            return None  # no solution
    
                        # zero row with non-zero rhs can only be accepted
                        # if there is another equivalent row, so look for
                        # them and delete them
                        nrows = matrix.rows
                        rowi = matrix.row(i)
                        ip = None
                        j = i + 1
                        while j < matrix.rows:
                            # do we need to see if the rhs of j
                            # is a constant multiple of i's rhs?
                            rowj = matrix.row(j)
                            if rowj == rowi:
                                matrix.row_del(j)
                            elif rowj[:-1] == rowi[:-1]:
                                if ip is None:
                                    _, ip = rowi[-1].as_content_primitive()
                                _, jp = rowj[-1].as_content_primitive()
                                if not (simplify(jp - ip) or simplify(jp + ip)):
                                    matrix.row_del(j)
    
                            j += 1
    
                        if nrows == matrix.rows:
                            # no solution
                            return None
                    # zero row or was a linear combination of
                    # other rows or was a row with a symbolic
                    # expression that matched other rows, e.g. [0, 0, x - y]
                    # so now we can safely skip it
                    matrix.row_del(i)
                    if not matrix:
                        # every choice of variable values is a solution
                        # so we return an empty dict instead of None
                        return dict()
                    continue
    
                # we want to change the order of colums so
                # the order of variables must also change
                syms[i], syms[k] = syms[k], syms[i]
                matrix.col_swap(i, k)
    
            pivot_inv = S.One/matrix[i, i]
    
            # divide all elements in the current row by the pivot
            matrix.row_op(i, lambda x, _: x * pivot_inv)
    
            for k in xrange(i + 1, matrix.rows):
                if matrix[k, i]:
                    coeff = matrix[k, i]
    
                    # subtract from the current row the row containing
                    # pivot and multiplied by extracted coefficient
                    matrix.row_op(k, lambda x, j: simplify(x - matrix[i, j]*coeff))
    
            i += 1
    
        # if there weren't any problems, augmented matrix is now
        # in row-echelon form so we can check how many solutions
        # there are and extract them using back substitution
    
        do_simplify = flags.get('simplify', True)
    
        if len(syms) == matrix.rows:
            # this system is Cramer equivalent so there is
            # exactly one solution to this system of equations
            k, solutions = i - 1, {}
    
            while k >= 0:
                content = matrix[k, m]
    
                # run back-substitution for variables
                for j in xrange(k + 1, m):
                    content -= matrix[k, j]*solutions[syms[j]]
    
                if do_simplify:
                    solutions[syms[k]] = simplify(content)
                else:
                    solutions[syms[k]] = content
    
                k -= 1
    
            return solutions
        elif len(syms) > matrix.rows:
            # this system will have infinite number of solutions
            # dependent on exactly len(syms) - i parameters
            k, solutions = i - 1, {}
    
            while k >= 0:
                content = matrix[k, m]
    
                # run back-substitution for variables
                for j in xrange(k + 1, i):
                    content -= matrix[k, j]*solutions[syms[j]]
    
                # run back-substitution for parameters
                for j in xrange(i, m):
                    content -= matrix[k, j]*syms[j]
    
                if do_simplify:
                    solutions[syms[k]] = simplify(content)
                else:
                    solutions[syms[k]] = content
    
                k -= 1
    
            return solutions
        else:
            return []   # no solutions
        
        
        
        
            

        
