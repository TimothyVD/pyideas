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
    >>> Parameters = {'k1':1/10,'k1m':1/20,
              'k2':1/20,'k2m':1/20,
              'k3':1/200,'k3m':1/175,
              'k4':1/200,'k4m':1/165}
    >>> System =    {'dEn':'k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ',
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
    
    def __init__(self, System, Parameters, Modelname = 'MyModel'):
        '''

        '''
        
        self.Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))
        self.System = collections.OrderedDict(sorted(System.items(), key=lambda t: t[0]))    
        self.modelname = Modelname        
        
        self._Variables = [i[1:] for i in self.System.keys()]
        
        self._analytic_local_sensitivity()
        self._write_model_to_file()

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
        model sensitivities.
        
        Returns
        --------
        TODO
        
        Notes
        ------
        
        The output is best viewed by the write_model_to_file method        
        
        See Also
        ---------
        _write_model_to_file
        
        '''
        # t is always the symbol for time and is initialised here
        t = sympy.symbols('t')        
        
        # Symbolify system states
        for i in range(len(self.System)):
            exec(self.System.keys()[i][1:]+" = sympy.symbols('"+self.System.keys()[i][1:]+"')")

        # Symbolify parameters
        for i in range(len(self.Parameters)):
            exec(self.Parameters.keys()[i]+" = sympy.symbols('"+self.Parameters.keys()[i]+"')")
                    
        self._system_matrix = sympy.zeros(len(self.System),1)
        self._states_matrix = sympy.zeros(len(self._Variables),1)
        
        # Set up symbolic matrix of ODE system and states
        for i in range(len(self.System)):
            self._system_matrix[i,0] = eval(self.System.values()[i])
            self._states_matrix[i,0] = eval(self._Variables[i])
        
        self._parameter_matrix = sympy.zeros(len(self.Parameters),1)
        # Set up symbolic matrix of parameters
        for i in range(len(self.Parameters)):
            self._parameter_matrix[i,0] = eval(self.Parameters.keys()[i])
        
        # Initialize and calculate matrices for analytic sensitivity calculation
        # dfdtheta
        dfdtheta = self._system_matrix.jacobian(self._parameter_matrix)
        self.dfdtheta = np.array(dfdtheta)
        # fdx
        dfdx = self._system_matrix.jacobian(self._states_matrix)
        self.dfdx = np.array(dfdx)
        # dxdtheta
        dxdtheta = np.zeros([len(self._states_matrix),len(self.Parameters)])
        self.dxdtheta = np.asmatrix(dxdtheta)
        
#        #dgdtheta
#        dgdtheta = np.zeros([sum(self.Measurable_States.values()),len(self.Parameters)])
#        self.dgdtheta = np.array(dgdtheta)
#        #dgdx
#        dgdx = np.eye(len(self.states_matrix))*self.Measurable_States.values()
#        #Remove zero rows
#        self.dgdx = np.array(dgdx[~np.all(dgdx == 0, axis=1)])
        
        
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

    def _write_model_to_file(self):
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
        
        file.write('from numpy import *\n\n')
        
        # Write function for solving ODEs only
        file.write('def system(ODES,t,Parameters):\n')
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
        file.write('\n')
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
        file.write('\n')    
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n')
        
        file.write('    return '+str(self.System.keys()).replace("'","")+'\n\n\n')
        
        # Write function for solving ODEs of both system and analytical sensitivities
        file.write('def system_with_sens(ODES,t,Parameters):\n')
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
        file.write('\n')
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n')
        file.write('\n')    
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

        file.write('    return '+str(self.System.keys()).replace("'","")+'+ list(dxdtheta.reshape(-1,))'+'\n')
                
        file.close()
        print '...done!'

    def solve_ode(self, TimeStepsDict = False, Initial_Conditions = False, 
                  plotit = True, with_sens = False):
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
        
        #        import MODEL_Halfreaction
        exec('import '+self.modelname)
        exec(self.modelname + ' = ' + 'reload('+self.modelname+')')
        	
        if with_sens == False:
            res = spin.odeint(eval(self.modelname+'.system'),self.Initial_Conditions.values(), self._Time,args=(self.Parameters,))
            #put output in pandas dataframe
            df = pd.DataFrame(res, index=self._Time, columns = self._Variables)
        else:
            res = spin.odeint(eval(self.modelname+'.system_with_sens'), np.hstack([np.array(self.Initial_Conditions.values()),np.asarray(self.dxdtheta).flatten()]), self._Time,args=(self.Parameters,))
            #put output in pandas dataframe
            df = pd.DataFrame(res[:,0:len(self._Variables)], index=self._Time,columns = self._Variables)
            analytical_sens = {}
            for i in range(len(self._Variables)):
                #Comment was bug!
                #analytical_sens[self._Variables[i]] = pd.DataFrame(res[:,len(self._Variables)*(1+i):len(self._Variables)*(1+i)+len(self.Parameters)], index=self._Time,columns = self.Parameters.keys())
                analytical_sens[self._Variables[i]] = pd.DataFrame(res[:,len(self._Variables)+len(self.Parameters)*(i):len(self._Variables)+len(self.Parameters)*(1+i)], index=self._Time,columns = self.Parameters.keys())
            self.analytical_sensitivity = analytical_sens        
            
        #plotfunction
        if plotit == True:
            if len(self._Variables) == 1:
                df.plot(subplots = False)
            else:
                df.plot(subplots = True)
               
        self.ode_solved = df
               
        return df
        
    def collinearity_check(self,variable):
        '''
        
        Change to total relative sensitivity instead of relative sensitivity 
        to parameter
        
        TODO adapt to new analyical sens, will possibly be deleted?
        '''
        try:
            self.analytic_sens
        except:
            print 'Running Analytical sensitvity analysis'
            self.analytic_local_sensitivity()
            print '... Done!'
        
        # Make matrix for storing collinearities per two parameters
        Collinearity_pairwise = np.zeros([len(self.Parameters),len(self.Parameters)])
        for i in range(len(self.Parameters)):
            for j in range(i+1,len(self.Parameters)):
                # Transpose is performed on second array because by selecting only one column, python automatically converts the column to a row!
                # Klopt enkel voor genormaliseerde sensitiviteiten                
                #Collinearity_index_pairwise[i,j] = np.sqrt(1/min(np.linalg.eigvals(np.array(np.matrix(np.vstack([self.analytic_sens[variable][self.Parameters.keys()[i]],self.analytic_sens[variable][self.Parameters.keys()[j]]]))*(np.vstack([self.analytic_sens[variable][self..Parameters.keys()[i]],self.analytic_sens[variable][self.Parameters.keys()[j]])).transpose()))))
                print 'Nothing interesting to calculate!'
        x = pd.DataFrame(Collinearity_pairwise, index=self.Parameters.keys(), columns = self.Parameters.keys())
        
        try:
            self.Collinearity_Pairwise[variable] = x
        except:
            self.Collinearity_Pairwise = {}
            self.Collinearity_Pairwise[variable] = x
        
        
        
    def analytic_local_sensitivity(self):
        self.solve_ode(with_sens = True, plotit = False)

    def numeric_local_sensitivity(self, perturbation_factor = 0.0001, 
                                  TimeStepsDict = False, 
                                  Initial_Conditions = False):
        '''Calculates numerical based local sensitivity 
        
        For every parameter calcluate the sensitivity of the output variables.
        
        Parameters
        -----------
        perturbation_factor : float (default 0.0001)
            factor for perturbation of the parameter value to approximate 
            the derivative
        TimeStepsDict : False|dict
            If False, the time-attribute is checked for and used. 
        Initial_Conditions : False|dict
            If False, the initial conditions  attribute is checked for and used.         
        
        Returns
        --------
        numerical_sens : dict
            each variable gets a t timesteps x k par DataFrame
            
        '''
        self._check_for_time(TimeStepsDict)
        self._check_for_init(Initial_Conditions) 
        
        #create a dictionary with everye key the variable and the values a dataframe
        numerical_sens = {}
        for key in self._Variables:
            #belangrijk dat deze dummy in loop wordt geschreven!
            dummy = np.empty((self._Time.size,len(self.Parameters)))
            numerical_sens[key] = pd.DataFrame(dummy, index=self._Time, columns = self.Parameters.keys())
        
        for parameter in self.Parameters:
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
            CAS = (modout_plus-modout)/(perturbation_factor*value2save) #dy/dp
            
            #we use now CPRS, but later on we'll adapt to CTRS
#            CPRS = CAS*value2save    
#            average_out = (modout_plus+modout_min)/2.
#            CTRS = CAS*value2save/average_out
            
            #put on the rigth spot in the dictionary
            for var in self._Variables:
#                numerical_sens[var][parameter] = CPRS[var][:].copy()
#                numerical_sens[var][parameter] = CTRS[var][:].copy()
                numerical_sens[var][parameter] = CAS[var][:].copy()
                
            #put back original value
            self.Parameters[parameter] = value2save
            
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
            

        
