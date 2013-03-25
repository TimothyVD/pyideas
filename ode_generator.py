# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:04:03 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator by Tvandaele
"""
from __future__ import division
import numpy as np
import scipy.integrate as spin
import sympy
import ordereddict as collections
import os
import pandas as pd



class odegenerator(object):
    '''
    
    '''
    
    def __init__(self, System, Parameters, Modelname = 'MyModel'):
        '''
        
        Parameters
        ------------
        System : OrderedDict
            Ordered dict with the keys as the derivative of a state (written as 'd'+State),
            the values of the dictionary is the ODE system written as a string
        Parameters : OrderedDict
            Ordered dict with parameter names as keys, parameter values are the values
            of the dictionary
        
        '''
        
        self.Parameters = collections.OrderedDict(sorted(Parameters.items(), key=lambda t: t[0]))
        self.System = collections.OrderedDict(sorted(System.items(), key=lambda t: t[0]))    
        self.modelname = Modelname        
        
        self._Variables = [i[1:] for i in M1.System.keys()] 


    def set_time(self,timedict):
        '''
        
        Parameters
        -----------
        timedict : dict
            three elments
        '''
        if timedict['start'] > timedict['end']:
            raise Exception('End timestep must be smaller then start!')
        if timedict['nsteps'] < (timedict['end'] - timedict['start']):
            raise Exception('Step too small')
        
        self._TimeDict = timedict
        self._Time = np.linspace(timedict['start'], timedict['end'], timedict['nsteps'])

    def set_initial_conditions(self, inic):
        '''
        eg inic = {'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.}
        '''
        self.Initial_Conditions = collections.OrderedDict(sorted(inic.items(), key=lambda t: t[0]))

    def set_measured_states(self, Measurable_States):
        '''
        Measurable_States -> list instead of dict; returns ordered dict
        ['En', 'Es']        
        
        eg Measurable_States = {'En':0,'Es':0,'SA':1,'SB':1,'PP':1,'PQ':1,'EsQ':0,'EP':0}
        '''
        Measured_temp = {}
        for key in self.System:
            Measured_temp[key] = 0
        
        self._MeasuredList=[]
        
        for measured in Measurable_States:
            dmeasured = 'd' + measured
            if dmeasured in  Measured_temp:
                Measured_temp[dmeasured] = 1
                self._MeasuredList.append(measured)
            
            else:
                raise Exception('The variable',measured,'is not part of the current model.')

        self.Measurable_States = collections.OrderedDict(sorted(Measured_temp.items(), key=lambda t: t[0]))

    def get_variables(self):
        '''
        '''
        print self._Variables

    def get__measured_variables(self):
        '''
        Help function for getting the values one can measure in the lab
        '''
        print self._MeasuredList

    def get__time(self):
        '''
        '''
        print 'start timestep is ', self._TimeDict['start']
        print 'end timestep is ', self._TimeDict['end']
        print 'number of timesteps for printing is ', self._TimeDict['nsteps']
        
    def analytic_local_sensitivity(self):
        '''
        Analytic derivation of the local sensitivities
        
        TODO: help!
        '''
        
        # Symbolify system states
        for i in range(len(self.System)):
            exec(self.System.keys()[i][1:]+" = sympy.symbols('"+self.System.keys()[i][1:]+"')")
        # Symbolify parameters
        for i in range(len(self.Parameters)):
            exec(self.Parameters.keys()[i]+" = sympy.symbols('"+self.Parameters.keys()[i]+"')")
        
        # Make empty list for saving symbolic sensitivity formulas
        self.Sensitivity_list = []
        # Make empty list for saving combined symbols corresponding with sensitivity
        self.Sensitivity_symbols = []
        
        # Calculate direct and indirect sensitivities
        for j in range(self.System.__len__()+1):
            for i in range(self.System.__len__()):
                for k in range(len(self.Parameters)):
                    ##Evaluation of the system
                    self.Sensitivity_list.append(str(eval(self.System.values()[i])))
                    # Symbolic derivative of the system to a certain parameter
                    if j ==0:
                        # Make symbol for pythonscript
                        self.Sensitivity_symbols.append(self.System.keys()[i]+'d'+self.Parameters.keys()[k])
                        # Perform partial derivation to certian parameter
                        self.Sensitivity_list[-1] = sympy.diff(self.Sensitivity_list[-1],eval(self.Parameters.keys()[k]))
                    # First replace certain state by its derivative and then perform partial derivative to specific parameter
                    else:
                        # Make symbol for pythonscript
                        self.Sensitivity_symbols.append(self.System.keys()[i]+self.System.keys()[j-1]+'X'+self.System.keys()[j-1]+'d'+self.Parameters.keys()[k])
                        # Replace state by its derivative
                        exec(self.System.keys()[j-1][1:]+" = sympy.symbols('("+self.System.values()[j-1].replace(" ","")+")')")
                        # Evaluate                    
                        self.Sensitivity_list[-1] = eval(str(self.Sensitivity_list[-1]))
                        #temp = sympy.diff(temp,eval(System.keys()[j-1].replace("d","")))*sympy.diff(eval(System[System.keys()[j-1]]),eval(parameter_list[k]))
                        # Reset state to its original symbolic representation                    
                        exec(self.System.keys()[j-1][1:]+" = sympy.symbols('"+self.System.keys()[j-1][1:]+"')")
                        # Perform partial derivation to certian parameter
                        self.Sensitivity_list[-1] = sympy.diff(self.Sensitivity_list[-1],eval(Parameters.keys()[k]))
                       
                    # Multiply sensitivity with the value of the parameter
                    self.Sensitivity_list[-1] = self.Sensitivity_list[-1]*eval(self.Parameters.keys()[k])#/eval(symbol_list[i]+'+1e-6')
        print 'Sensitivity Symbols: '
        print self.Sensitivity_symbols
        print '\n'
        print 'Sensitivity list: '
        print self.Sensitivity_list

    def _check_for_meas(self, Measurable_States):
        '''
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
        '''
        '''
        #CHECK FOR THE INITIAL CONDITIONS
        if Initial_Conditions == False:
            try:
                print 'Used initial conditions: ', self.Initial_Conditions
            except:
                raise Exception('No initial conditions are provided for the current model')            
        else:
            self.set_initial_conditions(Initial_Conditions)               
            print 'Updated initial conditions are used'

    def _check_for_time(self, Timesteps):
        '''
        '''
        #CHECK FOR THE INITIAL CONDITIONS
        if Timesteps == False:
            try:
                print 'Used timesteps: ', self._TimeDict
            except:
                raise Exception('No time step information is provided for the current model')            
        else:
            self.set_time(Timesteps)               
            print 'Updated initial conditions are used'

    def taylor_series_approach(self, iterations, Measurable_States = False,
                             Initial_Conditions = False):
        '''
        Identifiability: TaylorSeriesApproach
        
        Parameters
        -----------
        iterations : int
            Number of derivatives the algorithm has to calculate (TaylorSeries!)
        Measurable_States : list or False
            if False, the previously set variables are used; otherwise this
            contains all the measured states in a list
        inic : Dict of False
            if False, the previously set conditions are used; otherwise this 
            dict contains initial conditions for all states

        Returns
        --------
        Identifiability_Pairwise : array
            Contains for every Measurable state and every iteration, an array
            (number parameters x number parameters), values of 1 show that this
            parameter pair is not interchangeable. Values of 0 shows that this pair is 
            interchangeable. Values of 10 can be ignored.
        
        Identifiability_Ghostparameter : array
            Contains for every Measurable state and every iteration, an array
            (1 x number parameters), a value of 1 show that this parameter is unique. 
            A value of 0 states that this parameter is not uniquely identifiable.
            
        References
        ----------
        .. [1] E. Walter and L. Pronzato, "Identification of parametric models from experimental data.", 1997.
         
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
                    # Take derivative of previous element fo list
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
                for j in range(len(Parameters)):
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
        
        return self.Identifiability_Pairwise, self.Identifiability_Ghostparameter

    def _make_canonical(self):
        '''
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


    def identifiability_check_laplace_transform(self, Measurable_States = False, 
                              Initial_Conditions = False):
        '''
        TODO
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

    def write_model_to_file(self, with_sens = False):
        """
        MakeFile(file_name): makes a file.
        
        if with_sens is True, the sensitivity definitions are also written to the file
        """
        temp_path = os.path.join(os.getcwd(),self.modelname+'.py')
        print 'File is printed to: ', temp_path
        print 'Filename used is: ', self.modelname
        file = open(temp_path, 'w+')
        file.seek(0,0)
        
        file.write('#'+self.modelname+'\n')
        
    #    file.write('\n#Parameters\n\n')
    
        #for i in range(len(Parameters)):
            #file.write(str(Parameters.keys()[i]) + ' = ' + str(Parameters.values()[i])+'\n')
            
        #file.write('\nParameters = '+str(Parameters.keys()).replace("'","")+'\n')
            
        file.write('\n#System definition\n\n')
        
    #    file.write('States = '+str(System.keys()).replace("'d","").replace("'","")+'\n\n')
        
        file.write('def system(States,t,Parameters):\n')
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            file.write('    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n")
        file.write('\n')
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i]).replace("d","") + ' = States['+str(i)+']\n')
        file.write('\n')    
        for i in range(len(self.System)):
            file.write('    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n')
        file.write('    return '+str(self.System.keys()).replace("'","")+'\n')
        
        if with_sens == True:
            try:
                self.Sensitivity_list
            except:
                raise Exception('Run analytic_local_sensitivity first!')
                
            print 'Sensitivities are printed to the file....'
            file.write('\n#Sensitivities\n\n')
        
            file.write('def Sensitivity_direct(States,Parameters):\n')
            temp = []
            for i in range(len(self.System)):
                file.write('    '+self.System.keys()[i][1:]+" = States['"+self.System.keys()[i][1:]+"']\n")
            file.write('\n')
            for i in range(len(self.Parameters)):
                file.write('    '+self.Parameters.keys()[i]+" = Parameters['"+self.Parameters.keys()[i]+"']\n")
            file.write('\n')
            for i in range(len(self.System)*len(self.Parameters)):
                file.write('    '+self.Sensitivity_symbols[i]+' = '+str(self.Sensitivity_list[i])+'\n')
                exec(self.Sensitivity_symbols[i]+" = sympy.symbols('"+self.Sensitivity_symbols[i]+"')")
                temp.append(eval(self.Sensitivity_symbols[i]))
            file.write('    Output = {}\n')
            for i in range(self.System.__len__()):
                for j in range(len(self.Parameters)):
                    file.write("    Output['"+'d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+"'] = "+'d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+'\n')
           
            file.write('    return Output\n')
        #    pprint.pprint(temp,file)
            file.write('\n')
            temp = []
            test = []
            file.write('def Sensitivity_indirect(States,Parameters):\n')
            for i in range(len(self.System)):
                file.write('    '+self.System.keys()[i][1:]+" = States['"+self.System.keys()[i][1:]+"']\n")
            file.write('\n')
            for i in range(len(Parameters)):
                file.write('    '+self.Parameters.keys()[i]+" = Parameters['"+self.Parameters.keys()[i]+"']\n")
            file.write('\n')
            for i in range(len(self.System)*len(self.Parameters),len(self.Sensitivity_symbols)):
                file.write('    '+self.Sensitivity_symbols[i]+' = '+str(self.Sensitivity_list[i])+'\n')
                temp.append(self.Sensitivity_symbols[i])
            file.write('    Output = {}\n')
            for i in range(self.System.__len__()):
                for j in range(len(self.Parameters)):
                    file.write('    d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+' = ')
                    for k in range(System.__len__()):
                        file.write('d'+self.System.keys()[i][1:]+'d'+self.System.keys()[k][1:]+'Xd'+self.System.keys()[k][1:]+'d'+self.Parameters.keys()[j]+' + ') 
                    file.seek(-3,2)
                    file.write('\n')
        #            for k in range(System.__len__()):
        #                file.write("    Output['"+'d'+state_list[i]+'d'+state_list[k]+'Xd'+state_list[k]+'d'+parameter_list[j]+"'] = " + 'd'+state_list[i]+'d'+state_list[k]+'Xd'+state_list[k]+'d'+parameter_list[j]+'\n')

                    exec('d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+" = sympy.symbols('"+'d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+"')")
                    test.append(eval('d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]))
            
            for i in range(self.System.__len__()):
                for j in range(len(self.Parameters)):
                    file.write("    Output['"+'d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+"'] = "+'d'+self.System.keys()[i][1:]+'d'+self.Parameters.keys()[j]+'\n')
                    
            
            file.write('    return Output\n')
            print '...done!'
        file.close()

    def solve_ode(self, TimeStepsDict = False, Initial_Conditions = False, plotit = True):
        '''
        Solve the differential equation
        '''

        self._check_for_time(TimeStepsDict)
        self._check_for_init(Initial_Conditions)        
        
        import MODEL_Halfreaction
        res = spin.odeint(MODEL_Halfreaction.system,self.Initial_Conditions.values(), self._Time,args=(self.Parameters,))
        
        #put output in pandas dataframe
        df = pd.DataFrame(res, columns = self._Variables)
        print df
        
        #plotfunction
        if plotit == True:
            df.plot(subplots = True)
               
        return df



#PREPARE MODEL
Parameters = {'k1':1/10,'k1m':1/20,
              'k2':1/20,'k2m':1/20,
              'k3':1/200,'k3m':1/175,
              'k4':1/200,'k4m':1/165}
              
System =    {'dEn':'k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ',
             'dEs':'- k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ',
             'dSA':'- k1*En*SA + k1m*Es*PP',
             'dSB':'- k2*Es*SB + k2m*En*PQ',
             'dPP':'k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP',
             'dPQ':'k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ',
             'dEsQ':'k3*Es*PQ - k3m*EsQ',
             'dEP':'k4*En*PP - k4m*EP'}
                        
Modelname = 'MODEL_Halfreaction'

#INITIATE MODEL
M1 = odegenerator(System, Parameters, Modelname = Modelname)

#M1.analytic_local_sensitivity()
M1.set_measured_states(['SA', 'SB', 'PP', 'PQ'])
M1.set_initial_conditions({'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.})
#M1.taylor_series_approach(2)
#H1,H2 = M1.identifiability_check_laplace_transform()
M1.set_time({'start':1,'end':20,'nsteps':10000})
modeloutput = M1.solve_ode(plotit=True)
