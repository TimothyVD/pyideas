# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:26:56 2014

@author: timothy
"""

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