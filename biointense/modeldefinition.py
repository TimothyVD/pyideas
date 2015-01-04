# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 12:02:32 2015

@author: stvhoey
"""

def write_defheader(defstr):
    """
    """
    defstr += "def _fun_ODE(t, parameters, odes, *args, **kwargs):\n"

def write_whiteline(defstr):
    """
    """
    defstr += '\n'
    
def write_parameters(defstr, parameters):
    """
    
    Parameters
    ----------
    parameters : dict
        key gives parameter names and values the corresponding value
    """
    for parname, parvalue in parameters.iteritems():
        defstr += '    '+ parname + " = parameters['" + parname + "']\n"

def write_ode_indices(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the ode sequence is printed
    
    Parameters
    ----------
    system : list
    """
    for i, varname in enumerate(ode_variables):
        defstr += '    ' + varname + ' = ODES['+str(i)+']\n'    

def write_external_call(defstr, varname, fname, args):
    """
    
    Parameters
    -----------
    
    """    
    system += '    ' + varname + ' = fname(' + self._x_var + ')'+'\n'

def write_algebraic_lines():
    """
    """    
    



    
if self._has_algebraic:
    for i in range(len(self.Algebraic)):
        #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
        system +='    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n'
    system +='\n'

for i in range(len(self.System)):
    system +='    '+str(self.System.keys()[i]) + ' = ' + str(self.System.values()[i])+'\n'

system +='    return '+str(self.System.keys()).replace("'","")+'\n\n\n'   


def generate_model_definition(self):
    '''Write derivative of model as definition in file
    
    Writes a file with a derivative definition to run the model and
    use it for other applications
    
    Parameters
    -----------
    
    '''

    # Write function for solving ODEs only
    system = ""
    LSA_analytical = ""
    algebraic = ""
    algebraic_sens = ""
    if self._has_ODE and not self._has_def_ODE:
        if self._has_externalfunction:
            if self.ode_procedure == "ode":
                system +='def _fun_ODE(t,ODES,Parameters,input):\n'
            else:
                system +='def _fun_ODE(ODES,t,Parameters,input):\n'
        else:
            if self.ode_procedure == "ode":
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
            for i, ext_var in enumerate(self._externalvariables):
                system +='    '+ ext_var + ' = input['+str(i)+']('+self._x_var+')'+'\n'
                #system +='    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
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
            if self.ode_procedure == "ode":
                LSA_analytical += 'def _fun_ODE_LSA('+self._x_var+',ODES,Parameters,input):\n'
            else:
                LSA_analytical += 'def _fun_ODE_LSA(ODES,'+self._x_var+',Parameters,input):\n'
        else:
            if self.ode_procedure == "ode":
                LSA_analytical += 'def _fun_ODE_LSA('+self._x_var+',ODES,Parameters):\n'
            else:
                LSA_analytical += 'def _fun_ODE_LSA(ODES,'+self._x_var+',Parameters):\n'
        for i in range(len(self.Parameters)):
            #file.write('    '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
            LSA_analytical += '    '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
        LSA_analytical += '\n'
        for i in range(len(self.System)):
            LSA_analytical += '    '+str(self.System.keys()[i])[1:] + ' = ODES['+str(i)+']\n'
        LSA_analytical += '\n'
        if self._has_externalfunction:
            #for i, step in enumerate(self.externalfunction):
            for i, ext_var in enumerate(self._externalvariables):
                LSA_analytical +='    '+ ext_var + ' = input['+str(i)+']('+self._x_var+')'+'\n'
                #LSA_analytical += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
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
                algebraic += '\ndef _fun_alg(ODES,'+self._x_var+',Parameters, input):\n'
            else:
                algebraic += '\ndef _fun_alg(ODES,'+self._x_var+',Parameters):\n'
        elif self._has_externalfunction:
            algebraic += '\ndef _fun_alg('+self._x_var+',Parameters, input):\n'
        else:
            algebraic += '\ndef _fun_alg('+self._x_var+',Parameters):\n'
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
            for i, ext_var in enumerate(self._externalvariables):
                algebraic +='    '+ ext_var + ' = input['+str(i)+']('+self._x_var+')'+'\n'
            #for i, step in enumerate(self.externalfunction):
                #algebraic += '    input'+str(i) + ' = input['+str(i)+']('+self._x_var+')'+'\n'
            algebraic += '\n'
        for i in range(len(self.Algebraic)):
            #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+' + zeros(len(t))\n')
            algebraic += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])
            if self.solve_fast_way:
                algebraic += ' + np.zeros(len('+self._x_var+'))\n'
            else:
                algebraic += '\n'
        algebraic += '\n'
        algebraic += '    algebraic = array('+str(self.Algebraic.keys()).replace("'","")+').T\n\n'
        algebraic += '    return algebraic\n\n\n'
        
        #

        if self._has_ODE and not self._has_def_ODE:
            #Algebraic sens
            if self._has_externalfunction:
                algebraic_sens += '\ndef _fun_alg_LSA(ODES,'+self._x_var+',Parameters, input, dxdtheta):\n'
            else:
                algebraic_sens += '\ndef _fun_alg_LSA(ODES,'+self._x_var+',Parameters, dxdtheta):\n'
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
                for i, ext_var in enumerate(self._externalvariables):
                    algebraic_sens +='    '+ ext_var + ' = input['+str(i)+']('+self._x_var+')'+'\n'
                #for i, step in enumerate(self.externalfunction):
                #    algebraic_sens += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                algebraic_sens += '\n'
            for i in range(len(self.Algebraic)):
                algebraic_sens += '    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i]) +'\n'
                #file.write('    '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
            if self._print_on:
                print('Algebraic sensitivities are printed to string....')
            algebraic_sens += '\n    #Sensitivities\n\n'
                       
            # Write dgdtheta as symbolic array
            algebraic_sens += '    dgdtheta = np.zeros([len('+self._x_var+'), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.Parameters.keys())) + '])\n'
            for i, alg in enumerate(self.Algebraic.keys()):
                for j, par in enumerate(self.Parameters.keys()):
                    algebraic_sens += '    dgdtheta[:,' + str(i) + ',' + str(j) +'] = ' + str(self.dgdtheta[i,j])+'\n'

            # Write dgdx as symbolic array
            algebraic_sens += '    dgdx = np.zeros([len('+self._x_var+'), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.System.keys())) + '])\n'
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
                algebraic_sens += '\ndef _fun_alg_LSA('+self._x_var+',Parameters, input):\n\n'
            else:
                algebraic_sens += '\ndef _fun_alg_LSA('+self._x_var+',Parameters):\n\n'
            algebraic_sens += '    _temp_fix = np.zeros([len('+self._x_var+')])\n\n'
            for i in range(len(self.Parameters)):
                #file.write(' '+str(Parameters.keys()[i]) + ' = Parameters['+str(i)+']\n')
                algebraic_sens += ' '+str(self.Parameters.keys()[i]) + " = Parameters['"+self.Parameters.keys()[i]+"']\n"
            algebraic_sens += '\n'
            if self._has_externalfunction:
                #for i, step in enumerate(self.externalfunction):
                for i, ext_var in enumerate(self._externalvariables):
                    algebraic_sens +='    '+ ext_var + ' = input['+str(i)+']('+self._x_var+')'+'\n'
                 #   algebraic_sens += '    input'+str(i) + ' = input['+str(i)+'](t)'+'\n'
                algebraic_sens += '\n'
            for i in range(len(self.Algebraic)):
                #file.write(' '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic.values()[i])+'\n')
                algebraic_sens += ' '+str(self.Algebraic.keys()[i]) + ' = ' + str(self.Algebraic_swapped[i])+'\n'
            algebraic_sens += '\n'
            if self._print_on:
                print('Algebraic sensitivities are printed to the file....')
            algebraic_sens += '\n #Sensitivities\n\n'
                       
            # Write dgdtheta as symbolic array
            algebraic_sens += '    dgdtheta = '   
            algebraic_sens += pprint.pformat(self.dgdtheta + sympy.sympify('_temp_fix'))
            
            algebraic_sens += '\n\n    dydtheta = np.rollaxis(dgdtheta,2,0)'

        algebraic_sens += '\n\n    return dydtheta'+'\n\n\n'
    
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
        print('... All functions are generated!')