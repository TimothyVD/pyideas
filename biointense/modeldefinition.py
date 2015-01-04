# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 12:02:32 2015

@author: stvhoey
"""

def write_whiteline(defstr):
    """
    defstr : str
        str containing the definition to solve in model       
    """
    defstr += '\n'
    return defstr
    
def write_parameters(defstr, parameters):
    """
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model    
    parameters : dict
        key gives parameter names and values the corresponding value
    """
    for parname, parvalue in parameters.iteritems():
        defstr += "    {0} = parameters['{0}']\n".format(parname)
    return defstr        

def write_ode_indices(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the ode sequence is printed
    
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model    
    ode_variables : list
        variable names (!sequence is important)
    """
    for i, varname in enumerate(ode_variables):
        defstr += '    {0} = odes[{1}]\n'.format(varname, str(i))
    return defstr

def write_array_extraction(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the ode sequence is printed
    
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model    
    ode_variables : list
        variable names (!sequence is important)
    """
    for i, varname in enumerate(ode_variables):
        defstr += '    {0} = solved_variables[:, {1}]\n'.format(varname, 
                                                                str(i))
    return defstr    

def write_external_call(defstr, varname, fname, argnames):
    """
    
    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    varname : str
        variable name to calculate
    fname : str
        function name to call
    argnames : list
        other arguments that the function request (e.g. t or other varnames)
    """    
    #defstr += '    ' + varname + ' = ' + str(fname) + '('
    defstr += '    {0} = {1}('.format(varname, fname)
    for argument in argnames:
        defstr += str(argument) + ','
    defstr += ')\n'
    return defstr

def write_algebraic_lines(defstr, algebraic_right_side):
    """
    Based on the model equations of the algebraic-part model, the equations are 
    printed in the function
    
    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of 
        the equation
    """    
    for varname, expression in algebraic_right_side.iteritems():     
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'
        defstr += '    {0} = {1}\n'.format(varname, str(expression))
    return defstr

def write_algebraic_solve(defstr, algebraic_right_side):
    """
    Based on the model equations of the algebraic-part model, the equations are 
    printed in the function
    
    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of 
        the equation
    """    
    for varname, expression in algebraic_right_side.iteritems():     
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'
        defstr += '    {0} = {1} + np.zeros(len(x))\n'.format(varname, 
                                                              str(expression))
    return defstr

def write_ode_lines(defstr, ode_right_side):
    """
    Based on the model equations of the ode-part model, the equations are 
    given in the function
    
    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of 
        the equation
    """    
    for varname, expression in ode_right_side.iteritems():     
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'    
        defstr += '    d{0} = {1}\n'.format(varname, str(expression))
    return defstr

def write_derivative_return(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the return sequence is returned
    
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model    
    ode_variables : list
        variable names (!sequence is important)
    """
    ode_variables = ["d" + variable for variable in ode_variables]
    defstr += '    return [' + ', '.join(ode_variables) + ']\n\n'
    return defstr

def write_non_derivative_return(defstr, algebraic_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the return sequence is returned
    
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model    
    ode_variables : list
        variable names (!sequence is important)
    """
    #collect and return the resulting algebraic parts
    defstr += '    nonder = np.array([' + ', '.join(algebraic_variables) + ']).T\n'
    defstr += '    return nonder'
    return defstr

def generate_ode_derivative_definition(model):
    '''Write derivative of model as definition in file
    
    Writes a file with a derivative definition to run the model and
    use it for other applications
    
    Parameters
    -----------
    model : biointense.model
    
    '''
    modelstr = 'def _fun_ode(odes, t, parameters, *args, **kwargs):\n'
    # Get the parameter values 
    modelstr = write_parameters(modelstr, model.parameters)
    modelstr = write_whiteline(modelstr)
    # Get the current variable values from the solver
    modelstr = write_ode_indices(modelstr, model.variables['ode'])
    modelstr = write_whiteline(modelstr)
    # Write down necessary algebraic equations (if none, nothing written)
    modelstr = write_algebraic_lines(modelstr, model.systemfunctions['algebraic'])
    modelstr = write_whiteline(modelstr)

    # Write down external called functions - not yet provided!
    #write_external_call(defstr, varname, fname, argnames)
    #write_whiteline(modelstr)
    
    # Write down the current derivative values
    modelstr = write_ode_lines(modelstr, model.systemfunctions['ode'])
    modelstr = write_derivative_return(modelstr, model.variables['ode'])
    return modelstr
 
def generate_non_derivative_part_definition(model):  
    '''Write derivative of model as definition in file
    
    Writes a file with a derivative definition to run the model and
    use it for other applications
    
    Parameters
    -----------
    model : biointense.model
    
    '''    
    modelstr = 'def _fun_alg(x, parameters, *args, **kwargs):\n'
    # Get the parameter values 
    modelstr = write_parameters(modelstr, model.parameters)
    modelstr = write_whiteline(modelstr)
    
    # Put the variables in a separate array 
    
    modelstr = write_array_extraction(modelstr, model.variables['ode'])
    modelstr = write_whiteline(modelstr)

    # Write down external called functions - not yet provided!
    #write_external_call(defstr, varname, fname, argnames)
    #write_whiteline(modelstr)
    
    # Write down the equation of algebraic
    modelstr = write_algebraic_solve(modelstr, model.systemfunctions['algebraic'])
    modelstr = write_whiteline(modelstr)

    modelstr = write_non_derivative_return(modelstr, model.variables['algebraic'])
    return modelstr     
