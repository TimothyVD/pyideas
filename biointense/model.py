# -*- coding: utf-8 -*-
## License: LELIJKE DASHONDEN
## All rights to us and none to the others!

from __future__ import division

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

from modelbase import BaseModel
from modeldefinition import (generate_ode_derivative_definition,
                             generate_non_derivative_part_definition)
from solver import OdeSolver, AlgebraicSolver, HybridSolver


class _BiointenseModel(BaseModel):

    def __str__(self):
        """
        string representation
        """
        return ("Model name: " + str(self.name) + "\n"
                "Variables of interest:" + str(self.variables_of_interest) + "\n"
                "Parameters: \n" + str(self.parameters) + "\n"
                "Independent: \n" + str(self.independent) + "\n"
                "Model initialised: " + str(self._initial_up_to_date) + "\n")

    def _parse_system_string(self, system, parameters):
        """
        split the system in ODE & algebraic
        extract variable names
        first letter == d ==> to ODE
        else ==> to algebraic
        extract info from system and parameters and store them into the
        attributes
        """
        # assert that 'parameters' and 'system' are a dict
        if not isinstance(parameters, dict):
            raise TypeError("parameters is not a dict")
        if not isinstance(system, dict):
            raise TypeError("system is not a dict")
        # store the parameter
        self.parameters = parameters
        # extract system information
        # loop over the dictionairy: system
        for key, value in system.iteritems():
            # if first letter == d, equation is ODE
            if key[0] == "d":
                #get rid of the first letter, d
                self.systemfunctions['ode'][key[1:]] = value
                self._ordered_var['ode'].append(key[1:])
            else:
                self.systemfunctions['algebraic'][key] = value
                self._ordered_var['algebraic'].append(key)

    def initialize_model(self):
        """
        Parse system string equation to functions.
        """
        self._check_for_independent()

        #from modeldefinition import
        if self.systemfunctions.get('algebraic', None):
            self.fun_alg_str = generate_non_derivative_part_definition(self)
            exec(self.fun_alg_str)
            self.fun_alg = fun_alg
        if self.systemfunctions.get('ode', None):
            self.fun_ode_str = generate_ode_derivative_definition(self)
            exec(self.fun_ode_str)
            self.fun_ode = fun_ode

        self._initial_up_to_date = True

    def _args_ode_function(self, fun, **kwargs):
        """
        """
        externalfunctions = kwargs.get('externalfunctions')
        initial_conditions = [self.initial_conditions[var]
                              for var in self._ordered_var['ode']]
        args = (fun, initial_conditions,
                self._independent_values)
        if externalfunctions:
            args += tuple(((self.parameters, externalfunctions,),))
        else:
            args += tuple(((self.parameters,),))

        return args

    def _args_alg_function(self, fun, **kwargs):
        """
        """
        externalfunctions = kwargs.get('externalfunctions')
        args = (fun, self._independent_values)
        if externalfunctions:
            args += tuple(((self.parameters, externalfunctions,),))
        else:
            args += tuple(((self.parameters,),))

        return args

    def _args_ode_alg_function(self, **kwargs):
        """
        """
        externalfunctions = kwargs.get('externalfunctions')
        initial_conditions = [self.initial_conditions[var]
                              for var in self._ordered_var['ode']]
        args = (self.fun_ode, self.fun_alg, initial_conditions,
                self._independent_values)
        if externalfunctions:
            args += tuple(((self.parameters, externalfunctions,),))
        else:
            args += tuple(((self.parameters,),))

        return args

    def run(self, procedure="odeint"):
        """
        Run the model for the given set of parameters, indepentent variable
        values and output a datagrame with the variables of interest.

        """
        if not self._initial_up_to_date:
            self.initialize_model()

        ode_var = self._ordered_var.get('ode')
        alg_var = self._ordered_var.get('algebraic')

        if ode_var:
            var = [] + ode_var
            if alg_var:
                var += alg_var
                ode_alg_args = self._args_ode_alg_function()
                solver = HybridSolver(*ode_alg_args)
            else:
                ode_args = self._args_ode_function(self.fun_ode)
                solver = OdeSolver(*ode_args)
            result = solver.solve(procedure=procedure)#,
                                  #externalfunctions=self.externalfunctions)
        elif alg_var:
            var = [] + alg_var
            alg_args = self._args_alg_function(self.fun_alg)
            solver = AlgebraicSolver(*alg_args)
            #result = solver.solve(externalfunctions=self.externalfunctions)
            result = solver.solve()
        else:
            raise Exception("In an initialized Model, there should always "
                            "be at least a fun_ode or fun_alg.")

        index = pd.MultiIndex.from_arrays(self._independent_values.values(),
                                          names=self.independent)
        result = pd.DataFrame(result, index=index, columns=var)

        return result

    @classmethod
    def from_external(cls, ext_sys):
        """
        initialise system from external function
        integratei met andere paketten om het in een
        """
        return cls(None)

        # Can also be deleted

    def add_event(self, idname, variable, ext_fun, arguments):
        """
        Variable is defined by external influence. This can be either a
        measured value of input (e.g. rainfall) or a function that defines
        a variable in function of time

        See also:
        ---------
        functionMaker

        plug to different files: step input ...
        + add control to check whether external function addition is possible

        + check if var exists in ODE/algebraic => make aggregation function to
        contacate them.
        """
        self._initial_up_to_date = False
        self._has_external = True

        self.externalfunctions[idname] = {'variable': variable,
                                          'fun': ext_fun,
                                          'arguments': arguments}

    def list_current_events(self):
        """
        """
        return self.externalfunctions

    def exclude_event(self, idname):
        """
        """
        del self.externalfunctions[idname]

        if not bool(self.externalfunctions):
            self._has_external = False

    def _collect_time_steps(self):
        """
        """
        return NotImplementedError


class Model(_BiointenseModel):
    """
    The Model instance provides the ability to construct a model consisting of
    algebraic equations, ordinary differential equations or a combination of
    both.

    Parameters
    -----------
    modelname : string
        String name to define the model
    system : dict
        dict containing all equations (algebraic equations and ODEs). The keys
        of the algebraic variables are just the algebriac variables itself. The
        keys of the ODEs are 'd' plus the name of the state.
    parameters : dict
        dict with parameter names as keys, parameter values are the
        values of the dictionary


    Examples
    ---------
    >>> import numpy as np
    >>> from biointense import Model
    >>> parameters = {'Km': 150.,     # mM
                      'Vmax': 0.768,  # mumol/(min*U)
                      'E': 0.68}      # U/mL
    >>> system = {'v': 'Vmax*S/(Km + S)',
                  'dS': '-v*E',
                  'dP': 'v*E'}

    >>> M1 = Model('Michaelis-Menten', system, parameters)
    >>> M1.set_initial({'S':500.,
                        'P':0.})
    >>> M1.set_independent({'t': np.linspace(0, 2500, 10000)})
    >>> #run the model
    >>> modeloutput = M1.run()
    """

    def __init__(self, name, system, parameters, ode_independent='t',
                 comment=None):
        """
        uses the "biointense"-style model definition
        >>> sir = {'dS' : '-k*I*B/N',
                   'dI' : 'k*I*B/N - gam*I*t',
                   'dR' : 'gam*I',
                   'N' : 'S + I + R + NA'}
        >>> param = {'k': 2., 'gam' : 0.3}
        >>> name = 'SIR1'
        >>> Model(name, system, param)
        """
        super(Model, self).__init__(name, parameters, comment=comment)

        self._ordered_var = {'algebraic': [],
                             'ode': [],
                             'event': []}

        # solver communication
        self.systemfunctions = {'algebraic': {}, 'ode': {}}
        self.externalfunctions = {}
        self.initial_conditions = {}

        self._ode_independent = ode_independent

        # detect system equations
        self._system = system
        self._parse_system_string(self._system, self.parameters)
        self.variables = list(itertools.chain(*self._ordered_var.values()))
        self.variables_of_interest = deepcopy(self.variables)

        self.fun_alg = None
        self.fun_ode = None

        self._has_external = False

    def __repr__(self):
        """
        """
        return("Model name: " + str(self.name) +
               "\n Variables: \n" + str(self.variables) +
               "\n Variables of interest: \n" + str(self.variables_of_interest) +
               "\n Functions: \n" + str(self.systemfunctions) +
               "\n Parameters: \n" + str(self.parameters) +
               "\n Independent: \n" + str(self.independent) +
               "\n Initial conditions: \n" + str(self.initial_conditions) +
               "\n Model initialised: " + str(self._initial_up_to_date))

    def set_initial(self, initialValues):
        """
        set initial conditions
        check for type
        check for existance of the variable
        """
        if self.initial_conditions:
            warnings.warn("Warning: initial conditions are already given. "
                          "Overwriting original variables.")
        if not isinstance(initialValues, dict):
            raise TypeError("Initial values are not given as a dict")
        for key, value in initialValues.iteritems():
            if ((key in self._ordered_var['algebraic']) or
                (key in self._ordered_var['event']) or
                (key in self._ordered_var['ode'])):
                    self.initial_conditions[key] = value
            else:
                raise NameError('Variable ' + key + " does not exist within "
                                "the system")

    def _check_for_init(self):
        """
        """
        return NotImplementedError


class AlgebraicModel(_BiointenseModel):
    """
    The AlgebraicModel instance provides the ability to construct a model
    consisting of only algebraic equations, but with the ability to define
    multiple independent variables at once.

    Parameters
    -----------
    modelname : string
        String name to define the model
    system : dict
        dict containing all algebraic equations. The keys
        of the algebraic variables are just the algebriac variables itself
    parameters : dict
        dict with parameter names as keys, parameter values are the
        values of the dictionary


    Examples
    ---------
    >>> import numpy as np
    >>> from biointense import AlgebraicModel
    >>> parameters = {'Km': 150.,     # mM
                      'Kp': 200.,
                      'Vmax': 0.768,  # mumol/(min*U)
                      'E': 0.68}      # U/mL
    >>> system = {'v': 'Vmax*A*B/(Km*B + Kp*A + A*B)'}

    >>> M1 = AlgebraicModel('Double-Michaelis-Menten', system, parameters)
    >>> M1.set_independent({'A': np.linspace(0, 400, 25),
                            'B': np.linspace(0, 400, 25)},
                            method='cartesian')
    >>> #run the model
    >>> modeloutput = M1.run()
    >>> M1.plot_contourf('A', 'B', modeloutput)
    """

    def __init__(self, name, system, parameters, comment=None):
        """
        """
        self._check_if_odes(system.keys())

        super(AlgebraicModel, self).__init__(name, parameters, comment=comment)

        self._ordered_var = {'algebraic': [],
                             'event': []}

        # solver communication
        self.systemfunctions = {'algebraic': {}}
        self.externalfunctions = {}

        # Keep track of length of individiual independent
        self._independent_len = {}

        # detect system equations
        self._system = system
        self._parse_system_string(self._system, self.parameters)
        self.variables = list(itertools.chain(*self._ordered_var.values()))
        self.variables_of_interest = deepcopy(self.variables)

        self.fun_alg = None

    @staticmethod
    def _check_if_odes(system_keys):
        for eq in system_keys:
            if eq.startswith('d'):
                raise Exception('Algebraic class cannot work with ODEs')

    def __repr__(self):
        """
        """
        print("Model name: " + str(self.name) +
              "\n Variables: \n" + str(self.variables) +
              "\n Variables of interest: \n" + str(self.variables_of_interest) +
              "\n Functions: \n" + str(self.systemfunctions) +
              "\n Parameters: \n" + str(self.parameters) +
              "\n Independent: \n" + str(self.independent) +
              "\n Model initialised: " + str(self._initial_up_to_date))

    def set_independent(self, independent_dict, method='direct'):
        """
        Method to set the independent values of the AlgebraicModel instance.
        For the algebraic model it is possible to pass multiple independent
        arrays.

        Parameters
        -----------
        independent_dict: dict
            dict containing one or multiple key(s), each key is related with a
            numpy array.
        method: 'direct'|'cartesian'
            When passing multiple independents, these can be aligned in
            different ways. The first option is 'direct', which means that all
            arrays of the independent have the same length and the number of
            calculations is equal to the length of one array. The second option
            is 'cartesian' in which all possible combinations between the
            different independent values are generated. In case of two
            independents, one will have independent1*independent2 number of
            calculations.

        Example
        --------
        >>> M1.set_independent({'A': np.linspace(0, 400, 25),
                                'B': np.linspace(0, 400, 25)},
                               method='cartesian')
        """
        # check the data type of the input
        if not isinstance(independent_dict, dict) and \
            not isinstance(independent_dict, pd.core.frame.DataFrame):
                raise TypeError("independent_dict should be dict or pd.DF!")

        if method == "cartesian":
            independent = list(itertools.product(*independent_dict.values()))
        elif method == "direct":
            independent = independent_dict
        else:
            raise Exception('Method is not available!')

        independent = pd.DataFrame(independent,
                                   columns=independent_dict.keys())

        self._independent_len = {}
        self._independent_values = {}

        for key in independent_dict.keys():
            self._independent_len[key] = len(independent_dict[key])
            self._independent_values[key] = independent[key].values
        self.independent = self._independent_values.keys()

    def plot_contourf(self, independent_x, independent_y, output, ax=None,
                      **kwargs):
        """
        Parameters
        -----------
        independent_x: string
            Independent of interest to be shown at the x-axis.
        independent_y: string
            Independent of interest to be shown at the y-axis.
        output: pandas.Series
            algebraic equation to be shown as a contourplot (in function of
            independent_x and independent_y)
        ax: matplotlib.ax
            Pass ax to plot on.
        """
        shape = self._independent_len.values()
        x = np.reshape(self._independent_values[independent_x], shape)
        y = np.reshape(self._independent_values[independent_y], shape)
        z = np.reshape(output.values, shape)

        if ax is None:
            ax = plt.gca()

        cs = ax.contourf(x, y, z, **kwargs)
        ax.set_xlabel(independent_x)
        ax.set_ylabel(independent_y)
        plt.colorbar(cs)

        return ax


class ReactionModel(BaseModel):

    def __init__():
        """
        """

    @classmethod
    def from_diagram(cls):
        """
        Creates model based on the
        """


class EnzymaticModel(ReactionModel):

    def __init__():
        """
        """

    def _getCoefficients(self):
        """
        """

    @classmethod
    def make_quasi_steady_state(cls):
        """
        Converts the ODE system to the Quasi Steady State version

        Combines the old versions make_QSSA and QSSAtoModel to create QSSA
        model based on a defined ODE system.
        """
        return True

def check_mass_balance():
    """
    Check the mass balance of the model.

    This method calls the external utility _getCoefficients
    """
    return True
