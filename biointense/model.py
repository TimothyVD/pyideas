# -*- coding: utf-8 -*-
## License: LELIJKE DASHONDEN
## All rights to us and none to the others!

from __future__ import division

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy

from modelbase import BaseModel
from modeldefinition import (generate_ode_derivative_definition,
                             generate_non_derivative_part_definition)
from solver import (OdeSolver, OdeintSolver, OdespySolver,
                    HybridOdeintSolver, HybridOdeSolver,
                    HybridOdespySolver, AlgebraicSolver)


class Model(BaseModel):

    def __init__(self, name, system, parameters, comment=None):
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
        self.initial_conditions = {}

        # detect system equations
        self._system = system
        self._parse_system_string(self._system, self.parameters)
        self.variables = list(itertools.chain(*self._ordered_var.values()))
        self.variables_of_interest = deepcopy(self.variables)

        self.fun_ode = None
        self.fun_alg = None

    def __str__(self):
        """
        string representation
        """
        return "Model name: " + str(self.name) + \
            "\n Variables of interest: \n" + str(self.variables_of_interest) +\
            "\n Parameters: \n" + str(self.parameters) + \
            "\n Independent: \n" + str(self.independent) + \
            "\n Model initialised: " + str(self._initial_up_to_date)

    def __repr__(self):
        """
        """
        print("Model name: " + str(self.name) +
              "\n Variables: \n" + str(self.variables) +
              "\n Variables of interest: \n" + str(self.variables_of_interest) +
              "\n Functions: \n" + str(self.systemfunctions) +
              "\n Parameters: \n" + str(self.parameters) +
              "\n Independent: \n" + str(self.independent) +
              "\n Initial conditions: \n" + str(self.initial_conditions) +
              "\n Model initialised: " + str(self._initial_up_to_date))

    def _parse_system_string(self, system, parameters):
        """
        split the system in ODE & algebraic
        extract variable names
        first letter == d ==> to ODE
        else ==> to algebraic
        extract info from system and parameters and store them into the attributes
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
        if self.systemfunctions['algebraic']:
            self.fun_alg_str = generate_non_derivative_part_definition(self)
            exec(self.fun_alg_str)
            self.fun_alg = fun_alg
        if self.systemfunctions['ode']:
            self.fun_ode_str = generate_ode_derivative_definition(self)
            exec(self.fun_ode_str)
            self.fun_ode = fun_ode

        self._initial_up_to_date = True

    def run(self):
        """
        Run the model for the given set of parameters, indepentent variable
        values and output a datagrame with the variables of interest.

        """
        if not self._initial_up_to_date:
            self.initialize_model()

        if self.fun_ode and self.fun_alg:
            solver = HybridOdeintSolver(self)
        elif self.fun_ode:
            solver = OdeintSolver(self)
        elif self.fun_alg:
            solver = AlgebraicSolver(self)
        else:
            raise Exception("In an initialized Model, there should always "
                            "be at least a fun_ode or fun_alg.")

        result = solver.solve()

        return result

    @classmethod
    def from_external(cls, ext_sys):
        """
        initialise system from external function
        integratei met andere paketten om het in een
        """
        return cls(None)

        # Can also be deleted

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
            if ((key in self._ordered_var['algebraic'])
                    or (key in self._ordered_var['event'])
                    or (key in self._ordered_var['ode'])):
                self.initial_conditions[key] = value
            else:
                raise NameError('Variable ' + key + " does not exist within "
                                "the system")

    def _check_for_init(self):
        """
        """
        return NotImplementedError

    def add_event(self, variable, ext_fun, tijdsbehandeling, idname):
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

        return NotImplementedError

    def list_current_events(self):
        """
        """
        return NotImplementedError

    def exclude_event(self, idname):
        """
        """
        return NotImplementedError

    def _collect_time_steps(self):
        """
        """
        return NotImplementedError


class AlgebraicModel(Model):

    def __init__(self, name, system, parameters, comment=None):
        """
        uses the "biointense"-style model definition
        >>> system = {'v' : 'Vf * SA * SB/(Kp * SA + Km * SB + SA * SB)'}
        >>> param = {'Vf': 1., 'Km': 1., 'Kp': 1.}
        >>> name = 'pingpongbibi'
        >>> AlgebraicModel(name, system, param)
        """
        super(AlgebraicModel, self).__init__(name, system, parameters,
                                             comment=comment)

        self._independent_len = {}


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

    def _parse_system_string(self, system, parameters):
        """
        split the system in ODE & algebraic
        extract variable names
        first letter == d ==> to ODE
        else ==> to algebraic
        extract info from system and parameters and store them into the attributes
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
                raise Exception('Algebraic class cannot work with ODEs')
            else:
                self.systemfunctions['algebraic'][key] = value
                self._ordered_var['algebraic'].append(key)

    def set_initial(self, *args):
        """
        """
        raise Exception('No initial conditions can be set for algebraics')

    def set_independent(self, independent_dict, method='cartesian'):
        """
        """
        # check the data type of the input
        if not isinstance(independent_dict, dict):
            raise TypeError("independent_dict should be dict!")

        if method == "cartesian":
            independent = list(itertools.product(*independent_dict.values()))
        else:
            raise Exception('Method is not available!')

        independent = pd.DataFrame(independent,
                                   columns=independent_dict.keys())

        for key in independent_dict.keys():
            self._independent_len[key] = len(independent_dict[key])
            self._independent_values[key] = independent[key].values
        self.independent = self._independent_values.keys()

    def plot_contourf(self, independent_x, independent_y, output, ax=None,
                      **kwargs):
        """
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
