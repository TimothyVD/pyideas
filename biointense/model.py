# -*- coding: utf-8 -*-
## License: LELIJKE DASHONDEN
## All rights to us and none to the others!

from __future__ import division

import numpy as np

from modelbase import BaseModel
from modeldefinition import (generate_ode_derivative_definition, 
                             generate_non_derivative_part_definition)


class Model(BaseModel):

    def __init__(self, name, system, parameters, comment=None):
        """
        uses the "biointense"-style model definition
        """
        self.name = name
        self._check_name()

        self.variables = {'algebraic': [],
                          'ode': [],
                          'event': [],
                          'independent': []
                          }

        self.comment = comment

        # solver communication
        self.independent_values = None
        self.parameters = parameters
        self.systemfunctions = {'algebraic': {}, 'ode': {}}
        self.initial_conditions = {}

        # detect system equations
        self._system = system
        self._parse_system_string(self._system, self.parameters)

        self.variables_of_interest = []
        self._initial_up_to_date = False

        self.fun_ode = None
        self.fun_alg = None    

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
                self.variables['ode'].append(key[1:])
            else:
                self.systemfunctions['algebraic'][key] = value
                self.variables['algebraic'].append(key)

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


class AlgebraicModel(BaseModel):

    def __init__():
        """
        with multidimensional independant variables
        """

    def set_independent(self):
        """
        set independent variable, mostly time
        """


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
