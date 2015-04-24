
## License: LELIJKE DASHONDEN
## All rights to us and none to the others!
from __future__ import division

import warnings
import pandas as pd

class BaseModel(object):

    def __init__(self, name, parameters, comment=None):
        """

        ODE equations are defined by the pre-defined d, whereas algebraic
        equations are defined by equations without a d in front

        Example
        --------
        >>> param = {'k': 2., 'gam': 0.3}
        >>> name = 'SIR1'
        >>> BaseModel(name, param)
        """

        self.name = name
        self._check_name()
        self.comment = comment

        # solver communicationkernel
        self.modeltype = "basemodel"
        self.independent = []
        self._independent_values = {}
        self.parameters = parameters
        self.variables = []
        self.variables_of_interest = []
        self._initial_up_to_date = False

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
              "\n Variables of interest: \n" + str(self.variables_of_interest) +
              "\n Parameters: \n" + str(self.parameters) +
              "\n Independent: \n" + str(self.independent) +
              "\n Model initialised: " + str(self._initial_up_to_date))

    def _check_system(self):
        """
        check sys ...
        define ODE, algebraic, pde vars seperately
        if in sys && not in parameter:
            WARNING: another var found
        check if parameter of choice and independent vars are found in the
        system/ variable list.
        """
        return NotImplementedError

    def _check_name(self):
        """
        check if model name is a string
        """
        if not isinstance(self.name, str):
            raise TypeError("model name is not a string")

    def _check_parameters(self):
        """
        check is type == dict
        check for a match between system and parameters

        see: _checkParinODEandAlg(self): in ode_generator.py
        """
        return NotImplementedError

    def set_parameter(self, parameter, value):
        """
        """
        # check the data type of the input
        if not isinstance(parameter, str):
            raise TypeError("Parameter is not given as a string")
        if not isinstance(value, tuple((float, int))):
            raise TypeError("Value is not given as a float/int")
        self.parameters[parameter] = float(value)

    def set_parameters(self, pardict):

        for par, parvalue in pardict.items():
            if par not in self.parameters:
                raise KeyError("Parameter {} not in the model "
                               "parameters".format(par))
            self.parameters[par] = parvalue

    def set_independent(self, independent_dict):
        """
        """
        self.independent = independent_dict.keys()
        self._independent_values = independent_dict

    def _check_for_independent(self):
        """
        """
        return NotImplementedError

    def set_variables_of_interest(self, variables_of_interest):
        """
        set the variables to be exported to the output
        """
        if self.variables_of_interest:
            warnings.warn("Warning: variables of interest are already given. "
                          "Overwriting original variables.")
        # test if the input is a list
        if isinstance(variables_of_interest, list):
            for element in variables_of_interest:
                # if the input is a list, check if all inputs are strings
                if not isinstance(element, str):
                    raise TypeError("Elements in list are not strings")
            self.variables_of_interest = variables_of_interest
        # test if the input is a string
        elif isinstance(variables_of_interest, str):
            self.variables_of_interest = [variables_of_interest]

        # if the input is no string nor list of strings, raise error
        else:
            raise TypeError("Input is not a string nor a list of strings")

        return self

    def get_summary(self):
        """
        returns summary of the model
            parameters
            variables & type
            events
            time steps (init defined user, event, measurements)
            initial conditions
            ready to run!
        """
        return NotImplementedError

    def initialize_model(self):
        """
        make string object of model (templating based would be preferred)
        adjust variables to cope with events:
            S & S_event ==> aggregation function
        make Solver object
        set verbose option
        """
        return NotImplementedError

    def run(self):
        """
        Run the model for the given set of parameters, indepentent variable
        values and output a datagrame with the variables of interest.

        """
        if not self._initial_up_to_date:
            self.initialize_model
        return NotImplementedError

    def plot(self):
        """
        plot dataframe
        """
        return NotImplementedError

    def save(self):
        """
        Saves the object to a file, cfr. pickle
        """
        return NotImplementedError

    @classmethod
    def load(cls):
        """
        Loads the object from a file, cfr. pickle
        """
        return cls(None)

    def export_to(environment):
        """
        Converts the model to be used in other environment

        Parameters
        -----------
        environment : matlab, openModelica, libSBML
        """
        return NotImplementedError



