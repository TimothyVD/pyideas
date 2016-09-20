from __future__ import division
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from biointense.solver import AlgebraicSolver


FILE_EXTENSION = '.biointense'


class BaseModel(object):

    def __init__(self, name, parameters, variables, indep_names, fun):
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

        # solver communicationkernel
        self.modeltype = "basemodel"
        self._independent_names = indep_names
        self._independent_values = None
        self._independent_len = None
#        self._independent_len = None
        # Parameters
        self._parameters = None
        self.parameters = parameters
        # Model output
        self._variables = variables
        self._variables_of_interest = variables
        self._variables_of_interest_index = range(len(variables))
        # function
        self.fun = fun
        # Model initialised?
        self._initialised = False

    @staticmethod
    def _make_OrderedDict(mydict):
        r"""
        Check if instance is orderedDict, otherwise make it orderedDict
        Parameters
        -----------
        mydict: dict|orderedDict
        Returns
        --------
        OrderedDict
        """

        if not isinstance(mydict, OrderedDict):
            mydict = OrderedDict(sorted(mydict.items(), key=lambda t: t[0]))

        return mydict

    @staticmethod
    def _get_index_from_lists(reflist, newlist):
        """
        """
        array = np.empty([len(newlist)], dtype=int)
        for i, string in enumerate(newlist):
            array[i] = reflist.index(string)

        return array

    @property
    def parameters(self):
        """
        """
        return self._parameters

    @parameters.setter
    def parameters(self, pardict):
        """
        """
        if self._parameters is None:
            if isinstance(pardict, OrderedDict):
                self._parameters = pardict
            else:
                self._parameters = OrderedDict(sorted(pardict.items(),
                                                      key=lambda t: t[0]))

        # Check whether the parameters which have to be updated are available
        # in the model
        if set(pardict.keys()).issubset(self._parameters.keys()):
            self._parameters.update(pardict)
        else:
            raise KeyError("The defined parameters are not all available in "
                           "the model. Check for typos or reinitialise the "
                           "model.")

    @property
    def variables(self):
        """
        """
        return self._variables

    @property
    def variables_of_interest(self):
        """
        """
        return self._variables_of_interest

    @variables_of_interest.setter
    def variables_of_interest(self, varlist):
        """
        """
        if set(varlist).issubset(self._variables):
            self._variables_of_interest = varlist
            self._variables_of_interest_index = \
                self._get_index_from_lists(self._variables, varlist)
        else:
            raise KeyError("The defined variables are not all available in "
                           "the model. Check for typos or reinitialise the "
                           "model.")

    @property
    def independent(self):
        """
        """
        return self._independent_values

    @independent.setter
    def independent(self, independent_dict):
        """
        """
        sorted_indep = []
        for indep in self._independent_names:
            sorted_indep.append((indep, independent_dict[indep]))
        self._independent_values = OrderedDict(sorted_indep)
        self._independent_len = len(self._independent_values.values()[0])
        #self._independent_names = self._independent_values.keys()
#        self._independent_len = np.empty(len(self._independent_names),
         #                                dtype=int)
#        for i, indep in enumerate(self._independent_names):
#            self._independent_len[i] = len(self._independent_values[indep])

    def _run(self):
        """
        Run the model for the given set of parameters, indepentent variable
        values and output a dataframe with the variables of interest.

        """
        solver = AlgebraicSolver(self.fun, self.independent,
                                 (self.parameters,))

        result = solver.solve()

        return result[:, self._variables_of_interest_index]

    def run(self, procedure=None):
        """
        Run the model for the given set of parameters, independent variable
        values and output a datagrame with the variables of interest.
        """
        result = self._run()

        result = pd.DataFrame(result, columns=self._variables_of_interest)

        index = pd.MultiIndex.from_arrays(self._independent_values.values(),
                                          names=self._independent_names)

        result.index = index

        return result

    def save(self, filename):
        """
        Saves the object to a file, cfr. pickle

        Parameters
        -----------
        filename: str
            String with the (relative/absolute) path to the file. If the
            filename does not end with *.biointense*, this is automatically
            appended to the filename.
        """
        if not filename.endswith(FILE_EXTENSION):
            filename += FILE_EXTENSION
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print('Object has been save as {0}'.format(filename))

    @classmethod
    def load(cls, filename):
        """
        Loads the object from a file, cfr. pickle

        Parameters
        -----------
        filename: str
            String with the (relative/absolute) path to the file. If the
            filename does not end with *.biointense*, this is automatically
            appended to the filename.

        Returns
        --------
        object: biointense.Model|biointense.AlgebraicModel
            Model with containing all values and functions as it was saved.
        """
        if not filename.endswith(FILE_EXTENSION):
            filename += FILE_EXTENSION
        with open(filename, 'rb') as inputfile:
            temp_object = pickle.load(inputfile)
        return temp_object

#    def export_to(environment):
#        """
#        Converts the model to be used in other environment
#
#        Parameters
#        -----------
#        environment : matlab, openModelica, libSBML
#        """
#        return NotImplementedError
