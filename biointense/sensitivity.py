# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:21:40 2015

@author: timothy
"""
from __future__ import division
import numpy as np
import pandas as pd

import warnings

import matplotlib.pyplot as plt
from copy import deepcopy
from biointense.model import _BiointenseModel
import biointense.sensitivitydefinition as sensdef
from biointense.solver import OdeSolver, AlgebraicSolver

from itertools import product

def _get_sse(sens_plus, sens_min):
    """
    """
    sse = np.mean(((sens_plus - sens_min)**2))
    return sse

def _get_sae(sens_plus, sens_min):
    """
    """
    mre = np.mean(np.abs(sens_plus - sens_min))
    return mre

def _get_mre(sens_plus, sens_min):
    """
    """
    mre = np.max(np.abs((sens_plus[1:] - sens_min[1:])/sens_plus[1:]))
    return mre

def _get_sre(sens_plus, sens_min):
    """
    """
    sre = np.mean(np.abs(1 - sens_min[1:]/sens_plus[1:]))
    return sre

def _get_ratio(sens_plus, sens_min):
    """
    """
    ratio = np.max(np.abs(1 - sens_min[1:]/sens_plus[1:]))
    return ratio

SENS_QUALITY = {'SSE': _get_sse,
                'SAE': _get_sae,
                'MRE': _get_mre,
                'SRE': _get_sre,
                'RATIO': _get_ratio}


#==============================================================================
# class Sensitivity(object):
#     """
#     """
#
#     def __init__(self, model):
#         """
#         """
#         self.model = model
#
#     def get_sensitivity(self):
#         """perturbation
#         Solve model equations
#
#         return: For each independent value, returns state for all variables
#         """
#         return NotImplementedError
#==============================================================================


class LocalSensitivity(object):
    """
    """
    def __init__(self, model, parameters):
        """
        """
        self._model = model
        self._parameter_names = parameters
        self._parameter_values = self._get_parvals(parameters,
                                                   self._model.parameters)

    @property
    def model(self):
        return self._model

    @property
    def parameter_names(self):
        return self._parameter_names

    @property
    def parameter_values(self):
        return self._parameter_values

    @staticmethod
    def _get_parvals(parameter_list, parameter_dict):
        parval_array = np.empty(len(parameter_list))
        for i, par in enumerate(parameter_list):
            parval_array[i] = parameter_dict[par]
        return parval_array

    def _get_sensitivity(self, method='AS'):
        """
        """
        return NotImplemented

    def get_sensitivity(self, method='AS', as_dataframe=True):
        r"""
        Get numerical local sensitivity for the different parameters and
        variables of interest


        Parameters
        -----------
        method : 'AS'|'PRS'|'TRS'
            Three different ways of calculating the local senstivity are
            available: the absolute senstivity (AS), the parameter relative
            sensitivity and the total relative sensitivitiy (TRS).

            *Absolute Senstivity (AS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}


            *Parameter Relative Sensitivity (PRS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}\cdot\theta_j

            *Total Relative Senstivitity (TRS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}\cdot\frac{\partial \theta_j}{y_i}


        """
        local_sens = self._get_sensitivity(method=method)

        if as_dataframe:
            local_sens = local_sens.reshape([self._model._independent_len, -1])

            columns = pd.MultiIndex.from_tuples(list(product(self.model.variables_of_interest,
                                                             self.parameter_names)))

            local_sens = pd.DataFrame(local_sens, columns=columns)

            index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                              names=self.model._independent_names)
            local_sens.index = index

        return local_sens

    def _rescale_sensitivity(self, sens_array, scaling, cutoff=1e-16,
                             cutoff_replacement=1e-16):
        """
        """
        variables = self._model.variables_of_interest
        parameter_len = len(self._parameter_names)
        variable_len = len(self._model.variables_of_interest)

        if scaling not in ['AS', 'PRS', 'TRS']:
            raise Exception('This type of scaling is not known/implemented. '
                            'Please use AS, PRS or TRS scaling.')

        if scaling[1:] == 'RS':
            # Convert par_values to np.array with lenght = sensitivity
            par_values = np.ones([self._model._independent_len, 1,
                                  parameter_len])*self.parameter_values
            par_values = np.repeat(par_values, variable_len, axis=1)

            # Parameter relative sensitivity
            sens_array *= par_values

            if scaling == 'TRS':
                model_run = self._model._run().reshape([self._model._independent_len,
                                                        -1, 1])
                model_run = np.repeat(model_run, parameter_len, axis=-1)

                model_run[model_run < cutoff] = cutoff_replacement

                # Total relative sensitivity
                sens_array /= model_run

        return sens_array


class NumericalLocalSensitivity(LocalSensitivity):
    """

    Parameters
    -----------
    model : Model|AlgebraicModel


    Examples
    ---------
    >>> import numpy as np
    >>> from biointense import Model, NumericalLocalSensitivity
    >>> parameters = {'Km': 150.,     # mM
                      'Vmax': 0.768,  # mumol/(min*U)
                      'E': 0.68}      # U/mL
    >>> system = {'v': 'Vmax*S/(Km + S)',
                  'dS': '-v*E',
                  'dP': 'v*E'}
    >>> M1 = Model('Michaelis-Menten', system, parameters)
    >>> M1.initial_conditions = {'S':500., 'P':0.}
    >>> M1.independent = {'t': np.linspace(0, 2500, 10000)}
    >>> M1sens_num = NumericalLocalSensitivity(M1, parameters=['Km', 'Vmax'])
    >>> numsens = M1sens_num.get_sensitivity(method='AS')
    """
    def __init__(self, model, parameters=None, procedure="central"):
        """
        """
        self._model = model
        if parameters is None:
            parameters = model.parameters.keys()
        self._parameter_names = parameters
        self._parameter_values = self._get_parvals(parameters,
                                                   self._model.parameters)
        self._parameter_perturb = self._parameter_values.copy()

        self._procedure = None
        self.procedure = procedure
        self._initiate_par()
        self._initiate_var()

    def _initiate_forw_back_sens(self):
        """
        """
        dummy_np = np.empty([len(self.model._independent_values.values()[0]),
                             len(self.parameters),
                             len(self.model.variables_of_interest)])
        self._sens_forw = dummy_np.copy()
        self._sens_back = dummy_np.copy()

    def _initiate_par(self):
        """
        """
        self._par_order = {}
        for i, par in enumerate(self.parameters):
            self._par_order[par] = i

    def _initiate_var(self):
        """
        """
        self._var_order = {}
        for i, var in enumerate(self.model.variables_of_interest):
            self._var_order[var] = i

    @property
    def procedure(self):
        """
        """
        return self._procedure

    @procedure.setter
    def procedure(self, procedure):
        r"""
        Select which procedure (central, forward, backward) should be used to
        calculate the numerical local sensitivity.

        Parameters
        -----------
        procedure : 'forward'|'central'|'backward'
            Three different procedures are available, the central procedure
            allows to evaluate the numerical accuracy of the sensitivity. This
            is not the case for the forward and backward procedure. However,
            the additional information required for the central discretization
            comes at an extra computational cost.

            *Forward discretisation

            .. math:: \frac{\partial y_i(t, \theta_j)}{\partial \theta_j} = \frac{y(t, \theta_j + \Delta\theta_j) - y(t, \theta_j)}{\Delta\theta_j}

            *Central discretisation

            .. math:: \frac{\partial y_i(t, \theta_j)}{\partial \theta_j} = \frac{y(t, \theta_j + \Delta\theta_j) - y(t, \theta_j- \Delta\theta_j)}{2\Delta\theta_j}

            *Backward discretisation

            .. math:: \frac{\partial y_i(t, \theta_j)}{\partial \theta_j} = \frac{y(t, \theta_j) - y(t, \theta_j- \Delta\theta_j)}{\Delta\theta_j}

        """
        if procedure == "central":
            self._initiate_forw_back_sens()
        elif procedure in ("forward", "backward"):
            if self._sens_forw:
                del self._sens_forw
            if self._sens_back:
                del self._sens_back
        else:
            raise Exception("Procedure is not known, please choose 'forward', "
                            "'backward' or 'central'.")
        self._procedure = procedure

    @property
    def perturbation(self):
        return self._parameter_perturb

    @perturbation.setter
    def perturbation(self, perturbation_values):
        """
        Function to set perturbation for each of the parameters:

        Parameters
        -----------
        parameters: list|dict
            If parameters is a list, then the same perturbation factor is given
            to each of the parameters. If parameters is a dict, then each
            parameter can be given a different perturbation factor.

        perturbation_values: float|dict
            If perturbation_values is a float, all parameters are, this perturbation factor is used for all
            the parameters

        Examples
        ---------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from biointense import AlgebraicModel, NumericalLocalSensitivity
        >>> system = {'v': 'Vmax*S/(Km + S)'}
        >>> parameters = {'Vmax': 1e-2, 'Km': 0.4}
        >>> M1 = AlgebraicModel('Michaelis Menten', system, parameters)
        >>> M1.set_independent({'S': np.linspace(0., 5., 100.)})
        >>> # Select parameter for which a numerical sensitivity
        >>> M1sens = NumericalLocalSensitivity(M1, parameters=['Km'])
        >>> # Change the perturbation factor for Km
        >>> M1sens.set_perturbation(['Km'], perturbation=5e-1)
        >>> high_pert = M1sens.get_sensitivity()
        >>> # Decrease perturbation factor of Km by dictionary
        >>> M1sens.set_perturbation({'Km': 1e-7})
        >>> low_pert = M1sens.get_sensitivity()
        >>> plt.plot(high_pert)
        >>> plt.hold(True)
        >>> plt.plot(low_pert)
        >>> plt.legend(['Perturbation=5e-1', 'Perturbation=1e-7'])
        """
        self.parameters = []

        if isinstance(parameters, list):
            for par in parameters:
                self._parameter_values[par] = perturbation
        elif isinstance(parameters, dict):
            for par in parameters:
                self._parameter_values[par] = parameters[par]
        else:
            raise Exception('Parameters should be a list or a dict which is '
                            'valid for all parameters or a dict with for each '
                            'parameter the corresponding perturbation factor')

        for par, par_value in self._parameter_values.items():
            if par_value is not None:
                self.parameters.append(par)

        self._initiate_par()

    def _model_output_pert(self, parameter, perturbation):
        """
        """
        # Backup original parameter value
        orig_par_val = self.model.parameters[parameter]
        # Run model with parameter value plus perturbation
        self.model.set_parameter(parameter, orig_par_val*(1 + perturbation))
        model_output = self.model._run()
        # Reset parameter to original value
        self.model.set_parameter(parameter, orig_par_val)

        return model_output

    @staticmethod
    def _calc_sens(output_forw, output_back, parameter, perturbation):
        """
        """
        # Calculate sensitivity
        num_sens = (output_forw - output_back)/(perturbation*parameter)

        return num_sens

    def _get_num_sensitivity(self, output_std, parameter, perturbation):
        """
        """
        par_value = self.model.parameters[parameter]

        self._initiate_forw_back_sens()

        if self.procedure == "central":
            output_forw = self._model_output_pert(parameter, perturbation)
            output_back = self._model_output_pert(parameter, -perturbation)
            par_number = self._par_order[parameter]
            self._sens_forw[:, par_number, :] = \
                self._calc_sens(output_forw, output_std, par_value,
                                perturbation)
            self._sens_back[:, par_number, :] = \
                self._calc_sens(output_std, output_back, par_value,
                                perturbation)
            cent_sens = self._calc_sens(output_forw, output_back,
                                        par_value, 2*perturbation)
            output = cent_sens
        elif self.procedure == "forward":
            output_forw = self._model_output_pert(parameter, perturbation)

            forw_sens = self._calc_sens(output_forw, output_std,
                                        par_value, perturbation)
            output = forw_sens
        elif self.procedure == "backward":
            output_back = self._model_output_pert(parameter, -perturbation)

            back_sens = self._calc_sens(output_std, output_back,
                                        par_value, perturbation)
            output = back_sens
        else:
            raise Exception('Type of perturbation is not known, perturbation'
                            'should be central, forward, or backward')

        return output

    def _get_sensitivity(self, method='AS'):
        r"""
        Get numerical local sensitivity for the different parameters and
        variables of interest


        Parameters
        -----------
        method : 'AS'|'PRS'|'TRS'
            Three different ways of calculating the local senstivity are
            available: the absolute senstivity (AS), the parameter relative
            sensitivity and the total relative sensitivitiy (TRS).

            *Absolute Senstivity (AS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}


            *Parameter Relative Sensitivity (PRS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}\cdot\theta_j

            *Total Relative Senstivitity (TRS)

                .. math:: \frac{\partial y_i(t)}{\partial \theta_j}\cdot\frac{\partial \theta_j}{y_i}


        """
        output_std = self.model._run()
        num_sens = {}

        for par in self.parameters:
            num_sens[par] = self._get_num_sensitivity(output_std, par,
                                                  self._parameter_values[par])

        num_sens = pd.concat(num_sens, axis=1)
        num_sens = num_sens.reorder_levels([1, 0], axis=1).sort_index(axis=1)

        num_sens = self._rescale_sensitivity(num_sens, method)

        return num_sens

    def get_sensitivity_accuracy(self, criterion="SSE"):
        '''Quantify the sensitivity calculations quality

        Parameters
        -----------
        criterion name for evaluation of the sensitivity quality. One can
            choose between the Sum of Squared Errors (SSE),
            Sum of Absolute Errors (SAE), Maximum Relative Error (MRE),
            Sum or Relative Errors (SRE) or the ratio between the forward and
            backward sensitivity (RATIO). [1]_

        Returns
        --------
        acc_num_LSA : pandas.DataFrame
            The colums of the pandas DataFrame contain all the ODE variables,
            the index of the DataFrame contains the different model parameters.

        References
        -----------
        .. [1] Dirk J.W. De Pauw and Peter A. Vanrolleghem, Practical Aspects
               of Sensitivity Analysis for Dynamic Models
        '''
        if self.procedure != "central":
            raise Exception('The accuracy of the sensitivity function can only'
                            ' be estimated when using the central local'
                            ' sensitivity!')

        dummy_np = np.empty((len(self.parameters),
                             len(self.model.variables_of_interest)))
        acc_num_LSA = pd.DataFrame(dummy_np, index=self.parameters,
                                   columns=self.model.variables_of_interest)

        for var in self.model.variables_of_interest:
            var_num = self._var_order[var]
            sens_input = (self._sens_forw[:, :, var_num],
                          self._sens_back[:, :, var_num])
            if criterion in SENS_QUALITY:
                acc_num_LSA[var] = SENS_QUALITY[criterion](*sens_input)
            else:
                raise Exception("Criterion '" + criterion + "' is not a valid "
                                "criterion, please select one of following "
                                "criteria: SSE, SAE, MRE, SRE, RATIO")
        return acc_num_LSA

    def calc_quality_num_lsa(self, parameters, perturbation_factors,
                             criteria=['SSE', 'SAE', 'MRE', 'SRE', 'RATIO']):
        '''Quantify the sensitivity calculations quality

        Parameters
        -----------
        parameters : list
            list containing the strings of the different parameters of interest
        perturbation_factors : float|list|np.array
            Contains the different perturbation factors for which the different
            criteria need to be evaluated.
        criteria : list
            criteria for evaluation of the sensitivity quality. One can
            choose between the Sum of Squared Errors (SSE),
            Sum of Absolute Errors (SAE), Maximum Relative Error (MRE),
            Sum or Relative Errors (SRE) or the ratio between the forward and
            backward sensitivity (RATIO) or a combination of all of them. [1]_

        Returns
        --------
        res : pandas.DataFrame
            The colums of the pandas DataFrame contain all the variables,
            with the different parameters as subcolumns. The index of the
            DataFrame contains the different perturbation factors.

        References
        -----------
        .. [1] Dirk J.W. De Pauw and Peter A. Vanrolleghem, Practical Aspects
               of Sensitivity Analysis for Dynamic Models
        '''
        if isinstance(perturbation_factors, float):
            perturbation_factors = [perturbation_factors]
        elif not isinstance(perturbation_factors, tuple([np.ndarray, list])):
            raise Exception('perturbation_factors need to be a float (for one '
                            'value) or list of floats (multiple values)!')

        if isinstance(criteria, str):
            criteria = [criteria]
        elif not isinstance(criteria, list):
            raise Exception('criteria need to be a string (for one quality '
                            'measure) or list of strings (multiple quality '
                            'measures)!')

        res = {crit: {} for (crit) in criteria}
        for pert in perturbation_factors:
            self.set_perturbation(parameters, pert)
            num_sens = self.get_sensitivity()
            for crit in criteria:
                acc = self.get_sensitivity_accuracy(criterion=crit)
                res[crit][pert] = acc.transpose().stack()

        for crit in criteria:
            res[crit] = pd.DataFrame(res[crit]).transpose()

        # Combine all different DataFrames
        res = pd.concat(res, axis=1)
        # Reorder dataframe to have following order:
        # output / parameter / criterion
        # plotting can be achieved much more easy!
        res = res.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)

        return res

class DirectLocalSensitivity(LocalSensitivity):
    r"""
    Parameters
    -----------
    model : Model|AlgebraicModel


    Examples
    ---------
    >>> import numpy as np
    >>> from biointense import Model, DirectLocalSensitivity
    >>> parameters = {'Km': 150.,     # mM
                      'Vmax': 0.768,  # mumol/(min*U)
                      'E': 0.68}      # U/mL
    >>> system = {'v': 'Vmax*S/(Km + S)',
                  'dS': '-v*E',
                  'dP': 'v*E'}
    >>> M1 = Model('Michaelis-Menten', system, parameters)
    >>> M1.initial_conditions = {'S':500., 'P':0.}
    >>> M1.independent = {'t': np.linspace(0, 2500, 10000)}
    >>> M1sens_direct = DirectLocalSensitivity(M1, parameters=['Km', 'Vmax'])
    >>> sens_out = M1sens_direct.get_sensitivity(method='PRS')
    """
    def __init__(self, model, parameters=None):
        """
        """
        if not isinstance(model, _BiointenseModel):
            raise Exception("DirectLocalSensitivity can only be used for "
                            "(subclasses of the) _BiointenseModel class")

        if parameters is None:
            parameters = model.parameters.keys()

#==============================================================================
#         super(DirectLocalSensitivity, self).__init__(model, parameters)
#==============================================================================
        self._model = model
        self._parameter_names = parameters
        self._parameter_values = self._get_parvals(parameters,
                                                   self._model.parameters)

        self._dxdtheta_start = None
        self._dxdtheta_len = 0

        self._fun_alg = None
        self._fun_ode = None
        self._fun_alg_str = None
        self._fun_ode_str = None

        self._generate_sensitivity()

    @staticmethod
    def _flatten_list(some_list):
        return [item for sublist in some_list for item in sublist]

    @staticmethod
    def _get_ordered_values(var, functions):
        return [functions.get(i) for i in var]

    def _generate_sensitivity(self):
        """
        """
        odevar = self.model._ordered_var.get('ode', None)
        odefun = self.model.systemfunctions.get('ode', None)
        if odefun is not None:
            odefun_ord = self._get_ordered_values(odevar, odefun)
        else:
            odefun_ord = None
        algvar = self.model._ordered_var.get('algebraic', None)
        algfun = self.model.systemfunctions.get('algebraic', None)
        algfun_ord = self._get_ordered_values(algvar, algfun)

        if odevar:
            dfdtheta, dfdx, self._dxdtheta_start = sensdef.generate_ode_sens(
                odevar, odefun_ord, algvar, algfun_ord, self.parameter_names)
            self._dxdtheta_len = self._dxdtheta_start.size
            self._fun_ode_str =\
                sensdef.generate_ode_derivative_definition(
                    self.model, dfdtheta, dfdx, self.parameter_names)
            exec(self._fun_ode_str)
            self._fun_ode = fun_ode_lsa

        if algvar:
            dgdtheta, dgdx = sensdef.generate_alg_sens(odevar, odefun_ord,
                                                       algvar, algfun_ord,
                                                       self.parameter_names)
            self._fun_alg_str =\
                sensdef.generate_non_derivative_part_definition(
                    self.model, dgdtheta, dgdx, self.parameter_names)
            exec(self._fun_alg_str)
            self._fun_alg = fun_alg_lsa

    def _args_ode_function(self, fun, **kwargs):
        """
        """
        initial_conditions = [self.model.initial_conditions[var]
                              for var in self.model._ordered_var['ode']]
        initial_conditions += self._flatten_list(self._dxdtheta_start.tolist())
        args = (fun, initial_conditions,
                self.model._independent_values, (self.model.parameters,),)

        return args

    def _get_ode_sensitivity(self, procedure='odeint'):
        """
        """
        solver = OdeSolver(*self._args_ode_function(self._fun_ode))
        model_output = solver.solve(procedure=procedure)
        ode_values = model_output[:,:-self._dxdtheta_len]
        model_output = model_output[:,-self._dxdtheta_len:]

        dxdtheta = np.reshape(model_output, [-1,
                                             len(self.model.initial_conditions),
                                             len(self.parameter_names)])

#==============================================================================
#         #index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
#         #                                  names=self.model.independent)
#
#         columns = pd.MultiIndex.from_tuples(list(product(
#                         self.model._ordered_var['ode'],
#                         self.parameter_names)))
#                         #, sortorder=0)
#
#
#         indep_len = len(self.model._independent_values.values()[0])
#
# #        result = pd.DataFrame(model_output.reshape(indep_len, -1),
# #                              index=index, columns=columns)
#         result = pd.DataFrame(model_output.reshape(indep_len, -1),
#                               columns=columns)
#==============================================================================
        return ode_values, dxdtheta #model_output#result

    def _get_alg_sensitivity(self, ode_values=None, dxdtheta=None):
        """
        """
        solver = AlgebraicSolver(*self.model._args_alg_function(self._fun_alg),
                                 ode_values=ode_values, dxdtheta=dxdtheta)
        model_output = solver._solve_algebraic()

#==============================================================================
#         #index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
#         #                                  names=self.model.independent)
#
#         columns = pd.MultiIndex.from_tuples(list(product(
#                         self.model._ordered_var['algebraic'],
#                         self.parameter_names)))
#                         #, sortorder=0)
#
#         indep_len = len(self.model._independent_values.values()[0])
#
#         #result = pd.DataFrame(model_output.reshape(indep_len, -1),
#         #                      index=index, columns=columns)
#         result = pd.DataFrame(model_output.reshape(indep_len, -1),
#                               columns=columns)
#==============================================================================
        return model_output#result

    def _get_sensitivity(self, method='AS'):
        """
        """
        ode_values = None
        dxdtheta = None
        direct_alg_sens = None

        if self._fun_ode:
            ode_values, dxdtheta = self._get_ode_sensitivity()

        if self._fun_alg:
            direct_alg_sens = self._get_alg_sensitivity(ode_values=ode_values,
                                                        dxdtheta=dxdtheta)

        if ode_values is None:
            direct_sens = direct_alg_sens
        elif direct_alg_sens is None:
            direct_sens = dxdtheta
        else:
            direct_sens = np.concatenate((direct_alg_sens, dxdtheta), axis=1)

        direct_sens = direct_sens[:, self.model._variables_of_interest_index,
                                  :]

        return self._rescale_sensitivity(direct_sens, method)


#==============================================================================
#
#         direct_sens = pd.concat([direct_ode_sens, direct_alg_sens], axis=1)
#
#         direct_sens = self._rescale_sensitivity(direct_sens, method)
#==============================================================================

#==============================================================================
#         return direct_alg_sens, dxdtheta
#==============================================================================

#==============================================================================
# class GlobalSensitivity(Sensitivity):
#     """
#     """
#     def __init__(self, model):
#         """
#         """
#==============================================================================
