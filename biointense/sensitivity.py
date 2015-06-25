# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:21:40 2015

@author: timothy
"""
from __future__ import division
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from copy import deepcopy
from biointense.model import _BiointenseModel
import biointense.sensitivitydefinition as sensdef
from biointense.solver import OdeSolver, AlgebraicSolver

from itertools import product


class Sensitivity(object):
    """
    """

    def __init__(self, model):
        """
        """
        self.model = model

    def get_sensitivity(self):
        """perturbation
        Solve model equations

        return: For each independent value, returns state for all variables
        """
        return NotImplementedError


class LocalSensitivity(Sensitivity):
    """
    """
    def __init__(self, model, parameters):
        """
        """
        self.model = model
        self.parameters = parameters

    @property
    def parameter_values(self):
            return self.model.parameters

    def _rescale_sensitivity(self, sensitivity_PD, scaling):
        """
        """
        variables = list(sensitivity_PD.columns.levels[0])
        parameters = list(sensitivity_PD.columns.levels[1])
        perturb_par = pd.Series(self.parameter_values)[self.parameters]
        sensitivity_len = len(sensitivity_PD.index)
        parameter_len = len(parameters)

        # Problem with keeping the same order!
        par_values = []
        for par in parameters:
            par_values.append(self.model.parameters[par])
        # Convert par_values to np.array with lenght = sensitivity
        par_values = np.array(par_values)*np.ones([sensitivity_len,
                                                   parameter_len])

        if scaling == 'CPRS':
            # CPRS = CAS*parameter
            for var in variables:
                sensitivity_PD[var] = sensitivity_PD[var]*par_values
        elif scaling == 'CTRS':
            # CTRS
            if min(sensitivity_PD.mean()) == 0 or max(sensitivity_PD.mean()) == 0:
                raise Exception(scaling + ': It is not possible to use the '
                                'CTRS method for calculating sensitivity, '
                                'because one or more variables are fixed at '
                                'zero. Try to use another method or to change '
                                'the independent/initial conditions!')
            elif min(sensitivity_PD.min()) == 0 or min(sensitivity_PD.max()) == 0:
                for var in variables:
                    sensitivity_PD[var] = sensitivity_PD[var]*par_values/sensitivity_PD[var].mean()
            else:
                for var in variables:
                    sensitivity_PD[var] = sensitivity_PD[var]*par_values/np.tile(np.array(sensitivity_PD[var]),(len(par_values),1)).T
        elif scaling != 'CAS':
            raise Exception('You have to choose one of the sensitivity '
                            'methods which are available: CAS, CPRS or CTRS')

        return sensitivity_PD


class NumericalLocalSensitivity(LocalSensitivity):
    """
    """
    def __init__(self, model, parameters=None, perturbation=1e-6,
                 procedure="central"):
        """
        """
        self.model = model
        self.parameters = []
        self._parameter_values = {}.fromkeys(self.model.parameters.keys())
        if parameters is None:
            self.set_perturbation(list(model.parameters.keys()),
                                  perturbation=perturbation)
        else:
            self.set_perturbation(parameters,
                                  perturbation=perturbation)
        self.set_procedure(procedure)
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

    def set_procedure(self, procedure):
        """
        Select which procedure (central, forward, backward) should be used to
        calculate the numerical local sensitivity.
        """
        if procedure == "central":
            self._initiate_forw_back_sens()
        elif procedure in ("forward", "backward"):
            try:
                del self._sens_forw
                del self._sens_back
            except NameError:
                pass
        else:
            raise Exception("Procedure is not known, please choose 'forward', "
                            "'backward' or 'central'.")
        self.procedure = procedure

    def set_perturbation(self, parameters, perturbation=1e-6):
        """
        Ignore current perturbations
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
        model_output = self.model.run()
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

    def _get_sensitivity(self, output_std, parameter, perturbation):
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

    def get_sensitivity(self, method='CAS'):
        """
        Get numerical local sensitivity for the different parameters and
        variables of interest
        """
        output_std = self.model.run()
        num_sens = {}

        for par in self.parameters:
            num_sens[par] = self._get_sensitivity(output_std, par,
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
            if criterion == 'SSE':
                acc_num_LSA[var] = self._get_sse(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'SAE':
                acc_num_LSA[var] = self._get_sae(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'MRE':
                acc_num_LSA[var] = self._get_mre(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'SRE':
                acc_num_LSA[var] = self._get_sre(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'RATIO':
                acc_num_LSA[var] = self._get_ratio(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
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
        criteria : SSE|SAE|MRE|SRE|RATIO
            criterion name for evaluation of the sensitivity quality. One can
            choose between the Sum of Squared Errors (SSE),
            Sum of Absolute Errors (SAE), Maximum Relative Error (MRE),
            Sum or Relative Errors (SRE) or the ratio between the forward and
            backward sensitivity (RATIO). [1]_

        Returns
        --------
        res : pandas.DataFrame
            The colums of the pandas DataFrame contain all the ODE variables,
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

    @staticmethod
    def _get_sse(sens_plus, sens_min):
        """
        """
        sse = np.mean(((sens_plus - sens_min)**2))
        return sse

    @staticmethod
    def _get_sae(sens_plus, sens_min):
        """
        """
        mre = np.mean(np.abs(sens_plus - sens_min))
        return mre

    @staticmethod
    def _get_mre(sens_plus, sens_min):
        """
        """
        mre = np.max(np.abs((sens_plus[1:] - sens_min[1:])/sens_plus[1:]))
        return mre

    @staticmethod
    def _get_sre(sens_plus, sens_min):
        """
        """
        sre = np.mean(np.abs(1 - sens_min[1:]/sens_plus[1:]))
        return sre

    @staticmethod
    def _get_ratio(sens_plus, sens_min):
        """
        """
        ratio = np.max(np.abs(1 - sens_min[1:]/sens_plus[1:]))
        return ratio


class DirectLocalSensitivity(LocalSensitivity):
    """
    """
    def __init__(self, model, parameters=None):
        """
        """
        if not isinstance(model, _BiointenseModel):
            raise Exception("DirectLocalSensitivity can only be used for "
                            "(subclasses of the) _BiointenseModel class")

        self.model = model
        self.parameters = parameters or self.model.parameters.keys()

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
        odefun_ord = self._get_ordered_values(odevar, odefun)
        algvar = self.model._ordered_var.get('algebraic', None)
        algfun = self.model.systemfunctions.get('algebraic', None)
        algfun_ord = self._get_ordered_values(algvar, algfun)

        if odevar:
            dfdtheta, dfdx, self._dxdtheta_start = sensdef.generate_ode_sens(
                odevar, odefun_ord, algvar, algfun_ord, self.parameters)
            self._dxdtheta_len = self._dxdtheta_start.size
            self._fun_ode_str = sensdef.generate_ode_derivative_definition(
                                   self.model, dfdtheta, dfdx, self.parameters)
            exec(self._fun_ode_str)
            self._fun_ode = fun_ode_lsa

        if algvar:
            dgdtheta, dgdx = sensdef.generate_alg_sens(odevar, odefun_ord,
                                                       algvar, algfun_ord,
                                                       self.parameters)
            self._fun_alg_str = sensdef.generate_non_derivative_part_definition(
                                   self.model, dgdtheta, dgdx, self.parameters)
            exec(self._fun_alg_str)
            self._fun_alg = fun_alg_lsa

    def _args_ode_function(self, fun, **kwargs):
        """
        """
        externalfunctions = kwargs.get('externalfunctions')
        initial_conditions = [self.model.initial_conditions[var]
                              for var in self.model._ordered_var['ode']]
        initial_conditions += self._flatten_list(self._dxdtheta_start.tolist())
        args = (fun, initial_conditions,
                self.model._independent_values)
        if externalfunctions:
            args += tuple(((self.model.parameters, externalfunctions,),))
        else:
            args += tuple(((self.model.parameters,),))

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
                                             len(self.parameters)])

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)

        columns = pd.MultiIndex.from_tuples(list(product(
                        self.model._ordered_var['ode'],
                        self.parameters)))
                        #, sortorder=0)


        indep_len = len(self.model._independent_values.values()[0])

        result = pd.DataFrame(model_output.reshape(indep_len, -1),
                              index=index, columns=columns)
        return ode_values, dxdtheta, result

    def _get_alg_sensitivity(self, ode_values=None, dxdtheta=None):
        """
        """
        solver = AlgebraicSolver(*self.model._args_alg_function(self._fun_alg),
                                 ode_values=ode_values, dxdtheta=dxdtheta)
        model_output = solver._solve_algebraic()

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)

        columns = pd.MultiIndex.from_tuples(list(product(
                        self.model._ordered_var['algebraic'],
                        self.parameters)))
                        #, sortorder=0)

        indep_len = len(self.model._independent_values.values()[0])

        result = pd.DataFrame(model_output.reshape(indep_len, -1),
                              index=index, columns=columns)
        return result

    def get_sensitivity(self, method='CAS'):
        """
        """
        ode_values = None
        dxdtheta = None
        direct_ode_sens = None
        direct_alg_sens = None

        if self._fun_ode:
            ode_values, dxdtheta, direct_ode_sens = self._get_ode_sensitivity()

        if self._fun_alg:
            direct_alg_sens = self._get_alg_sensitivity(ode_values=ode_values,
                                                        dxdtheta=dxdtheta)

        direct_sens = pd.concat([direct_ode_sens, direct_alg_sens], axis=1)

        direct_sens = self._rescale_sensitivity(direct_sens, method)

        return direct_sens

class GlobalSensitivity(Sensitivity):
    """
    """
    def __init__(self, model):
        """
        """
