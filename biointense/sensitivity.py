# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:21:40 2015

@author: timothy
"""
from __future__ import division
import numpy as np
import pandas as pd


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


class NumericalLocalSensitivity(LocalSensitivity):
    """
    """
    def __init__(self, model, perturb_parameters, perturbation=1e-6,
                 procedure="central"):
        """
        """
        self.model = model
        self.perturb_parameters = {}
        self.set_perturbation(perturb_parameters, perturbation=perturbation)
        self.set_procedure(procedure)
        self._initiate_par()
        self._initiate_var()

    def _initiate_forw_back_sens(self):
        """
        """
        dummy_np = np.empty([len(self.model.independent.values()[0]),
                             len(self.perturb_parameters),
                             len(self.model.variables_of_interest)])
        self._sens_forw = dummy_np.copy()
        self._sens_back = dummy_np.copy()

    def _initiate_par(self):
        """
        """
        self._par_order = {}
        for i, par in enumerate(self.perturb_parameters):
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
        res = {}
        if isinstance(parameters, list):
            for par in parameters:
                res[par] = perturbation
        elif isinstance(parameters, dict):
            res = parameters
        else:
            raise Exception('Parameters should be a list or a dict which is '
                            'valid for all parameters or a dict with for each '
                            'parameter the corresponding perturbation factor')
        self.perturb_parameters = res
        self._initiate_par()

    def update_perturbation(self, parameters, perturbation=1e-6):
        """
        Keep all current perturbations: only overwrite the ones which are set
        """
        if isinstance(parameters, list):
            for par in parameters:
                self.perturb_parameters[par] = perturbation
        elif isinstance(parameters, dict):
            self.perturb_parameters = parameters
        else:
            raise Exception('Parameters should be a list or a dict which is '
                            'valid for all parameters or a dict with for each '
                            'parameter the corresponding perturbation factor')

    def remove_perturbation(self, parameters):
        """
        Keep all current perturbations: only overwrite the ones which are set
        """
        if isinstance(parameters, list):
            for par in parameters:
                try:
                    del self.perturb_parameters[par]
                except NameError:
                    print(par + "is currently not a perturbed parameter, so "
                          "ignoring the remove option for this parameter.")
        else:
            raise Exception('Parameters should be a list or a dict which is '
                            'valid for all parameters or a dict with for each '
                            'parameter the corresponding perturbation factor')
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

    def _get_sensitivity(self, parameter, perturbation):
        """
        """
        output_std = self.model.run()
        par_value = self.model.parameters[parameter]

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
            cent_sens = self._calc_sens(output_std, output_back,
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

    def get_sensitivity(self):
        """
        Get numerical local sensitivity for the different parameters and
        variables of interest
        """
        num_sens = {}
        for par in self.perturb_parameters.items():
            num_sens[par[0]] = self._get_sensitivity(par[0], par[1])

        num_sens = pd.concat(num_sens, axis=1)
        num_sens = num_sens.reorder_levels([1, 0], axis=1).sort_index(axis=1)

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

        dummy_np = np.empty((len(self.perturb_parameters),
                             len(self.model.variables_of_interest)))
        acc_num_LSA = pd.DataFrame(dummy_np, index=self.perturb_parameters,
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
    def __init__(self, model):
        """
        """


class GlobalSensitivity(Sensitivity):
    """
    """
    def __init__(self, model):
        """
        """
