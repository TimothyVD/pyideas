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
    def __init__(self, model, parameters, perturbation_factor=1e-6,
                 procedure="central"):
        """
        """
        self.model = model
        self.parameters = parameters
        self.perturbation_factor = perturbation_factor
        self.procedure = procedure
        self._par_order = {}
        for i, par in enumerate(parameters):
            self._par_order[par] = i
        self._var_order = {}
        for i, var in enumerate(self.model.variables):
            self._var_order[var] = i

        dummy_np = np.empty([len(self.model.xdata), len(parameters,
                             len(self.model.variables))])
        self._sens_forw = dummy_np.copy()
        self._sens_back = dummy_np.copy()

    def _model_output_pert(self, parameter, perturbation):
        # Backup original parameter value
        orig_par_val = self.model.parameters[parameter]
        # Run model with parameter value plus perturbation
        self.model.parameters[parameter] = \
            orig_par_val*(1 + perturbation)
        model_output = self.model.run()
        # Reset parameter to original value
        self.model.parameters[parameter] = orig_par_val

        return model_output

    def _calc_sens(self, output_forw, output_back, parameter, perturbation):
        """
        """
        # Calculate sensitivity
        num_sens = (output_forw - output_back)/(perturbation*parameter)

        return num_sens

    def _get_sensitivity(self, parameter, perturbation):
        """
        """
        output_std = self.model.run()

        if self.procedure == "central":
            output_forw = self._model_output_pert(parameter, perturbation)
            output_back = self._model_output_pert(parameter, -perturbation)

            par_number = self._par_order[parameter]
            self._sens_forw[:, par_number, :] = \
                self._calc_sens(output_forw, output_std, parameter,
                                perturbation)
            self._sens_back[:, par_number, :] = \
                self._calc_sens(output_std, output_back, parameter,
                                perturbation)
            cent_sens = self._calc_sens(output_std, output_back,
                                        parameter, 2*perturbation)
            output = cent_sens
        elif self.procedure == "forward":
            output_forw = self._model_output_pert(parameter, perturbation)

            forw_sens = self._calc_sens(output_forw, output_std,
                                        parameter, perturbation)
            output = forw_sens
        elif self.procedure == "backward":
            output_back = self._model_output_pert(parameter, -perturbation)

            back_sens = self._calc_sens(output_std, output_back,
                                        parameter, perturbation)
            output = back_sens
        else:
            raise Exception('Type of perturbation should be central, forward,\
                            or backward')

        return output

    def get_sensitivity(self):
        """
        """
        if isinstance(self.perturbation_factor, float):
            for par in self.parameters:
                num_sens = self._get_sensitivity(par, self.perturbation_factor)
        elif isinstance(self.perturbation_factor, list):
            for i, par in enumerate(self.parameters):
                num_sens = \
                    self._get_sensitivity(par, self.perturbation_factor[i])
        else:
            raise TypeError("parameters should be a string or a list of\
                            strings.")

        return num_sens

    def get_sensitivity_accuracy(self, criterion="SSE"):
        '''Quantify the sensitivity calculations quality

        Parameters
        -----------
        criterion : SSE|SAE|MRE|SRE
            criterion name for evaluation of the sensitivity quality. One can
            choose between the Sum of Squared Errors (SSE),
            Sum of Absolute Errors (SAE), Maximum Relative Error (MRE) or
            Sum or Relative Errors (SRE).

        Returns
        --------
        acc_num_LSA : pandas.DataFrame
            The colums of the pandas DataFrame contain all the ODE variables,
            the index of the DataFrame contains the different model parameters.
        '''
        if self.procedure != "central":
            raise Exception('The accuracy of the sensitivity function can only\
                be estimated when using the central local senstivitity!')

        dummy_np = np.empty((len(self.parameters), len(self.model.variables)))
        acc_num_LSA = pd.DataFrame(dummy_np, index=self.parameters,
                                   columns=self.model.variables)

        for var in self.model.variables:
            var_num = self._var_order[var]
            if criterion == 'SSE':
                acc_num_LSA[var_num] = self._calc_SSE(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'SAE':
                acc_num_LSA[var_num] = self._calc_SAE(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'MRE':
                acc_num_LSA[var_num] = self._calc_MRE(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            elif criterion == 'SRE':
                acc_num_LSA[var_num] = self._calc_SRE(
                    self._sens_forw[:, :, var_num],
                    self._sens_back[:, :, var_num])
            else:
                raise Exception("Criterion '" + criterion + "' is not a valid\
                    criterion, please select one of following criteria:\
                    SSE, SAE, MRE, SRE")
        return acc_num_LSA

    @staticmethod
    def _calc_SSE(sens_plus, sens_min):
        """
        """
        SSE = np.mean(((sens_plus - sens_min)**2))
        return SSE

    @staticmethod
    def _calc_SAE(sens_plus, sens_min):
        """
        """
        SAE = np.mean(np.abs(sens_plus - sens_min))
        return SAE

    @staticmethod
    def _calc_MRE(sens_plus, sens_min):
        """
        """
        MRE = np.max(np.abs((sens_plus - sens_min)/sens_plus))
        return MRE

    @staticmethod
    def _calc_SRE(sens_plus, sens_min):
        """
        """
        SRE = np.mean(np.abs(1 - sens_min/sens_plus))
        return SRE


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
