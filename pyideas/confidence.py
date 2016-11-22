# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:59:56 2015

@author: timothy
"""
import numpy as np
from scipy import stats
import pandas as pd

import warnings

from pyideas.sensitivity import (DirectLocalSensitivity,
                                 NumericalLocalSensitivity)


class BaseConfidence(object):
    """
    """

    def __init__(self, sens, sens_method='AS', cutoff=1e-16,
                 cutoff_replacement=1e-16):
        """
        """
        self.cutoff = cutoff
        self.cutoff_replacement = cutoff_replacement

        self._sens = sens
        self._model = sens.model

        self._independent = self._model.independent
        self._sens_method = sens_method

        self.par_relations = None
        self.uncertainty = None

    @property
    def variables(self):
        return self._model.variables_of_interest

    @property
    def parameter_names(self):
        return self._sens.parameter_names

    @property
    def parameter_values(self):
        return self._sens.parameter_values

    def _get_sensmatrix(self):
        """
        """
        sensmatrix = self._sens._get_sensitivity()

        return sensmatrix

    def _get_uncertainty(self):
        """
        """
        return self.uncertainty._get_uncertainty(self._model._run(),
                                                 self.variables)

    @staticmethod
    def _dotproduct(sensmatrix, weight):
        """
        """
        # dy/dp*weight
        dydx_weigth = np.einsum('ijk,ijl->ikl', sensmatrix, weight)
        # dy/dp*weigth*[dy/dp]^T
        dydx_weight_dydx = np.einsum('ijk,ikl->ijl', dydx_weigth, sensmatrix)

        return dydx_weight_dydx

    def _calc_FIM_time(self, sensmatrix, uncertainty):
        """
        Help function for get_FIM

        Parameters
        -----------
        ECM_PD: pandas DataFrame
            The pandas DataFrame needs to contain the error covariance matrix
            for the different measured outputs
        """

        # Perform FIM calculation
        # FIM = dy/dx*1/Q*[dy/dx]^T

        # Avoid division by zero (already done in uncertainty class)
#==============================================================================
#         uncertainty[uncertainty <= self.cutoff] = self.cutoff_replacement
#==============================================================================
        # Calculate inverse of ECM_PD
        # 1/Q
        MECM_inv = np.linalg.inv(np.eye(len(self.variables)) *
                                 np.atleast_3d(uncertainty))
        # Set all very low numbers to zero (just a precaution, so that
        # solutions would be the same as the old get_FIM method). This is
        # probably not necessary!
        MECM_inv[MECM_inv <= 1e-20] = 0.

        FIM_timestep = self._dotproduct(sensmatrix, MECM_inv)

        return FIM_timestep

    def get_FIM_time(self):
        """
        """
        FIM_time = self._calc_FIM_time(self._get_sensmatrix(),
                                       self._get_uncertainty())
        return FIM_time

    def get_FIM(self):
        """
        """
        # FIM = sum(FIM(t))
        return np.sum(self.get_FIM_time(), axis=0)

    def _calc_PEECM(self, FIM):
        """
        """
        try:
            PEECM = np.linalg.inv(FIM)
        except:
            warnings.warn('Pseudo inverse was used!')
            PEECM = np.linalg.pinv(FIM)
        return PEECM

    def get_PEECM(self):
        """
        """
        return self._calc_PEECM(self.get_FIM())

    def _parameter_confidence(self, n_p, FIM, alpha):
        """
        """

        PEECM = self._calc_PEECM(FIM)

        CI = np.zeros([PEECM.shape[1], 9])

        # Adapt order to the one used in confidence
        CI[:, 0] = self.parameter_values
        CI[:, 1] = np.sqrt(PEECM.diagonal())
        for i, variance in enumerate(np.array(PEECM.diagonal())):
            # TODO check whether sum or median or... should be used
            # TODO Check of de absolute waarde hier gebruikt mag worden!!!!
            CI[i, 2:4] = stats.t.interval(alpha, n_p,
                                          loc=CI[i, 0],
                                          scale=np.sqrt(abs(variance)))
            CI[i, 4] = stats.t.interval(alpha, n_p,
                                        scale=np.sqrt(abs(variance)))[1]
            CI[:, 5] = abs(CI[:, 4]/CI[:, 0])*100
            CI[:, 6] = CI[:, 0]/np.sqrt(abs(PEECM.diagonal()))
            CI[:, 7] = stats.t.interval(alpha, n_p)[1]
            CI[:, 8] = CI[:, 6] >= CI[i, 7]

        CI = pd.DataFrame(CI, columns=['value', 'std dev', 'lower', 'upper', 'delta',
                                       'percent', 't_value', 't_reference',
                                       'significant'],
                          index=self.parameter_names)

        return CI

    def get_parameter_confidence(self, alpha=0.95):
        """Calculate confidence intervals for all parameters

        Parameters
        -----------
        alpha: float
            confidence level of a two-sided student t-distribution (Do not
            divide the required alpha value by 2). For example for a confidence
            level of 0.95, the lower and upper values of the interval are
            calculated at 0.025 and 0.975.

        Returns
        --------
        CI: pandas DataFrame
            Contains for each parameter the value of the variable, lower and
            upper value of the interval, the delta value which represents half
            the interval and the relative uncertainty in percent.

        """
        dat_len = self._model._independent_len
        par_len = len(self._sens.parameter_names)
        n_p = dat_len - par_len

        FIM = self.get_FIM()

        CI = self._parameter_confidence(n_p, FIM, alpha)
        return CI

    def get_parameter_correlation(self):
        """Calculate correlations between parameters

        Returns
        --------
        R: pandas DataFrame
            Contains for each parameter the correlation with all the other
            parameters.

        """
        PEECM = self.get_PEECM()
        R = np.zeros(PEECM.shape)

        for i in range(0, len(PEECM)):
            for j in range(i, len(PEECM)):
                corr = PEECM[i, j]/(np.sqrt(PEECM[i, i]*PEECM[j, j]))
                R[i, j] = corr
                R[j, i] = corr

        R = pd.DataFrame(R, columns=self.parameter_names,
                         index=self.parameter_names)

        return R

    def get_MPECM(self):
        '''Calculate model prediction error covariance matrix

        Returns
        --------
        model_prediction_ECM: pandas DataFrame
            Contains for every timestep the corresponding model prediction
            error covariance matrix.

        '''
        # Parameter correlations should be taken into account, because these
        # lower the model output uncertainty
        PEECM = np.repeat(np.atleast_3d(self.get_PEECM()).T,
                          self._model._independent_len, axis=0)

        # PEECM = np.multiply(PEECM, np.eye(self._par_len))
        # TODO Check if results are ok when var are not measured at the
        # same time
        MPECM = self._dotproduct(np.rollaxis(self._get_sensmatrix(), 2, 1),
                                 PEECM)

        MPECM[MPECM <= self.cutoff] = self.cutoff_replacement

        return MPECM

    def get_model_confidence(self, alpha=0.95):
        '''Calculate confidence intervals for variables

        Parameters
        -----------
        alpha: float
            confidence level of a two-sided student t-distribution (Do not
            divide the required alpha value by 2). For example for a confidence
            level of 0.95, the lower and upper values of the interval are
            calculated at 0.025 and 0.975.

        Returns
        --------
        model_confidence: dict
            Contains for each variable a pandas DataFrame which contains for
            every timestep the value of the variable,lower and upper value of
            the interval, the delta value which represents half the interval
            and the relative uncertainty in percent.

        '''
        dat_len = self._model._independent_len
        par_len = len(self._sens.parameter_names)
        n_p = dat_len - par_len

        modeloutput = self._model._run()
        MPECM = self.get_MPECM()

        sigma = {}

        for i, var in enumerate(self.variables):
            sigma_var = np.zeros([dat_len, 5])

            sigma_var[:, 0] = modeloutput[:, i]

            uncertainty = stats.t.interval(alpha, n_p, loc=modeloutput[:, i],
                                           scale=np.sqrt(MPECM.diagonal(
                                               axis1=1, axis2=2))[:, 0])

            sigma_var[:, 1] = uncertainty[0]
            sigma_var[:, 2] = uncertainty[1]

            sigma_var[:, 3] = np.abs((sigma_var[:, 2] - sigma_var[:, 0]))
            # Create mask to avoid counter division by zero
            rel_uncertainty = np.divide(sigma_var[:, 3], sigma_var[:, 0])
            # sigma_var[:, 4] = np.zeros(self._data_len)
            sigma_var[:, 4] = np.abs(rel_uncertainty)*100

            sigma_var = pd.DataFrame(sigma_var, columns=['value', 'lower',
                                                         'upper', 'delta',
                                                         'percent'])#,
#                                     index=self.sens_PD.index)
            sigma[var] = sigma_var

        sigma = pd.concat(sigma, axis=1)

        return sigma

    def get_model_correlation(self, alpha=0.95):
        '''Calculate correlation between variables

        Parameters
        -----------
        alpha: float
            confidence level of a two-sided student t-distribution (Do not
            divide the required alpha value by 2). For example for a confidence
            level of 0.95, the lower and upper values of the interval are
            calculated at 0.025 and 0.975.

        Returns
        --------
        model_correlation: pandas DataFrame
            For every possible combination of variables a column is made which
            represents the correlation between the two variables.
        '''

        MPECM = self.get_MPECM()

        corr = np.empty([self._model._independent_len,
                         len(self.variables),
                         len(self.variables)])
        # Calculation can further be optimised by only calculating unique
        # correlations, i.e. not for i,i (is definitely 1) and also not for
        # i,j and j,i since the matrix is symmetric.
        for i, var1 in enumerate(self.variables):
            for j, var2 in enumerate(self.variables):
                corr[:, i, j] = (MPECM[:, i, j] /
                                 np.sqrt(MPECM[:, i, i]*MPECM[:, j, j]))

        return corr


class CalibratedConfidence(BaseConfidence):
    """
    """
    def __init__(self, calibrated, sens_method='AS'):
        """
        """
        if calibrated.model.modeltype is "_BiointenseModel":
            super(self.__class__, self).__init__(DirectLocalSensitivity(
                calibrated.model, calibrated.dof), sens_method=sens_method)
        else:
            super(self.__class__, self).__init__(NumericalLocalSensitivity(
                calibrated.model, calibrated.dof))

        self._measurements = calibrated.measurements
        self._data = calibrated.measurements._data
        self.uncertainty = calibrated.measurements._uncertainty

    def _get_uncertainty(self):
        """
        """
        try:
            # Check if conditions have been altered
            # For calibration => No
            # For OED => Yes
            np.testing.assert_equal(self._model._independent_values,
                                    self._measurements._independent_values)
            return self._measurements._meas_uncertainty
        except AssertionError:
            return self.uncertainty._get_uncertainty(self._model._run(),
                                                     self.variables)


class TheoreticalConfidence(BaseConfidence):
    """
    """

    def __init__(self, sens, uncertainty, sens_method='AS'):
        """
        """
        super(self.__class__, self).__init__(sens, sens_method=sens_method)
        self.uncertainty = uncertainty

#==============================================================================
#     # TODO!
#     @classmethod
#     def from_calibrated(cls, calibrated_confidence):
#         temp = cls(calibrated_confidence.sens,
#                    calibrated_confidence.uncertainty,
#                    sens_method=calibrated_confidence.sens_method)
#         return temp
#
#==============================================================================


