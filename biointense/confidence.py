# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:59:56 2015

@author: timothy
"""
import numpy as np
from scipy import stats
import pandas as pd
from copy import deepcopy

from biointense.sensitivity import DirectLocalSensitivity

#def get_error_pd( )


class BaseConfidence(object):
    """
    """

    def __init__(self, sens, sens_method='CAS'):
        """
        """
        self.sens = sens
        # self.sens_PD = sens.get_sensitivity(method='CAS')
        self.model = sens.model
        # self.model_output = self.model._run()
        self.independent = self.model.independent
        self.sens_method = sens_method

        self.variables = list(self.sens_PD.columns.levels[0])
        self.parameters = deepcopy(self.sens.parameters)

        self.repeats_per_sample = 1
        # self.parameter_values = pd.Series({par: self.model.parameters[par] for
        #                                   par in self.parameters})

        self._sens_matrix = None

        self._par_len = len(self.parameters)
        self._var_len = len(self.variables)

        self.par_relations = None

        self._FIM = None
        # Measurement Error Covariance Matrix
        self._MECM = None
        # Parameter Estimation Error Covariance Matrix
        self._PEECM = None
        # Model Prediction Error Covariance Matrix
        self._MPECM = None

    @property
    def sens_PD(self):
        sens_PD = self.sens._get_sensitivity(method=self.sens_method)
        # Temp fix!
        # Necessary because order is changing
        self.parameters = list(sens_PD.columns.get_level_values(1))

        return sens_PD

    @property
    def _data_len(self):
        return len(self.sens_PD.index)*self.repeats_per_sample

    @property
    def _mask_parameters(self):
        """
        """
        if self.par_relations is None:
            return 1.
        else:
            temp_all = np.concatenate(self.par_relations.values())
            unique_all = np.unique(temp_all)

            mask_dict = {}
            for par, exp in self.par_relations.items():
                mask_dict[par] = unique_all.copy()
                i = 0
                for val in mask_dict[par]:
                    if val in self.par_relations[par]:
                        mask_dict[par][i] = 1
                    else:
                        mask_dict[par][i] = 0
                    i += 1

            mask_array = None
            for par in self.parameters:
                if mask_array is None:
                    mask_array = np.atleast_2d(mask_dict[par])
                else:
                    mask_array = np.concatenate([mask_array,
                                                 np.atleast_2d(mask_dict[par])],
                                                axis=0)
            #combined = np.einsum('ij, kj-> jik', mask_array, mask_array)
            combined = np.atleast_3d(mask_array.T)
        return combined

    @property
    def model_output(self):
        return self.model._run()

    @property
    def parameter_values(self):
        return pd.Series({par: self.model.parameters[par] for par in self.parameters})

    def _sens_PD_2_matrix(self, sens_PD):
        """
        """
        len_index = len(sens_PD.index)

        return sens_PD.values.reshape([len_index, self._par_len,
                                       self._var_len])

    @property
    def sensmatrix(self):
        """
        """
        #if self._sens_matrix is None:
        sens_matrix = self._sens_PD_2_matrix(self.sens_PD)
        self._sens_matrix = self._mask_parameters*sens_matrix

        return self._sens_matrix

    @property
    def FIM(self):
        '''
        '''
        #if self._FIM is None:
        self._FIM, self._FIM_time = self._calc_FIM(self.sensmatrix,
                                                   self.uncertainty_PD)
        return self.repeats_per_sample*self._FIM

    def _calc_PEECM(self, FIM):
        '''
        '''
        try:
            PEECM = np.linalg.inv(FIM)
        except:
            raise Warning('Pseudo inverse was used!')
            PEECM = np.linalg.pinv(FIM)
        return PEECM

    @property
    def PEECM(self):
        '''
        '''
        #if self._PEECM is None:
        self._PEECM = self._calc_PEECM(self.FIM)
        return self._PEECM

    @property
    def FIM_time(self):
        '''
        '''
        #if self._FIM_time is None:
        self._FIM, self._FIM_time = self._calc_FIM(self.sensmatrix,
                                                   self.uncertainty_PD)
        return self.repeats_per_sample*self._FIM_time

    @FIM.deleter
    def FIM(self):
        self._FIM = None
        self._PEECM = None
        self._FIM_time = None

    @staticmethod
    def _dotproduct(sensmatrix, weight):
        """
        """
        # dy/dp*weight
        dydx_weigth = np.einsum('ijk,ikl->ijl', sensmatrix, weight)
        # [dy/dp]^T
        sensmatrix_t = np.rollaxis(sensmatrix, 2, 1)
        # dy/dp*weigth*[dy/dp]^T
        dydx_weight_dydx = np.einsum('ijk,ikl->ijl', dydx_weigth, sensmatrix_t)

        return dydx_weight_dydx

    def _calc_FIM(self, sensmatrix, uncertainty_PD):
        '''
        Help function for get_FIM

        Parameters
        -----------
        ECM_PD: pandas DataFrame
            The pandas DataFrame needs to contain the error covariance matrix
            for the different measured outputs
        '''

#        if sens_PD.index[0] == 0.0:
#            sens_start = 0
#        else:
#            sens_start = 1

        # Perform FIM calculation
        # FIM = dy/dx*1/Q*[dy/dx]^T

        # Calculate inverse of ECM_PD
        # 1/Q
        MECM_inv = np.linalg.inv(np.eye(self._var_len) *
                                 np.atleast_3d(uncertainty_PD))
        # Set all very low numbers to zero (just a precaution, so that
        # solutions would be the same as the old get_FIM method). This is
        # probably not necessary!
        MECM_inv[MECM_inv < 1e-20] = 0.

        FIM_timestep = self._dotproduct(sensmatrix, MECM_inv)

        # FIM = sum(FIM(t))
        FIM = np.sum(FIM_timestep, axis=0)
        return FIM, FIM_timestep

    def _parameter_confidence(self, n_p, FIM, alpha):
        '''
        '''

        PEECM = self._calc_PEECM(FIM)

        CI = np.zeros([PEECM.shape[1], 8])

        # Adapt order to the one used in confidence
        CI[:, 0] = self.parameter_values[self.parameters]
        for i, variance in enumerate(np.array(PEECM.diagonal())):
            # TODO check whether sum or median or... should be used
            # TODO Check of de absolute waarde hier gebruikt mag worden!!!!
            CI[i, 1:3] = stats.t.interval(alpha, n_p,
                                          loc=CI[i, 0],
                                          scale=np.sqrt(abs(variance)))
            CI[i, 3] = stats.t.interval(alpha, n_p,
                                        scale=np.sqrt(abs(variance)))[1]
        CI[:, 4] = abs(CI[:, 3]/CI[:, 0])*100
        CI[:, 5] = CI[:, 0]/np.sqrt(abs(PEECM.diagonal()))
        CI[:, 6] = stats.t.interval(alpha, n_p)[1]
        CI[:, 7] = CI[:, 5] >= CI[i, 6]

        CI = pd.DataFrame(CI, columns=['value', 'lower', 'upper', 'delta',
                                       'percent', 't_value', 't_reference',
                                       'significant'],
                          index=self.parameters)

        return CI

    def get_parameter_confidence(self, alpha=0.95):
        '''Calculate confidence intervals for all parameters

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

        '''
        n_p = self._data_len - self._par_len

        FIM = self.FIM

        CI = self._parameter_confidence(n_p, FIM, alpha)
        return CI

    def get_parameter_correlation(self):
        '''Calculate correlations between parameters

        Returns
        --------
        R: pandas DataFrame
            Contains for each parameter the correlation with all the other
            parameters.

        '''
        R = np.zeros(self.PEECM.shape)

        for i in range(0, len(self.PEECM)):
            for j in range(i, len(self.PEECM)):
                corr = self.PEECM[i, j]/(np.sqrt(self.PEECM[i, i] *
                                                 self.PEECM[j, j]))
                R[i, j] = corr
                R[j, i] = corr

        R = pd.DataFrame(R, columns=self.parameters, index=self.parameters)

        return R

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
        n_p = self._data_len - self._par_len

        sigma = {}

        for i, var in enumerate(self.variables):
            sigma_var = np.zeros([len(self.model_output[var]), 5])

            sigma_var[:, 0] = self.model_output[var].values

            uncertainty = stats.t.interval(alpha, n_p, loc=self.model_output[var].values,
                                           scale=np.sqrt(self.MPECM.diagonal(
                                           axis1=1, axis2=2))[:, 0])

            sigma_var[:, 1] = uncertainty[0]
            sigma_var[:, 2] = uncertainty[1]

            sigma_var[:, 3] = np.abs((sigma_var[:, 2] - sigma_var[:, 0]))
            # Create mask to avoid counter division by zero
            rel_uncertainty = np.divide(sigma_var[:, 3], sigma_var[:, 0])
            mask = np.isfinite(rel_uncertainty)
            # sigma_var[:, 4] = np.zeros(self._data_len)
            sigma_var[:, 4] = np.abs(rel_uncertainty[mask])*100

            sigma_var = pd.DataFrame(sigma_var, columns=['value', 'lower',
                                                         'upper', 'delta',
                                                         'percent'],
                                     index=self.sens_PD.index)
            sigma[var] = sigma_var

        sigma = pd.concat(sigma, axis=1)

        return sigma

    @property
    def MPECM(self):
        '''Calculate model prediction error covariance matrix

        Returns
        --------
        model_prediction_ECM: pandas DataFrame
            Contains for every timestep the corresponding model prediction
            error covariance matrix.

        '''
        # Parameter correlations should be taken into account, because these
        # lower the model output uncertainty
        if self._MPECM is None:
            PEECM = np.repeat(np.atleast_3d(self.PEECM).T,
                              len(self.sens_PD.index), axis=0)

            # PEECM = np.multiply(PEECM, np.eye(self._par_len))
            # TODO Check if results are ok when var are not measured at the
            # same time
            self._MPECM = self._dotproduct(np.rollaxis(self.sensmatrix, 2, 1),
                                           PEECM)

        return self._MPECM

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
        raise Exception('Needs review')

        if len(self.variables) == 1:
            corr = pd.DataFrame(np.ones([1, 1]),
                                columns=self.variables,
                                index=self.variables)
        else:
            comb_gen = list(combinations(self.variables, 2))
            for i, comb in enumerate(comb_gen):
                if i is 0:
                    combin = [comb[0] + '-' + comb[1]]
                else:
                    combin.append([comb[0] + '-' + comb[1]])

#            if self._data.get_measured_xdata()[0] == 0:
#                time_uncertainty = self._data.get_measured_xdata()[1:]
#            else:
#                time_uncertainty = self._data.get_measured_xdata()

            corr = np.zeros([self._data_len, len(combin)])
            for h, timestep in enumerate(time_uncertainty):
                tracker = 0
                for i, var1 in enumerate(self.get_all_outputs()[:-1]):
                    for j, var2 in enumerate(self.get_all_outputs()[i+1:]):
                        corr[h, tracker] = (self.MPECM[h, i, j+1] /
                                            np.sqrt(self.MPECM[h, i, i] *
                                            self.MPECM[h, j+1, j+1]))
                        tracker += 1

            corr = pd.DataFrame(corr, columns=combin,
                                index=self._data.get_measured_xdata())
            return corr


class CalibratedConfidence(BaseConfidence):
    """
    """
    def __init__(self, calibrated, sens_method='CAS'):
        """
        """
        super(CalibratedConfidence, self).__init__(DirectLocalSensitivity(calibrated.model,
                                                                          calibrated.dof),
                                                   sens_method=sens_method)
        self.uncertainty = calibrated.measurements._uncertainty
        self._measurements = calibrated.measurements
        self.data = calibrated.measurements.Data

    @property
    def uncertainty_PD(self):
        """
        """
        return self._measurements._Error_Covariance_Matrix_PD


class TheoreticalConfidence(BaseConfidence):
    """
    """

    def __init__(self, sens, uncertainty, sens_method='CAS'):
        """
        """
        super(TheoreticalConfidence, self).__init__(sens,
                                                    sens_method=sens_method)
        self.uncertainty = uncertainty

    @classmethod
    def from_calibrated(cls, calibrated_confidence):
        temp = cls(calibrated_confidence.sens,
                   calibrated_confidence.uncertainty,
                   sens_method=calibrated_confidence.sens_method)
        return temp

    @property
    def uncertainty_PD(self):
        """
        """
        return self.uncertainty.get_uncertainty(self.model_output)


