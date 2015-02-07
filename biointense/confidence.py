# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:59:56 2015

@author: timothy
"""
import numpy as np
from scipy import stats
import pandas as pd


class confidence(object):
    """
    """

    def __init__(self, sens):
        """
        """
        self.sens = sens
        self.model = sens.model

    @staticmethod
    def _sens_PD_2_matrix(sens_PD):
        """
        """
        len_index = len(sens_PD.index)
        len_variables = len(sens_PD.columns.levels[0])
        len_parameters = len(sens_PD.columns.levels[1])

        return np.array(sens_PD).reshape([len_index,
                                          len_parameters,
                                          len_variables])

    def _calc_FIM(self, sens_PD, ECM_PD):
        '''
        Help function for get_FIM

        Parameters
        -----------
        ECM_PD: pandas DataFrame
            The pandas DataFrame needs to contain the error covariance matrix
            for the different measured outputs
        '''
        sensmatrix = self._sens_PD_2_matrix(sens_PD)

        if sens_PD.index[0] == 0.0:
            sens_start = 0
        else:
            sens_start = 1

        # Perform FIM calculation
        # FIM = dy/dx*1/Q*[dy/dx]^T

        # Calculate inverse of ECM_PD
        ECM_inv = np.linalg.inv(np.eye(len(self.get_measured_outputs())) *
                                np.atleast_3d(ECM_PD))
        # Set all very low numbers to zero (just a precaution, so that
        # solutions would be the same as the old get_FIM method). This is
        # probably not necessary!
        ECM_inv[ECM_inv < 1e-20] = 0.
        FIM_timestep = np.einsum('ijk,ikl->ijl',
                                 np.einsum('ijk,ikl->ijl',
                                           sensmatrix[sens_start:, :, :],
                                           ECM_inv),
                                 np.rollaxis(sensmatrix[sens_start:, :, :],
                                             2, 1))
        FIM = np.sum(FIM_timestep, axis=0)
        return FIM

    def get_newFIM(self):
        '''

        '''
        sens_PD = self.sens.get_sensitivity()
        FIM = self._calc_FIM(sens_PD, self.ECM_PD)

        return FIM

    def _get_parameter_confidence(self, alpha, FIM, n_p):
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
        ECM = np.array(np.linalg.inv(FIM))

        CI = np.zeros([ECM.shape[1], 8])

        CI[:, 0] = self.Parameters.values()
        for i, variance in enumerate(np.array(ECM.diagonal())):
            # TODO check whether sum or median or... should be used
            # TODO Check of de absolute waarde hier gebruikt mag worden!!!!
            CI[i, 1:3] = stats.t.interval(alpha, n_p,
                                          loc=self.Parameters.values()[i],
                                          scale=np.sqrt(abs(variance)))
            CI[i, 3] = stats.t.interval(alpha, n_p,
                                        scale=np.sqrt(abs(variance)))[1]
        CI[:, 4] = abs(CI[:, 3]/self.Parameters.values())*100
        CI[:, 5] = self.Parameters.values()/np.sqrt(abs(ECM.diagonal()))
        CI[:, 6] = stats.t.interval(alpha, n_p)[1]
        for i in np.arange(ECM.shape[1]):
            CI[i, 7] = 1 if CI[i, 5] >= CI[i, 6] else 0

        if self._print_on:
            if (CI[:, 7] == 0).any():
                print("Some of the parameters show a non significant t_value, "
                      "which suggests that the confidence intervals of that "
                      "particular parameter include zero and such a situation "
                      "means that the parameter could be statistically "
                      "dropped from the model. However it should be noted "
                      "that sometimes occurs in multi-parameter models "
                      "because of high correlation between the parameters.")
            else:
                print("T_values seem ok, all parameters can be regarded as "
                      "reliable.")

        CI = pd.DataFrame(CI, columns=['value', 'lower', 'upper', 'delta',
                                       'percent', 't_value', 't_reference',
                                       'significant'],
                          index=self.Parameters.keys())

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
        self._check_for_FIM()
        self.ECM = np.linalg.inv(self.FIM)

        n_p = sum(self._data.Data.count())-len(self.Parameters)

        self.parameter_confidence = self._get_parameter_confidence(alpha,
                                                                   self.FIM,
                                                                   n_p)

        return self.parameter_confidence

    def get_parameter_correlation(self):
        '''Calculate correlations between parameters

        Returns
        --------
        R: pandas DataFrame
            Contains for each parameter the correlation with all the other
            parameters.

        '''
        self._check_for_FIM()
        ECM = np.linalg.inv(self.FIM)
        self.ECM = ECM

        self.parameter_correlation = self._get_parameter_correlation(self.FIM)
        return self.parameter_correlation

    def _get_parameter_correlation(self, FIM):
        '''Calculate correlations between parameters

        Returns
        --------
        R: pandas DataFrame
            Contains for each parameter the correlation with all the other
            parameters.

        '''
        ECM = np.linalg.inv(FIM)

        R = np.zeros(ECM.shape)

        for i in range(0, len(ECM)):
            for j in range(0, len(ECM)):
                R[i, j] = ECM[i, j]/(np.sqrt(ECM[i, i]*ECM[j, j]))

        R = pd.DataFrame(R, columns=self.Parameters.keys(),
                         index=self.Parameters.keys())

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
        self._get_model_prediction_ECM()

        time_len = len(self._data.get_measured_xdata())
        par_len = len(self.Parameters)

        sigma = {}
        np.zeros([time_len, 5])

        time_uncertainty = self._data.get_measured_xdata()

        for i, var in enumerate(self.get_measured_outputs()):
            sigma_var = np.zeros([time_len, 5])

            for j, timestep in enumerate(time_uncertainty):
                if self._model.algeb_solved[var].ix[timestep] == 0:
                    sigma_var[j, 0:5] = 0
                else:
                    sigma_var[j, 0] = self._model.algeb_solved[var].ix[timestep]
                    sigma_var[j, 1:3] = stats.t.interval(alpha,
                                                         sum(self._data.Data.count()) - par_len,
                                                         loc=sigma_var[j, 0],
                                                         scale=np.sqrt(self.model_prediction_ECM[j, :, :].diagonal()[i]))
                    sigma_var[j, 3] = abs((sigma_var[j, 2] - sigma_var[j, 0]))
                    sigma_var[j, 4] = abs(sigma_var[j, 3]/sigma_var[j, 0])*100
            sigma_var = pd.DataFrame(sigma_var, columns=['value', 'lower',
                                                         'upper', 'delta',
                                                         'percent'],
                                     index=self._data.get_measured_xdata())
            sigma[var] = sigma_var

        self.model_confidence = sigma
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
        self._get_model_prediction_ECM()

        time_len = len(self._data.get_measured_xdata())

        if len(self.get_all_outputs()) == 1:
            if self._print_on:
                print('Model only has one output!')
            corr = pd.DataFrame(np.ones([1, 1]),
                                columns=self.get_all_outputs()[0],
                                index=self._data.get_measured_xdata())
        else:
            comb_gen = list(combinations(self.get_all_outputs(), 2))
            print(comb_gen)
            for i, comb in enumerate(comb_gen):
                if i is 0:
                    combin = [comb[0] + '-' + comb[1]]
                else:
                    combin.append([comb[0] + '-' + comb[1]])

            if self._data.get_measured_xdata()[0] == 0:
                time_uncertainty = self._data.get_measured_xdata()[1:]
            else:
                time_uncertainty = self._data.get_measured_xdata()

            corr = np.zeros([time_len, len(combin)])
            for h, timestep in enumerate(time_uncertainty):
                tracker = 0
                for i, var1 in enumerate(self.get_all_outputs()[:-1]):
                    for j, var2 in enumerate(self.get_all_outputs()[i+1:]):
                        corr[h, tracker] = (self.model_prediction_ECM[h, i, j+1] /
                                            np.sqrt(self.model_prediction_ECM[h, i, i] *
                                            self.model_prediction_ECM[h, j+1, j+1]))
                        tracker += 1

            corr = pd.DataFrame(corr, columns=combin,
                                index=self._data.get_measured_xdata())
            return corr
