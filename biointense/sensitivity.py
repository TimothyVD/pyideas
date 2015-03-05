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
import biointense.modeldefinition as moddef


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
            return dict((par, self._parameter_values[par]) for par in
                        self.parameters if par in self._parameter_values)

    def _rescale_sensitivity(self, sensitivity_PD, scaling):
        """
        """
        variables = list(sensitivity_PD.columns.levels[0])
        parameters = list(sensitivity_PD.columns.levels[1])
        perturb_par = pd.Series(self._parameter_values)[self.parameters]
        sensitivity_len = len(sensitivity_PD.index)
        parameter_len = len(self.parameters)

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
        self._parameter_values = {}.fromkeys(model.parameters.keys())
        if parameters is None:
            self.set_perturbation(model.parameters.keys(),
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

    def _get_sensitivity(self, parameter, perturbation):
        """
        """
        output_std = self.model.run()
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
        num_sens = {}
        for par in self.parameters:
            num_sens[par] = self._get_sensitivity(par,
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
    def __init__(self, model):
        """
        """
        if not issubclass(model.__class__, _BiointenseModel):
            raise Exception("DirectLocalSensitivity can only be used for "
                            "(subclasses of the) _BiointenseModel class")
        self.model = model

        self._dfdtheta = None
        self._dfdx = None
        self._dxdtheta = None

        self._dgdx = None
        self._dgdtheta = None

        self._algebraic_swap = None

    def _generate_ode_sens(self):
        '''Analytic derivation of the local sensitivities of ODEs

        Sympy based implementation to get the analytic derivation of the
        ODE sensitivities. Algebraic variables in the ODE equations are replaced
        by its equations to perform the analytical derivation.
        '''

        # Set up symbolic matrix of system states
        system_matrix = sympy.Matrix(sympy.sympify(
            self.model.systemfunctions['ode'].values(), _clash))
        # Set up symbolic matrix of variables
        states_matrix = sympy.Matrix(sympy.sympify(
            self.model.systemfunctions['ode'].keys(), _clash))
        # Set up symbolic matrix of parameters
        parameter_matrix = sympy.Matrix(sympy.sympify(
            self.model.parameters.keys(), _clash))

        # Replace algebraic stuff in system_matrix to perform LSA
        if bool(self.model.systemfunctions['algebraic']):
            # Set up symbolic matrix of algebraic
            algebraic_matrix = sympy.Matrix(sympy.sympify(
                self.model.systemfunctions['algebraic'].keys(), _clash))
            # Replace algebraic variables by its equations
            h = 0
            while (np.sum(np.abs(system_matrix.jacobian(algebraic_matrix))) != 0) and \
                (h <= len(self.model.systemfunctions['algebraic'].keys())):
                for i, alg in enumerate(self.model.systemfunctions['algebraic'].keys()):
                    system_matrix = system_matrix.replace(sympy.sympify(alg, _clash), sympy.sympify(self.Algebraic.values()[i], _clash))
                h += 1

        # Initialize and calculate matrices for analytic sensitivity calculation
        # dfdtheta
        dfdtheta = system_matrix.jacobian(parameter_matrix)
        self.dfdtheta = np.array(dfdtheta)
        # dfdx
        dfdx = system_matrix.jacobian(states_matrix)
        self.dfdx = np.array(dfdx)
        # dxdtheta
        dxdtheta = np.zeros([len(states_matrix),len(self.Parameters)])
        self.dxdtheta = np.asmatrix(dxdtheta)

    def _generate_alg_sens(self):
        """
        """
        # Set up symbolic matrix of variables
        if bool(self.model.systemfunctions['ode']):
            states_matrix = sympy.Matrix(sympy.sympify(
                self.model.systemfunctions['ode'].keys(), _clash))
        # Set up symbolic matrix of parameters
        parameter_matrix = sympy.Matrix(sympy.sympify(
            self.model.parameters.keys(), _clash))

        algebraic_matrix = self._alg_swap

        # Initialize and calculate matrices for analytic sensitivity calculation
        # dgdtheta
        dgdtheta = algebraic_matrix.jacobian(parameter_matrix)
        self.dgdtheta = np.array(dgdtheta)
        # dgdx
        if bool(self.model.systemfunctions['ode']):
            dgdx = algebraic_matrix.jacobian(states_matrix)
            self.dgdx = np.array(dgdx)

    @property
    def _alg_swap(self):
        '''Algebraic swapping and replacing function

        This function is a helper function for _alg_LSA, the aim of this function
        is to replace algebraic variables in other algebraic equations by equations
        which are only dependent on time, parameters and ODEs.

        See also
        ---------
        _alg_LSA
        '''

        if self._algebraic_swap is None:
            h = 0
            algebraic_matrix = sympy.Matrix(sympy.sympify(
                self.model.systemfunctions['algebraic'].values(), _clash))
            algebraic_keys = sympy.Matrix(sympy.sympify(
                self.model.systemfunctions['algebraic'].keys(), _clash))
            while (np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys))) != 0) and (h <= len(self.model.systemfunctions['algebraic'].keys())):
                for i, alg in enumerate(self.Algebraic.keys()):
                    algebraic_matrix = algebraic_matrix.replace(sympy.sympify(alg, _clash),
                                                                sympy.sympify(self.model.systemfunctions['algebraic'].values()[i], _clash))
                h += 1

            self._algebraic_swap = algebraic_matrix

        return self._algebraic_swap

    def _generate_sensitivity(self):
        """
        """
        has_ode = bool(self.model.systemfunctions['ode'])
        has_alg = bool(self.model.systemfunctions['algebraic'])

        if has_ode:
            self._generate_ode_sens()
        if has_alg:
            self._generate_alg_sens()

    def _generate_ode_derivative_definition(self, model):
        '''Write derivative of model as definition in file

        Writes a file with a derivative definition to run the model and
        use it for other applications

        Parameters
        -----------
        model : biointense.model

        '''
        modelstr = 'def fun_ode_lsa(odes, t, parameters, *args, **kwargs):\n'
        # Get the parameter values
        modelstr = moddef.write_parameters(modelstr, model.parameters)
        modelstr = moddef.write_whiteline(modelstr)
        # Get the current variable values from the solver
        modelstr = moddef.write_ode_indices(modelstr, model._ordered_var['ode'])
        modelstr = moddef.write_whiteline(modelstr)
        # Write down necessary algebraic equations (if none, nothing written)
        modelstr = moddef.write_algebraic_lines(modelstr, model.systemfunctions['algebraic'])
        modelstr = moddef.write_whiteline(modelstr)

        # Write down external called functions - not yet provided!
        #write_external_call(defstr, varname, fname, argnames)
        #write_whiteline(modelstr)

        # Write down the current derivative values
        modelstr = moddef.write_ode_lines(modelstr, model.systemfunctions['ode'])
        modelstr = moddef.write_derivative_return(modelstr, model._ordered_var['ode'])

        modelstr += '\n    #Sensitivities\n\n'

        # Calculate number of states by using inputs
        modelstr += '    state_len = len(odes)/(len(parameters)+1)\n'
        # Reshape ODES input to array with right dimensions in order to perform matrix multiplication
        modelstr += '    dxdtheta = array(odes[state_len:].reshape(state_len,len(parameters)))\n\n'

        # Write dfdtheta as symbolic array
        modelstr += '    dfdtheta = '
        modelstr += pprint.pformat(self._dfdtheta)
        # Write dfdx as symbolic array
        modelstr += '\n    dfdx = '
        modelstr += pprint.pformat(self._dfdx)
        # Calculate derivative in order to integrate this
        modelstr += '\n    dxdtheta = dfdtheta + dot(dfdx,dxdtheta)\n'

        modelstr += ('    return '+str(self.model.systemfunctions['ode']).replace("'","") + ""
                           '+ list(dxdtheta.reshape(-1,))\n\n\n')

        return modelstr

    def _generate_non_derivative_part_definition(self, model):
        '''Write derivative of model as definition in file

        Writes a file with a derivative definition to run the model and
        use it for other applications

        Parameters
        -----------
        model : biointense.model

        '''
        modelstr = 'def fun_alg_lsa(independent, parameters, *args, **kwargs):\n'
        # Get independent
        modelstr = write_independent(modelstr, model.independent)
        modelstr = write_whiteline(modelstr)
        # Get the parameter values
        modelstr = write_parameters(modelstr, model.parameters)
        modelstr = write_whiteline(modelstr)

        # Put the variables in a separate array
        if len(model._ordered_var.get('ode', [])):
            modelstr = write_array_extraction(modelstr, model._ordered_var['ode'])
            modelstr = write_whiteline(modelstr)

        # Write down external called functions - not yet provided!
        #write_external_call(defstr, varname, fname, argnames)
        #write_whiteline(modelstr)

        # Write down the equation of algebraic
        modelstr = write_algebraic_solve(modelstr,
                                         model.systemfunctions['algebraic'],
                                         model.independent[0])
        modelstr = write_whiteline(modelstr)

        # TODO!
        algebraic_sens += '\n    #Sensitivities\n\n'

        # Write dgdtheta as symbolic array
        algebraic_sens += '    dgdtheta = np.zeros([len('+self._x_var+'), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.Parameters.keys())) + '])\n'
        for i, alg in enumerate(self.Algebraic.keys()):
            for j, par in enumerate(self.Parameters.keys()):
                algebraic_sens += '    dgdtheta[:,' + str(i) + ',' + str(j) +'] = ' + str(self.dgdtheta[i,j])+'\n'

        # Write dgdx as symbolic array
        algebraic_sens += '    dgdx = np.zeros([len('+self._x_var+'), ' + str(len(self.Algebraic.keys())) + ', ' + str(len(self.System.keys())) + '])\n'
        for i, alg in enumerate(self.Algebraic.keys()):
            for j, par in enumerate(self.System.keys()):
                algebraic_sens += '    dgdx[:,' + str(i) + ',' + str(j) +'] = ' + str(self.dgdx[i,j])+'\n'

        # The two time-dependent 2D matrices should be multiplied with each other (dot product).
        # In order to yield a time-dependent 2D matrix, this is possible using the einsum function.
        algebraic_sens += "\n    dgdxdxdtheta = np.einsum('ijk,ikl->ijl',dgdx,dxdtheta)\n"

        algebraic_sens += '\n    dydtheta = dgdtheta + dgdxdxdtheta\n'

    def get_sensitivity(self, method='CAS'):
        """
        """
        num_sens = {}
        for par in self.parameters:
            num_sens[par] = self._get_sensitivity(par,
                                                  self._parameter_values[par])

        num_sens = pd.concat(num_sens, axis=1)
        num_sens = num_sens.reorder_levels([1, 0], axis=1).sort_index(axis=1)

        num_sens = self._rescale_sensitivity(num_sens, method)

        return num_sens



class GlobalSensitivity(Sensitivity):
    """
    """
    def __init__(self, model):
        """
        """
