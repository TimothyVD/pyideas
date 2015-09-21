# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:03:10 2015

@author: timothy
"""
import numpy as np
import pandas as pd

from biointense.optimisation import _BaseOptimisation


def A_criterion(FIM):
    '''OED design A criterion
    With this criterion, the trace of the inverse of the FIM is minimized,
    which is equivalent to minimizing the sum of the variances of the
    parameter estimates. In other words, this criterion minimizes the
    arithmetic average of the variances of the parameter estimate.
    Because this criterion is based on an inversion of the FIM,
    numerical problems will arise when the FIM is close to singular.
    '''
    return np.linalg.inv(FIM).trace()


def modA_criterion(FIM):
    '''OED design modified A criterion
    With this criterion, the trace of the inverse of the FIM is minimized,
    which is equivalent to minimizing the sum of the variances of the
    parameter estimates. In other words, this criterion minimizes the
    arithmetic average of the variances of the parameter estimate.
    Because this criterion is based on an inversion of the FIM,
    numerical problems will arise when the FIM is close to singular.
    '''
    return FIM.trace(axis1=-2, axis2=-1)


def D_criterion(FIM):
    '''OED design D criterion
    Here, the idea is to maximize the determinant of the FIM
    (Box and Lucas, 1959). The latter is inversely proportional to the
    volume of the confidence region of the parameter es- timates, and this
    volume is thus minimized when maximizing det (FIM). In other words,
    one minimizes the geometric average of the variances of the parameter
    estimates. More- over, D-optimal experiments possess the property of
    being invariant with respect to any rescaling of the parameters
    (Petersen, 2000; Seber and Wild, 1989). According to Walter and
    Pronzato (1997), the D-optimal design criterion is the most used
    criterion. However, several authors have pointed out that this
    criterion tends to give excessive importance to the parameter which
    is most influential.
    '''
    return np.linalg.det(FIM)


def E_criterion(FIM):
    '''OED design E criterion
    The E-optimal design criterion maximizes the smallest eigenvalue of
    the FIM and thereby minimizes the length of the largest axis of the
    confidence ellipsoid. Thus, these designs aim at minimizing the
    largest parameter estimation variance and thereby at maximizing the
    distance from the singular, unidentifiable case.
    '''
    return np.min(np.linalg.eigvals(FIM), axis=-1)


def modE_criterion(FIM):
    '''OED design modE criterion
    With this criterion, the focus is on the minimization of the condition
    number, which is the ratio between the largest and the smallest
    eigenvalue, or, in other words, the ratio of the shortest and the
    longest ellipsoid axes. The minimum of this ratio is one, which corresponds
    to the case where the shape of the confidence ellipsoid
    is a (hyper)sphere.
    '''
    w = np.linalg.eigvals(FIM)
    return np.max(w, axis=-1)/np.min(w, axis=-1)

OED_CRITERIA = {'A': A_criterion, 'modA': modA_criterion, 'D': D_criterion,
                'E': E_criterion, 'modE': modE_criterion}

OED_CRITERIA_MAXIMIZE = {'A': False, 'modA': True, 'D': True,
                         'E': True, 'modE': False}


class BaseOED(_BaseOptimisation):
    r"""
    """

    def __init__(self, confidence, dof_list, preFIM=None):
        super(BaseOED, self).__init__(confidence.model)
        self.confidence = confidence
        self.dof = dof_list

        # Take into account information of experiments which are already
        # performed
        self.preFIM = preFIM

        self._criterion = 'D'

    def _run_confidence(self, dof_array=None):
        """
        ATTENTION: Zero-point also added, need to be excluded for optimization
        """
        # run option
        if dof_array is not None:
            # Set new parameters values
            dof_dict = self._dof_array_to_dict(dof_array)
            self._dof_dict_to_model(dof_dict)

        FIM = self.confidence.FIM
        if self.preFIM:
            FIM += self.preFIM

        return FIM

    def _obj_fun(self, obj_crit, dof_array=None):
        """
        """
        # Run model & get confidence
        FIM = self._run_confidence(dof_array=dof_array)

        return OED_CRITERIA[obj_crit](FIM)

    def inspyred_optimize(self, criterion='D', prng=None, approach='PSO',
                          initial_parset=None, pop_size=16, max_eval=256,
                          **kwargs):
        """
        """
        self._criterion = criterion

        def inner_obj_fun(dof_array=None):
            return self._obj_fun(obj_crit, dof_array=dof_array)

        final_pop, ea = self._bioinspyred_optimize(
                               inner_obj_fun,
                               prng=prng,
                               approach=approach,
                               initial_parset=initial_parset,
                               pop_size=pop_size,
                               maximize=OED_CRITERIA_MAXIMIZE[criterion],
                               max_eval=max_eval, **kwargs)

        return final_pop, ea

    def select_optimal_individual(self, final_pop):
        """
        """
        if type(final_pop) != list:
            raise Exception('final_pop has to be a list!')

        if OED_CRITERIA_MAXIMIZE[self._criterion]:
            print('Individual with maximum fitness is selected!')
            return max(final_pop)
        else:
            print('Individual with minimum fitness is selected!')
            return min(final_pop)

    def brute_oed(self, step_dict, criterion='D', replacement=True):
        """
        """
        self._criterion = criterion

        independent_dict = {}
        for independent in step_dict.keys():
            independent_dict[independent] = \
                np.linspace(self._dof_distributions[independent].min,
                            self._dof_distributions[independent].max,
                            step_dict[independent])

        self.model.set_independent(independent_dict, method='cartesian')

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)

        if OED_CRITERIA_MAXIMIZE[criterion]:
            selection_criterion = np.argmax
        else:
            selection_criterion = np.argmin

        FIM_evolution = self.confidence.FIM_time

        FIM_tot = 0
        experiments = []
        for i in range(self._independent_samples):
            OED_criterion = OED_CRITERIA[criterion](FIM_evolution)

            optim_indep = selection_criterion(OED_criterion)
            experiments.append(index[optim_indep])

            FIM_tot += FIM_evolution[optim_indep, :, :]
            FIM_evolution = FIM_evolution + FIM_evolution[optim_indep, :, :]

            if not replacement:
                FIM_evolution[optim_indep, :, :] = 1e-20

        return (pd.DataFrame(experiments, columns=self.model.independent),
                FIM_tot)


class RobustOED(object):
    r"""
    The aim of Robust Optimal Experimental Design (robust OED) is to reduce
    the dependence of the (local) design on the chosen parameter values.
    For nonlinear models, the parameter values can have a major effect on
    the the OED.

    Parameters
    ----------
    confidence : biointense.confidence
        A biointense.confidene object contains information about both the
        model and the model uncertainty.
    independent_samples : int
        The number of samples for which an experimental design need to
        be set up.
    preFIM : numpy.ndarray
        Array containing information already available, if preFIM is None
        no information is taken into account to design experiments.

    Examples
    ---------
    >>> from biointense.model import AlgebraicModel
    >>> from biointense.sensitivity import (NumericalLocalSensitivity,
                                        DirectLocalSensitivity)
    >>> from biointense.confidence import TheoreticalConfidence
    >>> from biointense.uncertainty import Uncertainty
    >>> from biointense.oed import BaseOED, RobustOED
    >>> from biointense.parameterdistribution import ModPar

    >>> import numpy as np
    >>> import pandas as pd

    >>> # ACE = PP
    >>> # MPPA = PQ

    >>> system = {'v': ('Vr*ACE*MPPA/(Kal*ACE + Kac*MPPA + ACE*MPPA'
                        '+ Kal/Kacs*ACE**2 + Kac/Kas*MPPA**2)')}
    >>> parameters = {'Vr': 5.18e-4, 'Kal': 1.07, 'Kac': 0.54, 'Kacs': 1.24,
                  'Kas': 25.82}

    >>> M1 = AlgebraicModel('biointense_backward', system, parameters)

    >>> ACE = np.linspace(11., 100., 100)
    >>> MPPA = np.linspace(1., 10., 100)
    >>> M1.set_independent({'ACE': ACE, 'MPPA': MPPA}, method='cartesian')

    >>> M1.initialize_model()

    >>> M1sens = DirectLocalSensitivity(M1)

    >>> M1uncertainty = Uncertainty({'v': '1**2'})

    >>> M1conf = TheoreticalConfidence(M1sens, M1uncertainty)

    >>> M1bruteoed = BaseOED(M1conf, ['ACE', 'MPPA'])
    >>> M1bruteoed._independent_samples = 10
    >>> M1bruteoed.set_dof_distributions(
                        [ModPar('ACE', 1., 100.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])
    >>> indep_out, FIM_end = M1bruteoed.brute_oed({'ACE': 20, 'MPPA': 20},
                                              replacement=False, criterion='D')
    >>> plt.plot(indep_out['ACE'], indep_out['MPPA'], 'o')

    >>> M1oed = RobustOED(M1conf, 5)

    >>> M1oed.set_parameter_distributions(
                        [ModPar('Vr', 1e-7, 1e-1, 'randomUniform'),
                         ModPar('Kal', 0.01, 10, 'randomUniform'),
                         ModPar('Kac', 0.01, 10, 'randomUniform'),
                         ModPar('Kacs', 0.05, 10, 'randomUniform'),
                         ModPar('Kas', 5.0, 50., 'randomUniform')])

    >>> M1oed.set_independent_distributions(
                        [ModPar('ACE', 1., 100.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])

    >>> opt_independent, par_sets = M1oed.maximin(K_max=5)
    """
    def __init__(self, confidence, independent_samples, preFIM=None):
        """
        """
        self.confidence = confidence
        self.model = confidence.model
        self.independent_samples = independent_samples
        self._preFIM = preFIM

        self._dof = {'par': {'dof_len': None,
                             'dof_ordered': None,
                             'dof': None},
                     'ind': {'dof_len': None,
                             'dof_ordered': None,
                             'dof': None}}

        self._oed = {'par': None,
                     'ind': None}

        self._criterion = 'D'

        self.independent_dist = None

    def _set_dof_distributions(self, oed_type, modpar_list, samples):
        """

        """
        names = [dof.name for dof in modpar_list]
        self._oed[oed_type] = BaseOED(self.confidence, names,
                                      preFIM=self._preFIM)
        self._oed[oed_type]._independent_samples = samples
        self._oed[oed_type].set_dof_distributions(modpar_list)
        self._dof[oed_type]['dof'] = self._oed[oed_type].dof
        self._dof[oed_type]['dof_len'] = self._oed[oed_type]._dof_len
        self._dof[oed_type]['dof_ordered'] = self._oed[oed_type]._dof_ordered

    def set_parameter_distributions(self, modpar_list):
        """
        """
        self._set_dof_distributions('par', modpar_list, 0)

    def set_independent_distributions(self, modpar_list):
        """
        """
        self._set_dof_distributions('ind', modpar_list,
                                    self.independent_samples)

    def _outer_obj_fun_calc(self, independent_sample, parameter_sets):
        """
        """
        self._oed['ind']._dof_array_to_model(independent_sample)
        FIM_inner = []
        for parameter_sample in parameter_sets:
            FIM_inner.append(self._inner_obj_fun(parameter_sample, 'ind',
                                                 dist_checker=True))

        return FIM_inner

    def _outer_obj_fun(self, independent_sample, parameter_sets):
        """
        """
        FIM_inner = self._outer_obj_fun_calc(independent_sample,
                                             parameter_sets)

        return np.min(FIM_inner)

    def _inner_obj_fun(self, parameter_sample, oed_type, dist_checker=False):
        """
        """
        par_dict = self._oed[oed_type]._dof_array_to_dict_generic(
                            self._dof['par']['dof_len'],
                            self._dof['par']['dof_ordered'],
                            np.array(parameter_sample))

        self._oed[oed_type]._dof_dict_to_model(par_dict)

        dist_check = 1.
        if dist_checker and self.independent_dist is not None:
            dist_check = self._distance_checker(par_dict['independent'])

        return dist_check*OED_CRITERIA['D'](self._oed[oed_type].confidence.FIM)

    def _distance_checker(self, ind_dict):
        """
        """
        array_size = [self.independent_samples, self.independent_samples]
        ones_array = np.ones(array_size)
        dist_tracker = np.zeros(array_size)

        for ind, val in ind_dict.items():
            ind_array = ones_array * val
            ind_array = np.abs(ind_array - ind_array.T)
            np.fill_diagonal(ind_array, self.independent_dist[ind])

            dist_tracker += ind_array >= self.independent_dist[ind]

        if 0. in dist_tracker:
            dist_check = 0.
        else:
            dist_check = 1.
        return dist_check

    def my_constraint_function(self, candidate):
        """Return the number of constraints that candidate violates."""
        # In this case, we'll just say that the point has to lie
        # within a circle centered at (0, 0) of radius 1.
        if self._oed['ind']._obj_fun('D', candidate) < self.psi_independent:
            return 1
        else:
            return 0

    def _optimize_for_independent(self, parameter_sets, **kwargs):
        """
        """
        def temp_obj_fun(parray=None):
            return self._outer_obj_fun(parray, parameter_sets)

        final_pop, ea = self._oed['ind']._inspyred_optimize(
                                obj_fun=temp_obj_fun,
                                prng=None,
                                approach='PSO',
                                initial_parset=None,
                                pop_size=16,
                                maximize=True,
                                max_eval=1000,
                                **kwargs)

        best_individual = max(final_pop)

        return best_individual.candidate, best_individual.fitness

    def _optimize_for_parameters(self, **kwargs):
        """
        """
        def temp_obj_fun(parray=None):
            return self._inner_obj_fun(parray, 'par', dist_checker=False)

        final_pop, ea = self._oed['par']._inspyred_optimize(
                             obj_fun=temp_obj_fun,
                             prng=None,
                             approach='PSO',
                             initial_parset=None,
                             pop_size=16,
                             maximize=False,  # only valid for D
                             max_eval=1000,
                             **kwargs)

        worst_individual = min(final_pop)

        return worst_individual.candidate, worst_individual.fitness

    def maximin(self, approach='PSO', K_max=100):
        """
        Parameters
        -----------
        approach : str
            Which optimization approach should be followed. PSO|DEA|SA
        K_max : int
            Maximum number of internal loops

        Returns
        -------
        independent_sample : numpy.ndarray
            Contains robust set of independent samples
        parameter_sets : list
            Contains set of arrays with initial parameter set and the worst
            performing parameter samples.

        References
        -----------
        S.P. Asprey, S. Macchietto, Designing robust optimal dynamic
        experiments, Journal of Process Control, Volume 12, Issue 4, June 2002,
        Pages 545-556, ISSN 0959-1524,
        http://dx.doi.org/10.1016/S0959-1524(01)00020-8.
        (http://www.sciencedirect.com/science/article/pii/S0959152401000208)
        """

        parameter_sets = [self._oed['par']._dof_dict_to_array(
                                self.model.parameters.copy())]

        self.psi_parameter = [0]
        self.psi_independent = [1]
        self.maximin_success = False
        K = 0

        while self.psi_parameter[-1] < self.psi_independent[-1] and K <= K_max:
            # Try to optimize independents to maximize D criterion
            independent_sample, psi_independent = \
                self._optimize_for_independent(parameter_sets)
            self.psi_independent.append(psi_independent)

            # Convert output of independent to dof_dict output
            ind_dict = self._oed['ind']._dof_array_to_dict(independent_sample)

            # Adapt all dofs which were optimised in independent
            self._oed['par']._dof_dict_to_model(ind_dict)

            # Try to find worst performing parameter set (min D criterion)
            parameter_sample, psi_parameter = \
                self._optimize_for_parameters()
            self.psi_parameter.append(psi_parameter)

            # If parameter sample is not yet in parameter samples: append it.
            if not (np.any([(parameter_sample == x).all() for x in
                    parameter_sets])):
                parameter_sets.append(parameter_sample)

            # Increase K
            K += 1

            if self.psi_independent[-1] == 0.:
                self.psi_independent[-1] = 1000

        if self.psi_parameter[-1] >= self.psi_independent[-1]:
            self.maximin_success = True

        return independent_sample, parameter_sets
