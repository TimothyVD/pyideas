# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:03:10 2015

@author: timothy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

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
    Base class for performing Model-based Optimal Experimental Design.

    The idea is to optimise the expected information of your future experiments
    by altering the values of your independent variables.

    Examples
    ---------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pandas as pd
    >>> from biointense import (AlgebraicModel, DirectLocalSensitivity,
            TheoreticalConfidence, Uncertainty, BaseOED, ModPar)
    >>> system = {'v': ('Vr*ACE*MPPA/(Kal*ACE + Kac*MPPA + ACE*MPPA'
                        '+ Kal/Kacs*ACE**2 + Kac/Kas*MPPA**2)')}
    >>> parameters = {'Vr': 5.18e-4, 'Kal': 1.07, 'Kac': 0.54, 'Kacs': 1.24,
                      'Kas': 25.82}
    >>> M1 = AlgebraicModel('biointense_backward', system, parameters)
    >>> ACE = np.linspace(11., 100., 250)
    >>> MPPA = np.linspace(1., 10., 250)
    >>> M1.set_independent({'ACE': ACE, 'MPPA': MPPA}, method='cartesian')
    >>> M1.initialize_model()
    >>> M1sens = DirectLocalSensitivity(M1)
    >>> M1uncertainty = Uncertainty({'v': '1**2'})
    >>> M1conf = TheoreticalConfidence(M1sens, M1uncertainty)
    >>> M1oed = BaseOED(M1conf, ['ACE', 'MPPA'])
    >>> #M1oed._independent_samples = 10
    >>> M1oed.set_dof_distributions(
                        [ModPar('ACE', 1., 100.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])
    >>> indep_out, FIM_end = M1oed.brute_oed({'ACE': 20, 'MPPA': 20}, 20,
                                             replacement=False, criterion='D')
    >>> M1.plot_contourf('ACE', 'MPPA', M1.run())
    >>> plt.hold(True)
    >>> plt.plot(indep_out['ACE'], indep_out['MPPA'], 'o')
    >>> finalpop, ea = M1oed.inspyred_optimize()
    >>> best = M1oed.select_optimal_individual(finalpop)
    >>> M1oed._dof_array_to_dict(best.candidate)
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

        # Wrapper of the generic objective function to provide to
        # inspyred optimisation
        def inner_obj_fun(parray=None):
            return self._obj_fun(criterion, dof_array=parray)

        final_pop, ea =\
            self._inspyred_optimize(inner_obj_fun, prng=prng,
                                    approach=approach,
                                    initial_parset=initial_parset,
                                    pop_size=pop_size,
                                    # Depending on criterion used,
                                    # the objective function has to
                                    # be minimized or maximized
                                    maximize=OED_CRITERIA_MAXIMIZE[criterion],
                                    max_eval=max_eval, **kwargs)

        return final_pop, ea

    def select_optimal_individual(self, final_pop):
        """
        From the final population of inspyred_optimize, the most optimal
        individual needs to be selected. Depending on the OED criterion used,
        the individual with the minimum or maximum objective value is selected
        automatically.

        Parameters:
        ----------
        final_pop: list
            final_pop contains the population returned by inspyred_optimize,
            which (hopefully :-)) contains a more optimal design.

        Returns:
        --------
        individual: instance
            Most optimal individual, containing both the independent values and
            the value of the objective function
        """
        if not isinstance(final_pop, list):
            raise Exception('final_pop has to be a list!')

        if OED_CRITERIA_MAXIMIZE[self._criterion]:
            print('Individual with maximum fitness is selected!')
            return max(final_pop)
        else:
            print('Individual with minimum fitness is selected!')
            return min(final_pop)

    def _prepare_brute_approach(self, step_dict):
        """
        """
        self._set_dof_boundaries()

        # Create dict containing independent values
        independent_dict = {}
        independent_list = []
        parameter_dict = {}
        parameter_list = []
        initial_dict = {}
        initial_list = []
        for changeable in step_dict.keys():
            temp_dist = np.linspace(self._dof_distributions[changeable].min,
                                    self._dof_distributions[changeable].max,
                                    step_dict[changeable])
            if changeable in self._dof_ordered['independent']:
                independent_dict[changeable] = temp_dist
                independent_list.append(changeable)
            elif changeable in self._dof_ordered['initial']:
                initial_dict[changeable] = temp_dist
                initial_list.append(changeable)
            elif changeable in self._dof_ordered['parameter']:
                parameter_dict[changeable] = temp_dist
                parameter_list.append(changeable)
            else:
                raise Exception('Only initial conditions, parameters and '
                                'independent values can be altered in this '
                                'function!')

        # Write independent values to model
        if self.model.modeltype is "Model":
            self.model.set_independent(independent_dict)
        elif self.model.modeltype is "AlgebraicModel":
            self.model.set_independent(independent_dict, method='cartesian')

        initial_cond = list(itertools.product(*initial_dict.values()))
        parameters = list(itertools.product(*parameter_dict.values()))
        temp = initial_dict.values() + parameter_dict.values() +\
            independent_dict.values()
        arrays = np.array(list(itertools.product(*temp)))

        names = initial_list + parameter_list + independent_list
        index = pd.MultiIndex.from_arrays(arrays.T, names=names)

        return (initial_cond, initial_list, parameters, parameter_list, index,
                names)

    def brute_modeloutput(self, step_dict):
        r"""
        Examples
        ---------
        >>> output = M1oed.brute_modeloutput({'S': 100, 't': 100})

        >>> M1oed.plot_contourf('S', 't', output['P'])
        """
        (initial_cond, initial_list, parameters, parameter_list, index, names) = \
            self._prepare_brute_approach(step_dict)

        # For each sample, calculate model
        modeloutput_container = []
        for initial_values in initial_cond:
            initial = dict(zip(initial_list, initial_values))
            self.model.set_initial(initial)
            for parameter_values in parameters:
                par = dict(zip(parameter_list, parameter_values))
                self.model.set_parameters(par)
                modeloutput_container.append(self.model._run())

        output_var = self.model._ordered_var['ode'] +\
            self.model._ordered_var['algebraic']
        modeloutput = np.concatenate(modeloutput_container)

        return pd.DataFrame(modeloutput, index=index, columns=output_var)

    def brute_oed(self, step_dict, number_of_samples, criterion='D',
                  replacement=True):
        r"""
        Brute force way of selecting the most optimal experiments. For each

        Parameters
        ----------
        step_dict: dict
            For each of the independents, the step_dict contains a minimum
            value, a maximum value, and the numbers of steps. It will
            automatically, create a raster and calculate the FIM for each
            experiment.

        number_of_samples: int
            How many experiments do you want to design at once (using your
            preliminary knowledge)? Keep in mind that only designing one
            experiment at a time can be quite timeconsuming, however every
            iteration the new experiment can already be used as additional
            information to design a new experiment.

        criterion: 'A'|'modA'|'D'|'E'|'modE'
            Select OED criterion do you want to use? Standard setting is 'D'.

        replacement: True|False
            Do you want to be able to sample the same experiment more than
            once?

        Returns
        -------
        experiments: pd.DataFrame
            Pandas dataframe containing the conditions and/or times at which
            measurements should be performed.
        FIM_tot: np.array
            Numpy array containing the FIM, which is the sum of the self.preFIM
            and the newly designed experiments.

        Examples
        ---------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import pandas as pd
        >>> from biointense import (Model, DirectLocalSensitivity, Uncertainty,
                                    TheoreticalConfidence, ModPar, BaseOED)
        >>> system = {'v': 'Vmax*S/(Km + S)', 'dP': 'v', 'dS': '-v'}
        >>> parameters = {'Vmax': 0.15, 'Km': 2.5}
        >>> M1 = Model('biointense_backward', system, parameters)
        >>> M1.set_independent({'t': np.linspace(0., 500., 50)})
        >>> M1.set_initial({'S': 40.,'P': 0.})
        >>> M1.initialize_model()
        >>> M1.run()
        >>> M1sens = DirectLocalSensitivity(M1)
        >>> M1uncertainty = Uncertainty({'v': '10**2', 'S': '1**2',
                                         'P': '1**2',})
        >>> M1conf = TheoreticalConfidence(M1sens, M1uncertainty)
        >>> M1conf.get_parameter_confidence()
        >>> M1oed = BaseOED(M1conf, ['S', 't'])
        >>> M1oed.set_dof_distributions([ModPar('S', 1., 20.0,
                                                'randomUniform'),
                                         ModPar('t', 0., 500.0,
                                                'randomUniform')])
        >>> indep_out, FIM_end = M1oed.brute_oed({'S': 20, 't': 20}, 20,
                                                 replacement=False,
                                                 criterion='D')
        >>> output = M1oed.brute_modeloutput({'S': 100, 't': 100})
        >>> M1oed.plot_contourf('S', 't', output['P'])
        >>> plt.hold(True)
        >>> plt.plot(indep_out['S'], indep_out['t'], 'o')

        >>> M1oed._independent_samples = 5
        >>> finalpop, ea = M1oed.inspyred_optimize()
        >>> best = M1oed.select_optimal_individual(finalpop)
        >>> M1oed._dof_array_to_dict(best.candidate)
        """
        self._criterion = criterion
        self._independent_samples = number_of_samples

        (initial_cond, initial_list, parameters, parameter_list, index, names) = \
            self._prepare_brute_approach(step_dict)

        # Select whether objective function needs to be minimised or maximised
        # This depends on the OED criterion used
        if OED_CRITERIA_MAXIMIZE[criterion]:
            selection_criterion = np.argmax
        else:
            selection_criterion = np.argmin

        # For each sample, calculate FIM
        FIM_container = []
        for initial_values in initial_cond:
            initial = dict(zip(initial_list, initial_values))
            self.model.set_initial(initial)
            for parameter_values in parameters:
                par = dict(zip(parameter_list, parameter_values))
                self.model.set_parameters(par)
                FIM_container.append(self.confidence.FIM_time)

        FIM_evolution = np.concatenate(FIM_container)

        # Initialise the FIM
        FIM_tot = 0
        # Container to store most informative experiments
        experiments = []
        for i in range(int(number_of_samples)):
            OED_criterion = OED_CRITERIA[criterion](FIM_evolution)

            # Select most optimal sample
            optim_indep = selection_criterion(OED_criterion)
            # Add most optimal sample to experiment container
            experiments.append(index[optim_indep])

            # Add FIM of that specific sample to the FIM_tot/FIM_evolution,
            # since we want to optimise the information content of all the
            # samples. If we would not keep track of the current FIM, we would
            # always select the same individual
            FIM_tot += FIM_evolution[optim_indep, :, :]
            FIM_evolution = FIM_evolution + FIM_evolution[optim_indep, :, :]

            if not replacement:
                FIM_evolution[optim_indep, :, :] = 1e-20

        return (pd.DataFrame(experiments, columns=names), FIM_tot)

    def plot_contourf(self, independent_x, independent_y, output, ax=None,
                      **kwargs):
        r"""
        Parameters
        -----------
        independent_x: string
            Independent of interest to be shown at the x-axis.
        independent_y: string
            Independent of interest to be shown at the y-axis.
        output: pandas.Dataframe
            algebraic equation to be shown as a contourplot (in function of
            independent_x and independent_y)
        ax: matplotlib.ax
            Pass ax to plot on.

        Examples
        ---------
        >>> M1oed.plot_contourf('S', 't', output['P'])
        """
        x_values = output.index.get_level_values(independent_x)
        y_values = output.index.get_level_values(independent_y)
        shape = [len(x_values.unique()), len(y_values.unique())]
        x = np.reshape(x_values, shape)
        y = np.reshape(y_values, shape)
        z = np.reshape(output.values, shape)

        if ax is None:
            ax = plt.gca()

        cs = ax.contourf(x, y, z, **kwargs)
        ax.set_xlabel(independent_x)
        ax.set_ylabel(independent_y)
        plt.colorbar(cs)

        return ax


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
    >>> import numpy as np
    >>> import pandas as pd
    >>> from biointense import (AlgebraicModel, DirectLocalSensitivity,
            TheoreticalConfidence, Uncertainty,RobustOED, ModPar)
    >>> # ACE = PP
    >>> # MPPA = PQ
    >>> system = {'v': 'Vr*ACE*MPPA/(Kace*ACE + Kmppa*MPPA + ACE*MPPA)'}
    >>> parameters = {'Vr': 1e-2, 'Kace': 10., 'Kmppa': 5.}
    >>> M1 = AlgebraicModel('biointense_backward', system, parameters)
    >>> M1.set_independent({'ACE': np.linspace(11., 250., 100),
                            'MPPA': np.linspace(1., 10., 100)},
                           method='cartesian')
    >>> M1.initialize_model()
    >>> M1sens = DirectLocalSensitivity(M1)
    >>> M1uncertainty = Uncertainty({'v': '(v*0.10)**2'})
    >>> M1conf = TheoreticalConfidence(M1sens, M1uncertainty)
    >>> M1oed = RobustOED(M1conf, 5)
    >>> M1oed.set_parameter_distributions(
                        [ModPar('Vr', 1e-7, 1e-1, 'randomUniform'),
                         ModPar('Kace', 0.01, 50., 'randomUniform'),
                         ModPar('Kmppa', 0.01, 10., 'randomUniform')])
    >>> M1oed.set_independent_distributions(
                        [ModPar('ACE', 1., 250.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])
    >>> opt_independent, par_sets = M1oed.maximin(K_max=5)
    >>> M1oed._oed['ind']._dof_array_to_dict(opt_independent)
    """
    def __init__(self, confidence, independent_samples, preFIM=None):
        """
        """
        self.confidence = confidence
        self.model = confidence.model
        self.independent_samples = independent_samples
        self.preFIM = preFIM

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

        self.psi_independent = None
        self.psi_parameter = None
        self.maximin_success = False

    def _set_dof_distributions(self, oed_type, modpar_list, samples):
        """

        """
        names = [dof.name for dof in modpar_list]
        self._oed[oed_type] = BaseOED(self.confidence, names,
                                      preFIM=self.preFIM)
        self._oed[oed_type]._independent_samples = samples
        self._oed[oed_type].set_dof_distributions(modpar_list)
        self._dof[oed_type]['dof'] = self._oed[oed_type].dof
        self._dof[oed_type]['dof_len'] = self._oed[oed_type]._dof_len
        self._dof[oed_type]['dof_ordered'] = self._oed[oed_type]._dof_ordered

    def set_parameter_distributions(self, modpar_list):
        """
        Set distributions for the variables you CANNOT change BUT would like
        to calibrate/estimate more reliably, e.g. kinetic model parameters.
        This is useful when a model calibration exercise is started but no
        (accurate) parameter values are known only a rough estimation of the
        range. By taking the entire range of possible parameters combinations
        into account, the design will be less dependent on one specific
        parameter set.

        Parameters
        -----------
        modpar_list: list
            List containing ModPar instances, describing the ranges and
            distributions from which should be sampled from.

        Examples
        ---------
        >>> M1oed.set_parameter_distributions(
                        [ModPar('ACE', 1., 100.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])
        """
        self._set_dof_distributions('par', modpar_list, 0)

    def set_independent_distributions(self, modpar_list):
        """
        Set distributions for the variables you CAN change, like the
        measurement times, concentrations, experimental conditions,...

        Parameters
        -----------
        modpar_list: list
            List containing ModPar instances, describing the ranges and
            distributions from which should be sampled from.

        Examples
        ---------
        >>> M1oed.set_independent_distributions(
                        [ModPar('ACE', 1., 100.0, 'randomUniform'),
                         ModPar('MPPA', 1., 10.0, 'randomUniform')])
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
                                                 dist_checker=False))

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
            self._dof['par']['dof_len'], self._dof['par']['dof_ordered'],
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

#    def my_constraint_function(self, candidate):
#        """Return the number of constraints that candidate violates."""
#        # In this case, we'll just say that the point has to lie
#        # within a circle centered at (0, 0) of radius 1.
#        if self._oed['ind']._obj_fun('D', candidate) < self.psi_independent:
#            return 1
#        else:
#            return 0

    def _optimize_for_independent(self, parameter_sets, **kwargs):
        """
        """
        def temp_obj_fun(parray=None):
            return self._outer_obj_fun(parray, parameter_sets)

        kwargs = {'obj_fun': temp_obj_fun, 'prng': None, 'approach': 'PSO',
                  'initial_parset': None, 'pop_size': 16, 'maximize': True,
                  'max_eval': 1000}

        final_pop, ea = self._oed['ind']._inspyred_optimize(**kwargs)

        best_individual = max(final_pop)

        return best_individual.candidate, best_individual.fitness

    def _optimize_for_parameters(self, **kwargs):
        """
        """
        def temp_obj_fun(parray=None):
            return self._inner_obj_fun(parray, 'par', dist_checker=False)

        internal_kwargs = {'obj_fun': temp_obj_fun, 'prng': None,
                           'approach': 'PSO', 'initial_parset': None,
                           'pop_size': 16, 'maximize': False,  # Only for D
                           'max_eval': 1000}

        internal_kwargs.update(kwargs)

        final_pop, ea = self._oed['par']._inspyred_optimize(**internal_kwargs)

        worst_individual = min(final_pop)

        return worst_individual.candidate, worst_individual.fitness

    def maximin(self, approach='PSO', K_max=100):
        r"""
        This algorithm is a implementation of the worst-case approach,
        described in [2]_. The worst-case approach aims to determine experiment
        designs that optimise the worst possible performance for *any* value of
        :math:`\theta \in \Theta`

        .. math:: \phi_R = \arg\ \max_{\phi\in\Phi}\ \min_{\theta\in\Theta} \left\{M_I(\theta, \phi)\right\}

        Pseudocode
        *Given: a nominal vector of parameter values, :math:`\theta \in \Theta`

        *Step 0: set K:=1

        *Step 1: solve :math:`\Psi^{[K]} = \max\ \Psi`
            s.t. :math:`\Psi \leq \det(M_I^\phi(\theta, \phi))|_{\theta^{[k]}}, k=1,\ldots,K`

        *Step 2: solve :math:`\hat{\psi}^{[K]} = \min_{\theta}\ \det(M_I(\theta, \phi))` to obtain :math:`\theta^{[K+1]}`

        *Step 3: if :math:`\hat{\psi}^{[K]}<\psi^{[K]}`, then set K:=K+1 and repeat from Step 1

        *Step 4: stop: R-optimal experiment design is :math:`\phi^{[k]}`

        Einde

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
        [2] S.P. Asprey, S. Macchietto, Designing robust optimal dynamic
        experiments, Journal of Process Control, Volume 12, Issue 4, June 2002,
        Pages 545-556, ISSN 0959-1524,
        http://dx.doi.org/10.1016/S0959-1524(01)00020-8.
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

            # TODO Why do I do this? And why 1000?
            # I think that this was added to avoid that the algoritm would stop
            # at strange values. To be examined
            if self.psi_independent[-1] == 0.:
                self.psi_independent[-1] = 1000

        if self.psi_parameter[-1] >= self.psi_independent[-1]:
            self.maximin_success = True

        return independent_sample, parameter_sets
