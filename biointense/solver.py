# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 11:52:12 2015

@author: timothy
"""
from __future__ import division

from scipy.integrate import odeint, ode
from itertools import product

# Check whether odespy is installed
try:
    _ODESPY = True
    import odespy
except ImportError:
    _ODESPY = False

import numpy as np

# Define availabe integrators for each of all ode' approaches
ODE_INTEGRATORS = {}
ODE_INTEGRATORS['ode'] = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
if _ODESPY:
    ODE_INTEGRATORS['odespy'] = odespy.list_available_solvers()

# Standard setting for each ode approach
STD_ODE_INTEGRATOR = {'ode': 'lsoda', 'odespy': 'lsoda_scipy', 'odeint': ''}


class _Solver(object):
    r"""
    Base solver class for OdeSolver, AlgebraicSolver and HybridSolver
    """

    def __init__(self, fun, independent):
        """
        """
        self.fun = fun

    def _check_solver_sanity(self):
        """
        check whether solver setting is compatible with system
        check the external event: can the solver cope with the time step of
        the measurement?
        """

    def solve(self):
        """
        Solve model equations

        return: For each independent value, returns state for all variables
        """
        return NotImplementedError


class OdeSolver(_Solver):
    r"""

    """
    def __init__(self, fun_ode, initial_conditions, independent, args,
                 ode_solver_options=None, ode_integrator=None):
        """
        """
        self.fun_ode = fun_ode
        self.initial_conditions = initial_conditions
        self.independent = independent.values()[0]
        self.args = args
        self.ode_solver_options = ode_solver_options or {}
        self.ode_integrator = ode_integrator
        self._ode_procedure = {'odeint': self._solve_odeint,
                               'ode': self._solve_ode,
                               'odespy': self._solve_odespy}

    def _check_ode_integrator_setting(self, procedure):
        r"""
        Check if ode procedure is available

        Parameters
        -----------
        procedure: string
            For each of the odesolvers (*odeint*, *ode* and *odespy*), the
            available procedures are saved in *ODE_INTEGRATORS*.
        """
        if self.ode_integrator is None:
            self.ode_integrator = STD_ODE_INTEGRATOR[procedure]
        else:
            if procedure is not 'odeint' and \
               self.ode_integrator not in ODE_INTEGRATORS[procedure]:
                raise Exception(self.ode_integrator + ' is not available, '
                                'please choose one from the ODE_INTEGRATORS '
                                'list.')

    def _solve_odeint(self):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The scipy integrate odeint module can be found on [1]_.

        References
        -----------
        .. [1] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
        scipy.integrate.odeint.html
        """

        res = odeint(self.fun_ode,
                     self.initial_conditions,
                     self.independent,
                     args=self.args, **self.ode_solver_options)

        return res

    def _solve_ode(self):
        r"""
        Calculate the ode equations using scipy integrate ode solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The scipy integrate ode module can be found on [1]_.

        References
        -----------
        .. [1] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
        scipy.integrate.ode.html
        """
        self._check_ode_integrator_setting("ode")

        # Make wrapper function to
        def wrapper(independent_values, initial_conditions, parameters):
            r"""
            Wrapper function to switch order of input
            """
            return self.fun_ode(initial_conditions, independent_values,
                                parameters)

        solver = ode(wrapper).set_integrator(self.ode_integrator,
                                             **self.ode_solver_options)

        solver.set_initial_value(self.initial_conditions,
                                 self.independent)
        solver.set_f_params(*self.args)

        xdata = self.independent
        timesteps = xdata[1:] - xdata[:-1]
        model_output = []
        xdata = []
        model_output.append(solver.y)
        xdata.append(solver.t)
        for dt in timesteps:
            if solver.successful():
                solver.integrate(solver.t + dt)
                model_output.append(solver.y)
                xdata.append(solver.t)

        return np.array(model_output)

    def _solve_odespy(self):
        """
        Calculate the ode equations using scipy integrate ode solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The odespy package can be found on [1]_.

        References
        -----------
        .. [1] H. P. Langtangen and L. Wang. Odespy software package.
        URL: https://github.com/hplgit/odespy. 2014
        """
        if not _ODESPY:
            raise Exception('Odespy is not installed!')

        self._check_ode_integrator_setting("ode")

        solver = odespy.__getattribute__(self.ode_integrator)
        solver = solver(self.fun_ode)
        if self.ode_solver_options is not None:
            solver.set(**self.ode_solver_options)
        solver.set_initial_condition(self.initial_conditions)
        solver.set(f_args=self.args)

        xdata = self.independent
        model_output, xdata = solver.solve(xdata)

        return model_output

    def solve(self, procedure='odeint', **kwargs):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the ode equations in function of the
            independent values
        """
        self._check_ode_integrator_setting(procedure)

        self.ode_solver_options.update(kwargs)

        output = self._ode_procedure[procedure]()

        return output


class AlgebraicSolver(_Solver):
    """
    Class to calculate the algebraic equations/models
    """
    def __init__(self, fun_alg, independent, args, **kwargs):
        self.fun_alg = fun_alg
        self.independent = independent
        self.args = args
        self.kwargs = kwargs

    def _solve_algebraic(self):
        """
        """
        model_output = self.fun_alg(self.independent,
                                    *self.args, **self.kwargs)

        return model_output

    def solve(self):
        """
        Calculate the algebraic equations in function of the independent values

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the algebraic equation in function of the
            independent values
        """
        return self._solve_algebraic()


class HybridSolver(_Solver):
    """
    Class for solving hybrid system of odes and algebraic equations by using.
    """

    def __init__(self, fun_ode, fun_alg, initial_cond, independent_values,
                 args, **kwargs):

        self.fun_ode = fun_ode
        self.fun_alg = fun_alg
        self.initial_conditions = initial_cond
        self.independent = independent_values
        self.args = args

    def solve(self, procedure="odeint", **kwargs):
        """
        Solve hybrid system of odes and algebraic equations. After calculating
        the odes, the algebraic equations are calculated again.

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from both odes and algebraics
        """
        odesolver = OdeSolver(self.fun_ode, self.initial_conditions,
                              self.independent, self.args)
        # Solving ODEs
        odesolver.ode_solver_options.update(kwargs)
        ode_output = odesolver._ode_procedure[procedure]()

        # Solving Algebraic equations
        algsolver = AlgebraicSolver(self.fun_alg, self.independent, self.args,
                                    ode_values=ode_output)
        alg_output = algsolver.solve()

        result = np.concatenate((ode_output, alg_output), axis=1)

        return result
