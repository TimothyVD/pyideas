# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:27:24 2015

@author: timothy
"""
from biointense import DAErunner

# new
from biointense.model import Model


def run_second_order_old():
    '''
    y'' + 2*tau*omega*y' + omega**2*y = omega**2*K*u

    x1  = y
    x1' = y' = x2
    x2' = y''= -2*tau*omega*x2 - omega**2*x1 + K*omega**2
    '''
    parameters = {'K': 0.01, 'tau': 0.3, 'omega': 0.6}  # mM

    ode = {'dx1': 'x2',
           'dx2': '-2*tau*omega*x2 - (omega**2)*x1 + K*omega**2'}

    algebraic = {'v': 'x1'}

    M1 = DAErunner(ODE=ode, Algebraic=algebraic, Parameters=parameters,
                   Modelname='second_order', print_on=False, x_var='t')

    M1.set_xdata({'start': 0, 'end': 20, 'nsteps': 10000})  # seconds

    M1.set_measured_states(["v"])
    M1.set_initial_conditions({'x1': 0, 'x2': 0})

    M1.solve_ode(plotit=False)

    return M1


def run_second_order_new():
    '''
    y'' + 2*tau*omega*y' + omega**2*y = omega**2*K*u

    x1  = y
    x1' = y' = x2
    x2' = y''= -2*tau*omega*x2 - omega**2*x1 + K*omega**2
    '''
    parameters = {'K': 0.01, 'tau': 0.3, 'omega': 0.6}  # mM

    system = {'dx1': 'x2',
              'dx2': '-2*tau*omega*x2 - (omega**2)*x1 + K*omega**2'}

    M1 = Model('second_order', system, parameters)

    return M1

if __name__ == "__main__":
    M1 = run_second_order_old()
    M1.ode_solved['x1'].plot()

    M1_new = run_second_order_new()
