# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:17:24 2016

@author: timothy
"""
import numpy as np
import pandas as pd
from pyideas import (BaseModel, NumericalLocalSensitivity,
                     ParameterOptimisation, CalibratedConfidence, Measurements,
                     BaseOED, ModPar)
import matlab.engine
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

eng.addpath('/home/timothy/biointense/examples/matlab/', nargout=0)


def MMfun(independent, parameters):
    """
    """
    par4matlab = matlab.double([parameters['Vmax'], parameters['Km']])
    indep4matlab = matlab.double(np.atleast_2d(independent['S']).T.tolist())

    out = eng.MichaelisMenten(par4matlab, indep4matlab)

    return np.array(out)


parameters = {'Km': 150.,     # mM
              'Vmax': 0.768}  # mumol/(min*U)
M1 = BaseModel('MM', parameters, ['v'], ['S'], MMfun)
M1.independent = {'S': np.linspace(0, 500, 1000)}

M1.run().plot()

M1sens = NumericalLocalSensitivity(M1)

M1sens.get_sensitivity(method='PRS').plot()

analysis = M1sens.calc_quality_num_lsa(10**np.arange(-9., -2.))

analysis.plot(loglog=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

data = np.array([[0., 0.],
                 [10., 0.36],
                 [20., 0.53],
                 [30., 0.63],
                 [40., 0.65],
                 [50., 0.67]])

data_pd = pd.DataFrame(data, columns=['S', 'v'])
data_pd = data_pd.set_index('S')

M1data = Measurements(data_pd)
M1data.add_measured_errors({'v': 0.05}, method='Ternbach',
                           lower_accuracy_bound=0.05,
                           minimal_relative_error=1.)

M1optim = ParameterOptimisation(M1, M1data)
M1optim.local_optimize()

M1optim.modmeas.plot()

M1conf = CalibratedConfidence(M1optim)
M1conf.get_parameter_confidence()
M1conf.get_parameter_correlation()

M1conf.get_model_confidence()

M1OED = BaseOED(M1conf, ['S'])

M1OED.set_dof_distributions([ModPar('S', 5., 300.0, 'randomUniform')])

indep_out, FIM_end = M1OED.brute_oed({'S': 100}, 5,
                                     replacement=False, criterion='D')
