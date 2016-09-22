# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:17:24 2016

@author: timothy
"""
import numpy as np
import os
from biointense.modelbase import BaseModel
from biointense import NumericalLocalSensitivity
import matlab.engine
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

#eng.addpath(os.path.dirname(os.path.realpath(__file__)),nargout=0)
eng.addpath('/home/timothy/biointense/examples/matlab/',nargout=0)


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

analysis = M1sens.calc_quality_num_lsa(10**np.arange(-10., 0.))

analysis.plot(loglog=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))