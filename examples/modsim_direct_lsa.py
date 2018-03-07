# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:57:52 2015

@author: timothy
"""
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import DirectLocalSensitivity, NumericalLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.uncertainty import Uncertainty

import numpy as np
import pandas as pd


parameters = {'W0': 2.0805,
              'Wf': 9.7523,
              'mu': 0.0659}

system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

M1 = AlgebraicModel('Modsim1', system, parameters, ['t'])

M1.independent = {'t': np.array([0., 20., 29., 41., 50., 65., 72.])}

M1.initialize_model()

M1.run()

M1sens = DirectLocalSensitivity(M1)
M1sens.get_sensitivity()

M1sensnum = NumericalLocalSensitivity(M1)
outnum = M1sensnum.get_sensitivity()

uncertainty = Uncertainty({'W': '1'})
M1conf = TheoreticalConfidence(M1sens, uncertainty)
M1conf.get_parameter_confidence()

M1confnum = TheoreticalConfidence(M1sensnum, uncertainty)
M1confnum.get_parameter_confidence()

