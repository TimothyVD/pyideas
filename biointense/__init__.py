# -*- coding: utf-8 -*-
"""
@author: Timothy Van Daele & Stijn Van Hoey

pySTAN: python STructure Analyst
E-mail: stvhoey.vanhoey@ugent.be
"""

from biointense.version import version as __version__

from ode_generator import DAErunner
from ode_maker import odemaker
from optimalexperimentaldesign import *
from ode_optimization import *

from plotfunctions import *
from measurements import *
from parameterdistribution import *


if __name__ == '__main__':
    print 'Bio-Intense ODE/OED package: python package for model development with Ordinary Differential Equations and Optimal Experimental Design'
