# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:48:31 2015

@author: timothy
"""

from __future__ import division
import os, sys

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_almost_equal

import pyideas

# set working directory on super folder


def test_multidim():
    execfile(str(os.path.join(pyideas.BASE_DIR, "..", "examples",
                              "multidim_model.py")))


def test_second_order():
    sys.path.append(os.path.join(pyideas.BASE_DIR, "..", "examples"))
    import second_order_system
    second_order_system.run_second_order_old()
    second_order_system.run_second_order_new()