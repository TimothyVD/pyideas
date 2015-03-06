# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 19:02:39 2015

@author: timothy
"""
import os
from numpy.testing import assert_allclose

import biointense

# run the example file
execfile(str(os.path.join(biointense.BASE_DIR, "..", "examples",
                          "modsim_optimisation.py")))

def test():
    optim_old = run_modsim_models_old()
    optim_new = run_modsim_models_new()

    assert_allclose(optim_old, optim_new, rtol=1e-5)

if __name__ == "__main__":
    test()
