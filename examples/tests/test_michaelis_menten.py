# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 19:02:39 2015

@author: timothy
"""
import os
import numpy as np
from biointense import __path__ as biointense_path

# set working directory on super folder
execfile(str(os.path.join(biointense_path[0], "..", "examples",
                             "michaelis_menten.py")))

def test():
    result1, result2, result3 = michaelis_menten()

    np.testing.assert_array_almost_equal(result1, result2)
    np.testing.assert_array_almost_equal(result1, result3)

if __name__ == "__main__":
    test()
