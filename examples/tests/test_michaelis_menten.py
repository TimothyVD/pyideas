# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 19:02:39 2015

@author: timothy
"""
import os
import numpy as np
import pytest

import biointense

try:
    import odespy
    SKIP_ODESPY = False
except ImportError:
    SKIP_ODESPY = True


# run the example file
execfile(str(os.path.join(biointense.BASE_DIR, "..", "examples",
                          "michaelis_menten.py")))


@pytest.mark.skipif(SKIP_ODESPY, reason="odespy not installed")
def test():
    result1, result2, result3 = michaelis_menten()

    np.testing.assert_array_almost_equal(result1, result2)
    np.testing.assert_array_almost_equal(result1, result3)

if __name__ == "__main__":
    test()
