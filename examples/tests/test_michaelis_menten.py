# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 19:02:39 2015

@author: timothy
"""
import os

import pytest
from numpy.testing import assert_array_almost_equal

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

    assert_array_almost_equal(result1, result2)
    assert_array_almost_equal(result1, result3)


@pytest.mark.skipif(SKIP_ODESPY, reason="odespy not installed")
def test_result_old_new():
    _, result2, _ = michaelis_menten()
    result_old = michaelis_menten_old()

    assert_array_almost_equal(result2.values, result_old.values, decimal=14)


if __name__ == "__main__":
    test()
