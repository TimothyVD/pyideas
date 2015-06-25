# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 20:11:00 2015

@author: joris
"""

from __future__ import division
import os
import unittest
import nose

from pandas.util.testing import assert_almost_equal
import biointense

execfile(str(os.path.join(biointense.BASE_DIR, "..", "examples", "LSA_alg_ode.py")))

def test_LSA():

    dir_sens, num_sens = LSA_comparison()

    assert (dir_sens - num_sens).abs().max().max() > 1e-7, ('Direct sens is '
                                                            'not equal to '
                                                            'numerical sens')

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
