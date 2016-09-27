# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 20:11:00 2015

@author: joris
"""

from __future__ import division
import os
import unittest
import nose

from pandas.util.testing import assert_frame_equal
import pyideas

execfile(str(os.path.join(pyideas.BASE_DIR, "..", "examples",
                          "LSA_alg_ode.py")))

def test_LSA():

    dir_sens, num_sens = LSA_comparison()

    # TODO Use assert_frame_equal if possible (fails now!)
    # num_sens = num_sens.reindex(columns=dir_sens.columns)
    assert_frame_equal(dir_sens, num_sens, check_less_precise=True)

    #assert (num_sens - dir_sens).abs().max().max() > 1e-7

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
