#!/usr/bin/env python

"""
Parts of this file were taken from the pyzmq project
(https://github.com/zeromq/pyzmq) which have been permitted for use under the
BSD license. Parts are from lxml (https://github.com/lxml/lxml)
"""

import os
import sys

# try bootstrapping setuptools if it doesn't exist
try:
    import pkg_resources
    try:
        pkg_resources.require("setuptools>=0.6c5")
    except pkg_resources.VersionConflict:
        from ez_setup import use_setuptools
        use_setuptools(version="0.6c5")
    from setuptools import setup, Command
    _have_setuptools = True
except ImportError:
    # no setuptools installed
    from distutils.core import setup, Command
    _have_setuptools = False

setuptools_kwargs = {}
min_numpy_ver = '1.6'
if sys.version_info[0] >= 3:

    if sys.version_info[1] >= 3: # 3.3 needs numpy 1.7+
        min_numpy_ver = "1.7.0b2"

    setuptools_kwargs = {
                         'zip_safe': False,
                         'install_requires': ['python-dateutil >= 2',
                                              'pytz >= 2011k',
                                              'numpy >= %s' % min_numpy_ver],
                         'setup_requires': ['numpy >= %s' % min_numpy_ver],
                         }
    if not _have_setuptools:
        sys.exit("need setuptools/distribute for Py3k"
                 "\n$ pip install distribute")

else:
    min_numpy_ver = '1.6.1'
    setuptools_kwargs = {
        'install_requires': ['python-dateutil',
                            'pytz >= 2011k',
                             'numpy >= %s' % min_numpy_ver],
        'setup_requires': ['numpy >= %s' % min_numpy_ver],
        'zip_safe': False,
    }

    if not _have_setuptools:
        try:
            import numpy
            import dateutil
            setuptools_kwargs = {}
        except ImportError:
            sys.exit("install requires: 'python-dateutil < 2','numpy'."
                     " use pip or easy_install."
                     "\n $ pip install 'python-dateutil < 2' 'numpy'")

setup(name='Biointense',
      version='0.1',
      description='ODE analyser package',
      author='Timothy Van Daele & Stijn Van Hoey',
      author_email='github@biomath.ugent.be',
      url='https://github.ugent.be/biomath/biointense',
      packages=['biointense'],
      platforms='any'
     )

