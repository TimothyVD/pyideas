#!/usr/bin/env python

from setuptools import setup


NAME = 'Biointense'
VERSION = '0.1'
DESCRIPTION = 'ODE analyser package'
# LONG_DESCRIPTION = descr
AUTHOR = 'Timothy Van Daele, Stijn Van Hoey, Daan Van Hauwermeiren, ' \
         'Joris Van den Bossche'
AUTHOR_EMAIL = 'github@biomath.ugent.be',
URL = 'https://github.ugent.be/biomath/biointense'
# LICENSE =
PACKAGE_NAME = 'biointense'
EXTRA_INFO = dict(
    install_requires=['numpy'],
    # classifiers=[]
)


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      #long_description=long_description,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      #license,
      packages=['biointense'],
      **EXTRA_INFO)
