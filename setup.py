#!/usr/bin/env python

from setuptools import setup
import os


NAME = 'Biointense'
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


MAJOR = 0
MINOR = 1
MICRO = 1
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION

if not ISRELEASED:
    FULLVERSION += '.dev'


def write_version_py():
    """
    Write the version to biointense/version.py
    """
    cnt = """\
version = '%s'
short_version = '%s'
"""

    filename = os.path.join(os.path.dirname(__file__), 'biointense',
                            'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()


setup(name=NAME,
      version=FULLVERSION,
      description=DESCRIPTION,
      #long_description=long_description,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      #license,
      packages=['biointense'],
      **EXTRA_INFO)
