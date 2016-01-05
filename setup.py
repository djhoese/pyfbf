#!/usr/bin/env python
# encoding: utf-8
"""Basic installation script for pyfbf.

development:
python setup.py develop --install-dir=$HOME/Library/Python/2.7/lib/python/site-packages

distribution:
python setup.py sdist
scp dist/*.tar.gz larch:repos/pyfbf/

use:
python setup.py install --install-dir=$HOME/Library/Python
easy_install -d $HOME/Library/Python -vi http://larch.ssec.wisc.edu/eggs/repos pyfbf

"""
__docformat__ = "restructuredtext en"
import sys
from setuptools import setup, find_packages

classifiers = ""
version = '0.1.1'

tests_require = []
if sys.version_info[0] == 2:
    tests_require.append("mock")

setup(
    name='pyfbf',
    version=version,
    description="Library for working with flat binary files",
    classifiers=filter(None, classifiers.split("\n")),
    keywords='',
    author='Ray Garcia, Maciek Smuga-Otto, David Hoese; UW SSEC',
    author_email='ray.garcia@ssec.wisc.edu',
    license='GPLv3',
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy'
    ],
    test_suite='pyfbf',
    tests_require=tests_require
)

