#!/usr/bin/env python
# encoding: utf-8
"""Basic installation script for pyfbf.
"""
__docformat__ = "restructuredtext en"
from setuptools import setup, find_packages

classifiers = ""
version = '0.1'

setup(
    name='pyfbf',
    version=version,
    description="Library for working with flat binary files",
    classifiers=filter(None, classifiers.split("\n")),
    keywords='',
    author='David Hoese, SSEC',
    author_email='david.hoese@ssec.wisc.edu',
    license='GPLv3',
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy'
        ]
)

