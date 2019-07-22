#!/usr/bin/env python
from setuptools import setup, Command
# Make sure correct qt backend is used
import matplotlib
matplotlib.use('qt5agg')


setup(
    name='MMHelper',
    version='0.9',
    description='Mother Machine Analysis module',
    author='Jeremy Metz',
    author_email='j.metz@exeter.ac.uk',
    packages=['mmhelper', 'mmhelper.detection'],
)
