# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ionchannelABC',
    version='0.3.0',
    description='Approximate Bayesian computation for ion channel models',
    long_description=readme,
    author='Charles Houston',
    author_email='charles.houston@pm.me',
    url='https://github.com/charleshouston/ion-channel-ABC',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
