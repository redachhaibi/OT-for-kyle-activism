#!/usr/bin/env python

from setuptools import setup, find_packages

# Setup
setup(
    name='OT-for-kyle-activism',
    version='0.1',
    description='',
    long_description=""" 
    """,
    url='',
    author='Reda Chhaibi',
    author_email='chhaibi.reda@gmail.com',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    install_requires=["numpy", "matplotlib", "scipy", "plotly"],
    keywords='',
    packages=find_packages(),
)
