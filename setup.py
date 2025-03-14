from setuptools import setup, find_packages

setup(
    name='marl-wipysim',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'simpy',
        'setuptools',
        'networkx',
    ],)