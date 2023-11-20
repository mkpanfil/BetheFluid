from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="BetheFluid",
    version="0.2",
    description='Python package for solving GHD equations',
    author='Antoni Lis',
    packages=['BetheFluid'],
    install_requires=requirements,
    zip_safe=False
)