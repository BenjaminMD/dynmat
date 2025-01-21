from setuptools import find_packages, setup

setup(
    name="dynmat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mendeleev',
        'typing',
    ],
)
