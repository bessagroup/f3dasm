'''
Created on 2020-09-18 09:44:00
Last modified on 2020-09-23 17:07:39

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
from setuptools import setup, find_packages

# local library
import f3das


# setup

install_requires = ['numpy', 'matplotlib', 'pandas', 'salib',
                    'scipy']


setup(
    name="f3das",
    version=f3das.__version__,
    author="L. F. Pereira",
    author_email="lfpereira@fe.up.pt",
    packages=find_packages(include=['f3das', 'f3das.*']),
    install_requires=install_requires,
    description="Framework for Data-Driven Design and Analysis of Structures.",
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent", ],
    python_requires='>=3.6',)
