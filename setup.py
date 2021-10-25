'''
Created on 2020-09-18 09:44:00
@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
from setuptools import setup, find_packages

# local library
import f3dasm

install_requires = ['numpy', 'matplotlib', 'pandas', 'salib', ]


setup(
    name="f3dasm",
    version=f3dasm.__version__,
    author="L. F. Pereira",
    author_email="lfpereira@fe.up.pt",
    packages=find_packages(include=['f3dasm', 'f3dasm.*']),
    install_requires=install_requires,
    description="Framework for Data-Driven Design and Analysis of Structures.",
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: BSD License",
                 "Operating System :: OS Independent", ],
    python_requires='>=3.8',)
