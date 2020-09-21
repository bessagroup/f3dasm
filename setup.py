'''
Created on 2020-09-18 09:44:00
Last modified on 2020-09-21 08:16:08

@author: L. F. Pereira (lfpereira@fe.up.pt))

Notes
-----
1. See notes in `setup_abaqus.py`.
2. Prefer the use of environments, instead of installing in your main Python
installation.
2. The uninstall requires the manual remove of the created files. Go to your
Python installation (env folder if the case) and remove f3das folder. Open
'easy-install.pth' and delete any reference to f3das.
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
    packages=find_packages(exclude=['abaqus_scripts*', 'examples*']),
    install_requires=install_requires,
    description="Framework for Data-Driven Design and Analysis of Structures.",
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent", ],
    python_requires='>=3.6',)
