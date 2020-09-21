'''
Created on 2020-09-10 15:36:46
Last modified on 2020-09-21 08:18:25

@author: L. F. Pereira (lfpereira@fe.up.pt))

Notes
-----
1. It is not mandatory to install F3DAS in Abaqus, but if not, then F3DAS folder
has to be copied to the directory that contains the scripts for the application
of the framework. Both setups (in Python and in Abaqus) are required to avoid
copying F3DAS whenever the framework is run.
2. If F3DAS is installed in Abaqus and its folder is also in the running
directory, then problems may arise due to version conflicts, because Python
will use the version on the running directory (even if setup in Python was
done), whereas Abaqus will use the version installed there. If both versions
are equal, it will work properly.
4. The Abaqus uninstall requires the manual removal of the created files. Go to
"C:\SIMULIA\CAE\2017\win_b64\tools\SMApy\python2.7\Lib\site-packages" (or
equivalent) and remove f3das-related files.

References
----------
1. https://stackoverflow.com/a/12966345/11011913
2. Reitz et al. The Hitchhiker's Guide To Python. 2016
'''

# imports

# standard library
from distutils.core import setup
from pkgutil import walk_packages

# local library
import f3das


# define objects
def find_packages(path, prefix):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name


# setup

setup(name='f3das',
      author='L. F. Pereira',
      version=f3das.__version__,
      packages=list(find_packages(f3das.__path__, f3das.__name__)),)
