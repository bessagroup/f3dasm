################################################################################
#                              __INIT__.PY                                     #
################################################################################
# This function represents the __init__.py for the collection of modules that
# should allow to generate Gmsh models for meshes using the Gmsh-Python-API.

# set version information
from f3dasm.simulator.gmsh_wrapper._version import __version__

# import modules
from f3dasm.simulator.gmsh_wrapper import (
    Geometry,
    MeshExport,
    Model,
    Visualization,
)

# decide what happens for "from gmshModel import *"
__all__=["Geometry",
         "MeshExport",
         "Model",
         "Visualization",
         "__version__"]
