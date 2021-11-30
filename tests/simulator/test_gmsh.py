import unittest
import logging
import os
import uuid
import shutil

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_path import *

from f3dasm.simulator.gmsh_wrapper.Model import RandomInclusionRVE

class TestGmsh(unittest.TestCase):
    """
    This test lets you confirm that if your McNemar Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))

    def test_gmsh(self):
        initParameters={                                                                # save all possible parameters in one dict to facilitate the method call
            "inclusionSets": [1, 13],                                                   # place 13 inclusions with radius 1
            "inclusionType": "Cylinder",                                                # define inclusionType as "Cylinder"
            "inclusionAxis": [0, 0, 8],                                               # define inclusionAxis direction
            "size": [10, 10, 8],                                                      # set RVE size to [10,10,2.5]
            "origin": [0, 0, 0],                                                        # set RVE origin to [0,0,0]
            "periodicityFlags": [1, 1, 1],                                              # define all axis directions as periodic
            "domainGroup": "domain",                                                    # use "domain" as name for the domainGroup
            "inclusionGroup": "inclusions",                                             # use "inclusions" as name for the inclusionGroup
            "gmshConfigChanges": {"General.Terminal": 0,                                # deactivate console output by default (only activated for mesh generation)
                                "Mesh.CharacteristicLengthExtendFromBoundary": 0,     # do not calculate mesh sizes from the boundary by default (since mesh sizes are specified by fields)
            }
        }
        testRVE = RandomInclusionRVE(**initParameters)

        modelingParameters={                                                            # save all possible parameters in one dict to facilitate the method call
            "placementOptions": {"maxAttempts": 10000,                                  # maximum number of attempts to place one inclusion
                                "minRelDistBnd": 0.1,                                  # minimum relative (to inclusion radius) distance to the domain boundaries
                                "minRelDistInc": 0.1,                                  # minimum relative (to inclusion radius) distance to other inclusions}
            }
        }
        testRVE.createGmshModel(**modelingParameters)


        meshingParameters={                                                             # save all possible parameters in one dict to facilitate the method call
            "threads": None,                                                            # do not activate parallel meshing by default
            "refinementOptions": {"maxMeshSize": "auto",                                # automatically calculate maximum mesh size with built-in method
                                "inclusionRefinement": True,                          # flag to indicate active refinement of inclusions
                                "interInclusionRefinement": True,                     # flag to indicate active refinement of space between inclusions (inter-inclusion refinement)
                                "elementsPerCircumference": 18,                       # use 18 elements per inclusion circumference for inclusion refinement
                                "elementsBetweenInclusions": 3,                       # ensure 3 elements between close inclusions for inter-inclusion refinement
                                "inclusionRefinementWidth": 3,                        # use a relative (to inclusion radius) refinement width of 1 for inclusion refinement
                                "transitionElements": "auto",                         # automatically calculate number of transitioning elements (elements in which tanh function jumps from h_min to h_max) for inter-inclusion refinement
                                "aspectRatio": 1.5                                    # aspect ratio for inter-inclusion refinement: ratio of refinement in inclusion distance and perpendicular directions
            }
        }
        testRVE.createMesh(**meshingParameters)

        uid = str(uuid.uuid1())
        ex_dir = os.path.join(self.cur_dir, uid)
        if not os.path.exists(ex_dir):
            os.mkdir(ex_dir)
        gmsh_file = os.path.join(ex_dir,"randomInclusions3DCylinder.xdmf")
        testRVE.saveMesh(gmsh_file)


        # Show resulting mesh
        # To check the generated mesh, the result can also be visualized using built-in
        # methods.
        # testRVE.visualizeMesh()

        # Close Gmsh model
        # For a proper closing of the Gmsh-Python-API, the API has to be finalized. This
        # can be achieved by calling the close() method of the model
        testRVE.close()
        
        shutil.rmtree(ex_dir)

if __name__ == '__main__':
    unittest.main()