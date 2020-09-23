'''
Created on 2020-09-22 17:48:30
Last modified on 2020-09-22 17:51:30

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

'''
Created on 2020-09-08 17:01:36
Last modified on 2020-09-10 12:07:23

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *


#%% sketch and part

def overvelde_example_model(height, width):

    # sketch and part
    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=20.0)
    mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(-width / 2.0, -height / 2.0),
                                                            point2=(width / 2.0, height / 2.0))
    mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['Part-1'].BaseShell(sketch=mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']

    # material, section and section assignment
    mdb.models['Model-1'].Material(name='ExampleMaterial')
    mdb.models['Model-1'].materials['ExampleMaterial'].Elastic(table=((
        1000000000.0, 0.3), ))
    mdb.models['Model-1'].HomogeneousSolidSection(material='ExampleMaterial', name='Section-1', thickness=None)
    mdb.models['Model-1'].parts['Part-1'].SectionAssignment(offset=0.0,
                                                            offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
                                                                faces=mdb.models['Model-1'].parts['Part-1'].faces.findAt(((0.0, 0.0,
                                                                                                                           0.0), (0.0, 0.0, 1.0)))), sectionName='Section-1')

    # sets and surfaces
    mdb.models['Model-1'].parts['Part-1'].Set(edges=mdb.models['Model-1'].parts['Part-1'].edges.findAt(((-width / 2.0, -height / 4.0, 0.0), )),
                                              name='left_edge')
    mdb.models['Model-1'].parts['Part-1'].Surface(name='top_edge', side1Edges=mdb.models['Model-1'].parts['Part-1'].edges.findAt(((-width / 4.0, height / 2.0, 0.0), )))

    # mesh control, element type and mesh
    mdb.models['Model-1'].parts['Part-1'].setMeshControls(elemShape=QUAD, regions=mdb.models['Model-1'].parts['Part-1'].faces.findAt(((0.0, 0.0,
                                                                                                                                       0.0), )), technique=STRUCTURED)
    mdb.models['Model-1'].parts['Part-1'].setElementType(elemTypes=(ElemType(
        elemCode=CPS8R, elemLibrary=STANDARD), ElemType(elemCode=CPS6M,
                                                        elemLibrary=STANDARD)), regions=(
        mdb.models['Model-1'].parts['Part-1'].faces.findAt(((0.0, 0.0,
                                                             0.0), )), ))
    mdb.models['Model-1'].parts['Part-1'].seedPart(deviationFactor=0.1,
                                                   minSizeFactor=0.1, size=height / 4)
    mdb.models['Model-1'].parts['Part-1'].generateMesh()

    # assembly
    mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
    mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-1-1',
                                                part=mdb.models['Model-1'].parts['Part-1'])
    mdb.models['Model-1'].rootAssembly.regenerate()

    # step, BCs and loads
    mdb.models['Model-1'].StaticStep(initialInc=0.1, maxInc=0.1, name='Step-1',
                                     nlgeom=ON, previous='Initial')
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
                                         distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-1', region=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].sets['left_edge'],
                                         u1=0.0, u2=0.0, ur3=0.0)
    mdb.models['Model-1'].Pressure(amplitude=UNSET, createStepName='Step-1',
                                   distributionType=UNIFORM, field='', magnitude=-1e-05, name='Load-1',
                                   region=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].surfaces['top_edge'])

    # job and job submission
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
            explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
            memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF,
            multiprocessingMode=DEFAULT, name='EXAMPLE', nodalOutputPrecision=SINGLE,
            numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.jobs['EXAMPLE'].submit(consistencyChecking=OFF)
    mdb.jobs['EXAMPLE'].waitForCompletion()
