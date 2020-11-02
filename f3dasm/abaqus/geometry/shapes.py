'''
Created on 2020-10-15 09:36:46
Last modified on 2020-11-02 09:03:28

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (DEFORMABLE_BODY, THREE_D, ON, CLOCKWISE,
                             YZPLANE, XYPLANE, XZPLANE)

# standard library
import copy


# object definition

class Sphere(object):

    def __init__(self, r, center=None, periodic=False, tol=1e-4, name='SPHERE',
                 dims=None):
        self.r = r
        self.centers = []
        self.periodic = periodic
        self.tol = tol
        self.name = name
        # initialize variables
        self.parts = []
        # mesh definitions
        self.mesh_size = .02
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4
        # update centers
        if center is not None:
            self.add_center(center, dims)

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _center_exists(self, cmp_center):
        exists = False
        d = len(cmp_center)
        for center in self.centers:
            k = 0
            for elem_center, elem_cmpcenter in zip(center, cmp_center):
                if abs(elem_center - elem_cmpcenter) < self.tol:
                    k += 1
            if k == d:
                exists = True
                break

        return exists

    def _is_inside(self, center, dims):
        dist_squared = self.r**2
        for (c, dim) in zip(center, dims):
            if c < 0.:
                dist_squared -= c**2
            elif c > dim:
                dist_squared -= (c - dim)**2

        return dist_squared > 0

    def add_center(self, center, dims=None):
        if self._center_exists(center) or (dims is not None and not self._is_inside(center, dims)):
            return
        else:
            self.centers.append(center)
            if not self.periodic or dims is None:
                return

        for i, (pos_center, dim) in enumerate(zip(center, dims)):
            if (pos_center + self.r) > dim:
                new_center = copy.copy(center)
                new_center[i] -= dim
                self.add_center(new_center, dims)
            elif (pos_center - self.r) < 0:
                new_center = copy.copy(center)
                new_center[i] += dim
                self.add_center(new_center, dims)

    def create_part(self, model, rve=None):

        for i, center in enumerate(self.centers):
            name = '{}_{}'.format(self.name, i)
            self._create_part_by_center(model, center, name, rve)

    def _create_part_by_center(self, model, center, name, rve,):
        a, b = center[1] + self.r, center[1] - self.r

        # sketch
        sketch = model.ConstrainedSketch(name=name + '_PROFILE',
                                         sheetSize=2 * self.r)
        sketch.ConstructionLine(point1=(center[0], self.r), point2=(center[0], -self.r))
        sketch.ArcByCenterEnds(center=center[:2], point1=(center[0], a),
                               point2=(center[0], b), direction=CLOCKWISE)
        sketch.Line(point1=(center[0], a), point2=(center[0], b))

        # part
        part = model.Part(name=name, dimensionality=THREE_D,
                          type=DEFORMABLE_BODY)
        part.BaseSolidRevolve(sketch=sketch, angle=360.,)
        self.parts.append(part)

        # partitions for meshing
        self._create_partitions(center, part)

        # remove cells
        if rve is not None:
            self._remove_cells(center, part, rve)

    def _create_partitions(self, center, part):
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for c, plane in zip(center, planes):
            offset = c if plane is not XYPLANE else 0.
            feature = part.DatumPlaneByPrincipalPlane(principalPlane=plane,
                                                      offset=offset)
            datum = part.datums[feature.id]
            part.PartitionCellByDatumPlane(datumPlane=datum, cells=part.cells)

    def _remove_cells(self, center, part, rve):

        # initialization
        planes = [YZPLANE, XZPLANE, XYPLANE]
        variables = ['x', 'y', 'z']

        # delete cells
        for i in range(3):
            # partition position
            if (center[i] + self.r) > rve.dims[i]:
                sign = 1
            elif (center[i] - self.r) < 0.:
                sign = -1
            else:
                continue

            # partition by datum
            if sign > 0:
                x_max = rve.dims[i] if i != 2 else rve.dims[i] - center[i]
            else:
                x_max = 0. if i != 2 else -center[i]
            feature = part.DatumPlaneByPrincipalPlane(principalPlane=planes[i],
                                                      offset=x_max)
            datum = part.datums[feature.id]
            try:
                part.PartitionCellByDatumPlane(datumPlane=datum, cells=part.cells)
            except:  # in case partition already exists
                pass
            var_name = '{}Max'.format(variables[i]) if sign == -1 else '{}Min'.format(variables[i])
            kwargs = {var_name: x_max}
            faces = part.faces.getByBoundingBox(**kwargs)
            faces_to_delete = []
            for face in faces:
                if abs(face.getNormal()[i]) != 1.0 or (sign == 1 and face.pointOn[0][i] - self.tol > x_max) or (sign == -1 and face.pointOn[0][i] + self.tol < x_max):
                    faces_to_delete.append(face)

            # remove faces
            try:
                part.RemoveFaces(faceList=faces_to_delete, deleteCells=False)
            except:  # in case faces where already removed
                pass

    def create_instance(self, model):

        # create instance
        for i, (center, part) in enumerate(zip(self.centers, self.parts)):
            name = '{}_{}'.format(self.name, i)
            instance = model.rootAssembly.Instance(name=name,
                                                   part=part, dependent=ON)
            instance.translate(vector=(0., 0., center[2]))

    def generate_mesh(self):
        for part in self.parts:
            part.seedPart(size=self.mesh_size,
                          deviationFactor=self.mesh_deviation_factor,
                          minSizeFactor=self.mesh_min_size_factor)

            part.generateMesh()
