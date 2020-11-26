'''
Created on 2020-10-15 09:36:46
Last modified on 2020-11-26 12:49:39

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (DEFORMABLE_BODY, THREE_D, ON, COUNTERCLOCKWISE,
                             CLOCKWISE, YZPLANE, XYPLANE, XZPLANE, SIDE1, LINE)

# standard library
import copy
from abc import ABCMeta
import math

# local library
from .base import Geometry


# abstract object

class MicroShape(Geometry):
    __metaclass__ = ABCMeta

    def __init__(self, name, material=None, default_mesh=False):
        super(MicroShape, self).__init__(default_mesh=default_mesh)
        self.name = name
        self.material = material

    def draw_in_sketch(self, sketch):
        '''
        Perform operations in main sketch.
        '''
        pass

    def make_partition(self, model, tmp_part):
        pass


class PeriodicBall(MicroShape):
    __metaclass__ = ABCMeta

    def __init__(self, name, r, center, tol=1e-4, bounds=None, material=None):
        super(PeriodicBall, self).__init__(name, material, default_mesh=False)
        self.r = r
        self.tol = tol
        self.bounds = bounds
        # initialize variables
        self.particles = []
        # create required particles
        self.add_center(center)

    def _center_exists(self, cmp_center):
        exists = False
        d = len(cmp_center)
        for particle in self.particles:
            center = particle.center
            k = 0
            for elem_center, elem_cmpcenter in zip(center, cmp_center):
                if abs(elem_center - elem_cmpcenter) < self.tol:
                    k += 1
            if k == d:
                exists = True
                break

        return exists

    def _is_inside(self, center):
        dist_squared = self.r**2
        for (c, bounds) in zip(center, self.bounds):
            if c < bounds[0]:
                dist_squared -= (c - bounds[0])**2
            elif c > bounds[1]:
                dist_squared -= (c - bounds[1])**2

        return dist_squared > 0

    def add_center(self, center):
        if self._center_exists(center) or (self.bounds is not None and not self._is_inside(center)):
            return
        else:
            self._add_particle(center)

        for i, (pos_center, bounds) in enumerate(zip(center, self.bounds)):
            dim = bounds[1] - bounds[0]
            if (pos_center + self.r) > bounds[1]:
                new_center = copy.copy(center)
                new_center[i] -= dim
                self.add_center(new_center)
            elif (pos_center - self.r) < bounds[0]:
                new_center = copy.copy(center)
                new_center[i] += dim
                self.add_center(new_center)

    def _get_name(self):
        return '{}_{}'.format(self.name, len(self.particles))


# concrete shapes

class PeriodicSphere(PeriodicBall):

    def _add_particle(self, center):
        name = self._get_name()
        self.particles.append(Sphere(name=name, r=self.r, center=center, tol=self.tol,
                                     bounds=self.bounds, material=self.material))

    def create_part(self, model):

        parts = []
        for particle in self.particles:
            parts.append(particle.create_part(model))

        return parts

    def create_instance(self, model):

        instances = []
        for particle in self.particles:
            instances.append(particle.create_instance(model))

        return instances

    def generate_mesh(self):

        for particle in self.particles:
            particle.generate_mesh()


class PeriodicCircle(PeriodicBall):

    def _add_particle(self, center):
        name = self._get_name()
        self.particles.append(Circle(name=name, r=self.r, center=center,
                                     tol=self.tol, bounds=self.bounds,
                                     material=self.material))

    def draw_in_sketch(self, sketch):
        for particle in self.particles:
            particle.draw_in_sketch(sketch)

    def make_partition(self, model, tmp_part):

        # verification
        if self.material is None:
            return False

        # create partitions
        for particle in self.particles:
            particle.make_partition(model, tmp_part)

        return True

    def get_region(self, tmp_part):
        return [particle.get_region(tmp_part) for particle in self.particles]


class Sphere(MicroShape):

    def __init__(self, r, center, tol=1e-4, name='SPHERE',
                 bounds=(), material=None):
        '''
        Parameters
        ----------
        bounds : array-like e.g. ((x_min, x_max), (y_min, y_max))
            Bounds of the e.g. RVE. Sphere is cutted to be contained within
            bounds.
        '''
        super(Sphere, self).__init__(name, material)
        self.r = r
        self.center = center
        self.tol = tol
        self.bounds = bounds

    def create_part(self, model):

        # sketch
        sketch = self._create_sketch(model)

        # part
        self.part = model.Part(name=self.name, dimensionality=THREE_D,
                               type=DEFORMABLE_BODY)
        self.part.BaseSolidRevolve(sketch=sketch, angle=360.,)

        # partitions for meshing
        self._create_partitions()

        # remove cells
        if self._is_to_remove_cells():
            self._remove_cells()

        # assign section
        if self.material is not None:
            self._assign_section(region=(self.part.cells,))

        return self.part

    def _create_sketch(self, model):

        a, b = self.center[1] + self.r, self.center[1] - self.r

        # sketch
        sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                         sheetSize=2 * self.r)
        sketch.ConstructionLine(point1=(self.center[0], self.r),
                                point2=(self.center[0], -self.r))
        sketch.ArcByCenterEnds(center=self.center[:2],
                               point1=(self.center[0], a),
                               point2=(self.center[0], b), direction=CLOCKWISE)
        sketch.Line(point1=(self.center[0], a), point2=(self.center[0], b))

        return sketch

    def create_instance(self, model):
        instance = model.rootAssembly.Instance(name=self.name,
                                               part=self.part, dependent=ON)
        instance.translate(vector=(0., 0., self.center[2]))

        return instance

    def _create_partitions(self):
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for c, plane in zip(self.center, planes):
            offset = c if plane is not XYPLANE else 0.
            feature = self.part.DatumPlaneByPrincipalPlane(principalPlane=plane,
                                                           offset=offset)
            datum = self.part.datums[feature.id]
            self.part.PartitionCellByDatumPlane(datumPlane=datum, cells=self.part.cells)

    def _is_to_remove_cells(self):
        for bounds, c in zip(self.bounds, self.center):
            if c - self.r < bounds[0] or c + self.r > bounds[1]:
                return True

        return False

    def _remove_cells(self,):

        # initialization
        planes = [YZPLANE, XZPLANE, XYPLANE]
        variables = ['x', 'y', 'z']

        # delete cells
        for i in range(3):
            # partition position
            if (self.center[i] + self.r) > self.bounds[i][1]:
                sign = 1
            elif (self.center[i] - self.r) < self.bounds[i][0]:
                sign = -1
            else:
                continue

            # partition by datum
            if sign > 0:
                x_max = self.bounds[i][1] if i != 2 else self.bounds[i][1] - self.center[i]
            else:
                x_max = self.bounds[i][0] if i != 2 else self.bounds[i][0] - self.center[i]
            feature = self.part.DatumPlaneByPrincipalPlane(principalPlane=planes[i],
                                                           offset=x_max)
            datum = self.part.datums[feature.id]
            try:
                self.part.PartitionCellByDatumPlane(datumPlane=datum, cells=self.part.cells)
            except:  # in case partition already exists
                pass
            var_name = '{}Max'.format(variables[i]) if sign == -1 else '{}Min'.format(variables[i])
            kwargs = {var_name: x_max}
            faces = self.part.faces.getByBoundingBox(**kwargs)
            faces_to_delete = []
            for face in faces:
                if abs(face.getNormal()[i]) != 1.0 or (sign == 1 and face.pointOn[0][i] - self.tol > x_max) or (sign == -1 and face.pointOn[0][i] + self.tol < x_max):
                    faces_to_delete.append(face)

            # remove faces
            try:
                self.part.RemoveFaces(faceList=faces_to_delete, deleteCells=False)
            except:  # in case faces where already removed
                pass


class Circle(MicroShape):

    def __init__(self, r, center, tol=1e-4, name='CIRCLE', bounds=(), material=None):
        '''
        Parameters
        ----------
        bounds : array-like e.g. ((x_min, x_max), (y_min, y_max))
            Bounds of the e.g. RVE. Sphere is cutted to be contained within
            bounds.
        '''
        super(Circle, self).__init__(name, material)
        self.r = r
        self.center = center
        self.tol = tol
        self.bounds = bounds

    def draw_in_sketch(self, sketch):

        # verification
        if self.material is not None:
            return

        # draw in sketch
        if self._is_to_remove_cells():
            self._draw_partial_arc(sketch)
        else:
            self._draw_circle(sketch)

    def _draw_circle(self, sketch):
        pt = (self.center[0] + self.r, self.center[1])
        sketch.CircleByCenterPerimeter(center=self.center, point1=pt)

    def _draw_partial_arc(self, sketch):
        '''
        Notes
        -----
        Vertices cannot be cut (otherwise it will mess the application of the
        boundary conditions).
        '''

        # get intersection points
        inter_pts = self._get_intersection_pts()

        # find right points order
        if inter_pts[0][0] == self.bounds[0][0] or inter_pts[0][1] == self.bounds[1][1]:
            inter_pts[0], inter_pts[1] = inter_pts[1], inter_pts[0]

        # draw circle
        arc = sketch.ArcByCenterEnds(center=self.center, point1=inter_pts[0],
                                     point2=inter_pts[1],
                                     direction=COUNTERCLOCKWISE)

        # create breakpoints
        for pt in inter_pts:

            # find line (findAt sometimes finds ARC)
            geom = [g for g in sketch.geometry.values() if g.curveType == LINE]
            for g in geom:
                vertices = g.getVertices()
                v1, v2 = vertices[0].coords, vertices[1].coords
                index = 1 if v1[0] == v2[0] else 0
                cpl_index = 0 if v1[0] == v2[0] else 1
                lv, rv = min(v1[index], v2[index]), max(v1[index], v2[index])
                if pt[cpl_index] == v1[cpl_index] and lv <= pt[index] <= rv:
                    line = g
                    break

            # break line
            sketch.breakCurve(curve1=line, curve2=arc, point1=pt, point2=pt)

        # delete curves
        geom = [g for g in sketch.geometry.values() if g.curveType == LINE]
        for g in geom:
            v1, v2 = g.getVertices()
            if v1.coords in inter_pts and v2.coords in inter_pts:
                sketch.delete((g,))
                break

    def _is_to_remove_cells(self):
        for bounds, c in zip(self.bounds, self.center):
            if c - self.r < bounds[0] or c + self.r > bounds[1]:
                return True

        return False

    def _get_intersection_pts(self):

        # initialization
        xc, yc = self.center
        pts = []

        # in vertical edges
        for x in self.bounds[0]:
            aux = self.r**2 - (x - xc)**2
            if aux >= 0.:
                ys1 = math.sqrt(aux) + yc
                ys2 = -math.sqrt(aux) + yc
                if ys1 < self.bounds[1][1]:
                    pts.append((x, ys1))
                if ys2 > self.bounds[1][0]:
                    pts.append((x, ys2))

        # in horizontal edges
        for y in self.bounds[1]:
            aux = self.r**2 - (y - yc)**2
            if aux >= 0.:
                xs1 = math.sqrt(aux) + xc
                xs2 = -math.sqrt(aux) + xc
                if xs1 < self.bounds[0][1]:
                    pts.append((xs1, y))
                if xs2 > self.bounds[0][0]:
                    pts.append((xs2, y))

        return pts

    def make_partition(self, model, part):

        # verification
        if self.material is None:
            return False

        # create sketch
        face = part.faces[0]
        transf_sketch = part.MakeSketchTransform(sketchPlane=face, sketchPlaneSide=SIDE1,
                                                 origin=[0., 0., 0.])
        sketch = model.ConstrainedSketch(name=self.name, sheetSize=2 * self.r,
                                         transform=transf_sketch)

        # draw in sketch
        pt = (self.center[0] + self.r, self.center[1])
        sketch.CircleByCenterPerimeter(center=self.center, point1=pt)

        # partition sketch
        part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)

        # assign material
        region = self.get_region(part)
        self._assign_section(part=part, region=region)

        return True

    def get_region(self, part):
        center1 = list(self.center) + [0.]
        center2 = list(self.center) + [1.]
        faces = part.faces.getByBoundingCylinder(center1, center2, self.r + self.tol)

        return (faces[0],)
