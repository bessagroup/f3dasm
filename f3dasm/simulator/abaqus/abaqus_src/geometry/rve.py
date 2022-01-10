'''
Created on 2020-03-24 14:33:48
Last modified on 2020-10-27 18:30:27


@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create an RVE class from which more useful classes can inherit.

Notes
-----
-Based on code developed by M. A. Bessa.

References
----------
1. van der Sluis, O., et al. (2000). Mechanics of Materials 32(8): 449-462.
2. Melro, A. R. (2011). PhD thesis. University of Porto.
'''


# imports

# abaqus
from abaqusConstants import (TWO_D_PLANAR, DEFORMABLE_BODY, ON, FIXED,
                             THREE_D, DELETE, GEOMETRY, TET, FREE,
                             YZPLANE, XZPLANE, XYPLANE)
from part import (EdgeArray, FaceArray)
from mesh import MeshNodeArray
import regionToolset

# third-party
import numpy as np

# local library
from .utils import transform_point
from .utils import get_orientations_360
from ...utils.linalg import symmetricize_vector
from ...utils.linalg import sqrtm


# 2d RVE

class RVE2D:

    def __init__(self, length, width, center, name='RVE'):
        # TODO: generalize length and width to dims
        # TODO: is center really required?
        # TODO: inherit from a simply Python RVE?
        '''
        Notes
        -----
        -1st reference point represents the difference between right bottom
        and left bottom vertices.
        -2nd reference point represents the difference between left top
        and left bottom vertices.
        '''
        self.length = length
        self.width = width
        self.center = center
        self.name = name
        # variable initialization
        self.sketch = None
        self.part = None
        # mesh definitions
        self.mesh_size = .02
        self.mesh_tol = 1e-5
        self.mesh_trial_iter = 1
        self.mesh_refine_factor = 1.25
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4
        # additional variables
        self.edge_positions = ('RIGHT', 'TOP', 'LEFT', 'BOTTOM')
        self.vertex_positions = ('RT', 'LT', 'LB', 'RB')
        self.ref_points_positions = ('LR', 'TB')

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_part(self, model):

        # create RVE
        self._create_RVE_geometry(model)

        # create particular geometry
        # TODO: call it add inner sketches
        self._create_inner_geometry(model)

        # create part
        self.part = model.Part(name=self.name, dimensionality=TWO_D_PLANAR,
                               type=DEFORMABLE_BODY)
        self.part.BaseShell(sketch=self.sketch)

        # create PBCs sets (here because sets are required for meshing purposes)
        self._create_bounds_sets()

    def create_instance(self, model):

        # create instance
        model.rootAssembly.Instance(name=self.name,
                                    part=self.part, dependent=ON)

    def generate_mesh(self):

        # seed part
        self.part.seedPart(size=self.mesh_size,
                           deviationFactor=self.mesh_deviation_factor,
                           minSizeFactor=self.mesh_min_size_factor)

        # seed edges
        edges = [self.part.sets[self._get_edge_name(position)].edges[0] for position in self.edge_positions]
        self.part.seedEdgeBySize(edges=edges, size=self.mesh_size,
                                 deviationFactor=self.mesh_deviation_factor,
                                 constraint=FIXED)
        # generate mesh
        self.part.generateMesh()

    def generate_mesh_for_pbcs(self, fast=False):

        it = 1
        success = False
        while it <= self.mesh_trial_iter and not success:

            # generate mesh
            self.generate_mesh()

            # verify generated mesh
            if fast:
                success = self.verify_mesh_for_pbcs_quick()
            else:
                success = self.verify_mesh_for_pbcs()

            # prepare next iteration
            it += 1
            if not success:
                self.mesh_size /= self.mesh_refine_factor

        return success

    def verify_mesh_for_pbcs(self):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''

        # initialization
        success = True

        # get nodes
        right_nodes, top_nodes, left_nodes, bottom_nodes = self._get_all_sorted_edge_nodes()

        # verify if tolerance is respected (top and bottom)
        for top_node, bottom_node in zip(top_nodes, bottom_nodes):
            if abs(top_node.coordinates[0] - bottom_node.coordinates[0]) > self.mesh_tol:
                # TODO: return insteads
                success = False
                break

        # verify if tolerance is respected (right and left)
        if success:
            for right_node, left_node in zip(right_nodes, left_nodes):
                if abs(right_node.coordinates[1] - left_node.coordinates[1]) > self.mesh_tol:
                    success = False
                    break

        return success

    def verify_mesh_for_pbcs_quick(self):

        # get nodes
        right_nodes, top_nodes, left_nodes, bottom_nodes = self._get_all_sorted_edge_nodes()

        return len(top_nodes) == len(bottom_nodes) and len(right_nodes) == len(left_nodes)

    def apply_pbcs_constraints(self, model):

        # create reference points
        self._create_pbcs_ref_points(model)

        # get nodes
        right_nodes, top_nodes, left_nodes, bottom_nodes = self._get_all_sorted_edge_nodes()

        # get vertices
        rt_vertex, lt_vertex, lb_vertex, rb_vertex = self._get_all_vertices(only_names=True)

        # get reference points
        lr_ref_point, tb_ref_point = self._get_all_ref_points(only_names=True)

        # apply vertex constraints
        for ndof in range(1, 3):
            # right-top and left-bottom nodes
            model.Equation(name='Constraint-RT-LB-V-%i' % ndof,
                           terms=((1.0, rt_vertex, ndof),
                                  (-1.0, lb_vertex, ndof),
                                  (-1.0, lr_ref_point, ndof),
                                  (-1.0, tb_ref_point, ndof)))

            # left-top and right-bottom nodes
            model.Equation(name='Constraint-LT-RB-V-%i' % ndof,
                           terms=((1.0, rb_vertex, ndof),
                                  (-1.0, lt_vertex, ndof),
                                  (-1.0, lr_ref_point, ndof),
                                  (1.0, tb_ref_point, ndof)))

        # left-right edges constraints
        for i, (left_node, right_node) in enumerate(zip(left_nodes[1:-1], right_nodes[1:-1])):

            # create set with individual nodes
            left_node_set_name = 'NODE-L-%i' % i
            left_set_name = '%s.%s' % (self.name, left_node_set_name)
            self.part.Set(name=left_node_set_name, nodes=MeshNodeArray((left_node,)))
            right_node_set_name = 'NODE-R-%i' % i
            self.part.Set(name=right_node_set_name, nodes=MeshNodeArray((right_node,)))
            right_set_name = '%s.%s' % (self.name, right_node_set_name)

            # create constraint
            for ndof in range(1, 3):
                model.Equation(name='Constraint-L-R-%i-%i' % (i, ndof),
                               terms=((1.0, right_set_name, ndof),
                                      (-1.0, left_set_name, ndof),
                                      (-1.0, lr_ref_point, ndof)))

        # top-bottom edges constraints
        for i, (top_node, bottom_node) in enumerate(zip(top_nodes[1:-1], bottom_nodes[1:-1])):

            # create set with individual nodes
            top_node_set_name = 'NODE-T-%i' % i
            top_set_name = '%s.%s' % (self.name, top_node_set_name)
            self.part.Set(name=top_node_set_name, nodes=MeshNodeArray((top_node,)))
            bottom_node_set_name = 'NODE-B-%i' % i
            self.part.Set(name=bottom_node_set_name, nodes=MeshNodeArray((bottom_node,)))
            bottom_set_name = '%s.%s' % (self.name, bottom_node_set_name)

            # create constraint
            for ndof in range(1, 3):
                model.Equation(name='Constraint-T-B-%i-%i' % (i, ndof),
                               terms=((1.0, top_set_name, ndof),
                                      (-1.0, bottom_set_name, ndof),
                                      (-1.0, tb_ref_point, ndof)))

        # ???: need to create fixed nodes? why to not apply bcs e.g. LB and RB?

    def apply_bcs_displacement(self, model, epsilon_11, epsilon_22, epsilon_12,
                               green_lagrange_strain=True):

        # initialization
        epsilon = symmetricize_vector([epsilon_11, epsilon_12, epsilon_22])

        # TODO: receive only small deformations
        # create strain matrix
        if green_lagrange_strain:
            epsilon = self._compute_small_strain(epsilon)

        # apply displacement
        # TODO: continue here

        # TODO: fix left bottom node

    @staticmethod
    def _compute_small_strain(epsilon_lagrange):

        identity = np.identity(2)
        def_grad = sqrtm(2 * epsilon_lagrange + identity)

        return 1 / 2 * (def_grad + np.transpose(def_grad)) - identity

    def _create_RVE_geometry(self, model):

        # rectangle points
        pt1 = transform_point((-self.length / 2., -self.width / 2.),
                              origin_translation=self.center)
        pt2 = transform_point((self.length / 2., self.width / 2.),
                              origin_translation=self.center)

        # sketch
        self.sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                              sheetSize=2 * self.length)
        self.sketch.rectangle(point1=pt1, point2=pt2)

    def _create_bounds_sets(self):

        # TODO: update edge finding to be more robust (with bounding box)

        # create sets
        r = np.sqrt((self.length / 2.) ** 2 + (self.width / 2.)**2)
        for i, (edge_position, vertex_position, theta) in enumerate(
                zip(self.edge_positions, self.vertex_positions, get_orientations_360(0))):

            # find edge
            pt = transform_point((self.length / 2 * np.cos(theta), self.width / 2 * np.sin(theta), 0.),
                                 origin_translation=self.center)
            edge = self.part.edges.findAt((pt,))

            # create edge set
            edge_set_name = self._get_edge_name(edge_position)
            self.part.Set(name=edge_set_name, edges=edge)

            # find vertex
            ratio = self.length / self.width if i % 2 else self.width / self.length
            alpha = np.arctan(ratio)
            pt = transform_point((r * np.cos(alpha), r * np.sin(alpha), 0.),
                                 orientation=theta,
                                 origin_translation=self.center)
            vertex = self.part.vertices.findAt((pt,))

            # create vertex set
            vertex_set_name = self._get_vertex_name(vertex_position)
            self.part.Set(name=vertex_set_name, vertices=vertex)

    def _create_pbcs_ref_points(self, model):
        '''
        Notes
        -----
        -any coordinate for reference points position works.
        '''

        # initialization
        modelAssembly = model.rootAssembly

        # create reference points
        coord = list(self.center) + [0.]
        for position in self.ref_points_positions:
            ref_point = modelAssembly.ReferencePoint(point=coord)
            modelAssembly.Set(name=self._get_ref_point_name(position),
                              referencePoints=((modelAssembly.referencePoints[ref_point.id],)))

    def _get_all_sorted_edge_nodes(self):
        '''
        Notes
        -----
        -output is given in the order of position definition.
        '''

        nodes = []
        for i, position in enumerate(self.edge_positions):
            nodes.append(self._get_edge_nodes(position, sort_direction=(i + 1) % 2))

        return nodes

    def _get_all_vertices(self, only_names=False):
        '''
        Notes
        -----
        -output is given in the order of position definition.
        '''

        if only_names:
            vertices = ['%s.%s' % (self.name, self._get_vertex_name(position)) for position in self.vertex_positions]
        else:
            vertices = [self.part.sets[self._get_vertex_name(position)] for position in self.vertex_positions]

        return vertices

    def _get_all_ref_points(self, model=None, only_names=False):
        '''
        Notes
        -----
        -output is given in the order of position definition.
        -model is required if only_names is False.
        '''

        if only_names:
            ref_points = [self._get_ref_point_name(position) for position in self.ref_points_positions]
        else:
            ref_points = [model.rootAssembly.sets[self._get_ref_point_name(position)] for position in self.ref_points_positions]

        return ref_points

    def _get_edge_nodes(self, position, sort_direction=None):
        edge_name = self._get_edge_name(position)
        nodes = self.part.sets[edge_name].nodes
        if sort_direction is not None:
            nodes = sorted(nodes, key=lambda node: self._get_node_coordinate(node, i=sort_direction))

        return nodes

    @staticmethod
    def _get_edge_name(position):
        return '%s_EDGE' % position

    @staticmethod
    def _get_vertex_name(position):
        return 'VERTEX_%s' % position

    @staticmethod
    def _get_ref_point_name(position):
        return '%s_REF_POINT' % position

    @staticmethod
    def _get_node_coordinate(node, i):
        return node.coordinates[i]


class RVE3D(object):

    def __init__(self, dims, name='RVE'):
        self.dims = dims
        self.name = name
        # mesh definitions
        self.mesh_size = .02
        self.mesh_tol = 1e-5
        self.mesh_trial_iter = 1
        self.mesh_refine_factor = 1.25
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4
        # variable initialization
        self.particles = []
        self.part = None
        # additional variables
        self.face_positions = ('X-', 'X+', 'Y-', 'Y+', 'Z-', 'Z+')

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_particle(self, particle):
        self.particles.append(particle)

    def create_part(self, model):

        # create RVE
        sketch = self._create_RVE_geometry(model)

        # create particular geometry
        # self._create_inner_geometry(model)

        # create part
        self.part = model.Part(name=self.name, dimensionality=THREE_D,
                               type=DEFORMABLE_BODY)
        self.part.BaseSolidExtrude(sketch=sketch, depth=self.dims[2])

        # create particles parts
        for particle in self.particles:
            particle.create_part(model, self)

    def _create_RVE_geometry(self, model):

        # sketch
        sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                         sheetSize=2 * self.dims[0])
        sketch.rectangle(point1=(0., 0.), point2=(self.dims[0], self.dims[1]))

        return sketch

    def create_instance(self, model):

        # initialization
        modelAssembly = model.rootAssembly

        # create rve instance
        modelAssembly.Instance(name=self.name,
                               part=self.part, dependent=ON)

        # create particle instances
        for particle in self.particles:
            particle.create_instance(model)

        # create merged rve
        new_part_name = '{}_WITH_PARTICLES'.format(self.name)
        modelAssembly.InstanceFromBooleanMerge(name=new_part_name,
                                               instances=modelAssembly.instances.values(),
                                               keepIntersections=ON,
                                               originalInstances=DELETE,
                                               domain=GEOMETRY)
        self.part = model.parts[new_part_name]

        # create PBCs sets (here because sets are required for meshing purposes)
        self._create_bounds_sets()

    def generate_mesh(self, face_by_closest=True, simple_trial=False):

        # set mesh control
        self.part.setMeshControls(regions=self.part.cells, elemShape=TET,
                                  technique=FREE,)

        # generate mesh by simple strategy
        if simple_trial:
            # generate mesh
            self._seed_part()
            self.part.generateMesh()

            # verify mesh
            success = self.verify_mesh_for_pbcs(face_by_closest=face_by_closest)

        # retry meshing if unsuccessful
        if not simple_trial or not success:
            if simple_trial:
                print("Warning: Unsucessful mesh generation. Another strategy will be tried out")

            # retry meshing
            self._retry_meshing()

            # verify mesh
            success = self.verify_mesh_for_pbcs(face_by_closest=face_by_closest)

            if not success:
                print("Warning: Unsucessful mesh generation")

    def _seed_part(self):
        # seed part
        self.part.seedPart(size=self.mesh_size,
                           deviationFactor=self.mesh_deviation_factor,
                           minSizeFactor=self.mesh_min_size_factor)

        # seed exterior edges
        exterior_edges = self._get_exterior_edges()
        self.part.seedEdgeBySize(edges=exterior_edges, size=self.mesh_size,
                                 deviationFactor=self.mesh_deviation_factor,
                                 constraint=FIXED)

    def _retry_meshing(self):

        # delete older mesh
        self.part.deleteMesh()

        # 8-cube partitions (not necessarily midplane)
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for i, plane in enumerate(planes):
            feature = self.part.DatumPlaneByPrincipalPlane(principalPlane=plane,
                                                           offset=self.dims[i] / 2)
            datum = self.part.datums[feature.id]
            self.part.PartitionCellByDatumPlane(datumPlane=datum, cells=self.part.cells)

        # reseed (due to partitions)
        self._seed_part()

        # z+
        # generate local mesh
        k = self._mesh_half_periodically()
        # transition
        axis = 2
        faces = self.part.faces.getByBoundingBox(zMin=self.dims[2])
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # z-
        self._mesh_half_periodically(k, zp=False)

    def _mesh_half_periodically(self, k=0, zp=True):
        # initialization
        if zp:
            kwargs = {'zMin': self.dims[2] / 2}
        else:
            kwargs = {'zMax': self.dims[2] / 2}

        pickedCells = self.part.cells.getByBoundingBox(xMin=self.dims[0] / 2,
                                                       yMin=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 0
        faces = self.part.faces.getByBoundingBox(xMin=self.dims[0],
                                                 yMin=self.dims[1] / 2,
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMax=self.dims[0] / 2,
                                                       yMin=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 1
        faces = self.part.faces.getByBoundingBox(yMin=self.dims[1],
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMax=self.dims[0] / 2,
                                                       yMax=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 0
        faces = self.part.faces.getByBoundingBox(xMax=0.,
                                                 yMax=self.dims[1] / 2,
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k, s=1)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMin=self.dims[0] / 2,
                                                       yMax=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)

        return k

    def _copy_face_mesh_pattern(self, face, axis, k, s=0):
        pt = list(face.pointOn[0])
        pt[axis] = self.dims[axis] * s
        target_face = self.part.faces.findAt(pt)
        face_set = self.part.Set(name='_FACE_{}'.format(k), faces=FaceArray((face,)))
        self.part.Set(name='_TARGET_FACE_{}'.format(k), faces=FaceArray((target_face,)))
        k += 1
        vertex_indices = face.getVertices()
        vertices = [self.part.vertices[index].pointOn[0] for index in vertex_indices]
        coords = [list(vertex) for vertex in vertices]
        for coord in coords:
            coord[axis] = self.dims[axis] * s
        nodes = [face_set.nodes.getClosest(vertex) for vertex in vertices]
        self.part.copyMeshPattern(faces=face_set, targetFace=target_face,
                                  nodes=nodes,
                                  coordinates=coords)

    def _get_edge_nodes(self, pos_i, pos_j, sort_direction=None):
        edge_name = self._get_edge_name(pos_i, pos_j)
        nodes = self.part.sets[edge_name].nodes
        if sort_direction is not None:
            nodes = sorted(nodes, key=lambda node: self._get_node_coordinate(node, i=sort_direction))

        return nodes

    def _get_face_nodes(self, pos, sort_direction_i=None, sort_direction_j=None):
        face_name = self._get_face_name(pos)
        nodes = self.part.sets[face_name].nodes
        if sort_direction_i is not None and sort_direction_j is not None:
            d = self._get_decimal_places()
            nodes = sorted(nodes, key=lambda node: (
                self._get_node_coordinate_with_tol(node, i=sort_direction_i, decimal_places=d),
                self._get_node_coordinate_with_tol(node, i=sort_direction_j, decimal_places=d),))

        return nodes

    def verify_mesh_for_pbcs(self, face_by_closest=True):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''

        # verify edges
        if not self._verify_edges():
            return False

        # verify faces
        if face_by_closest:
            return self._verify_faces_by_closest()
        else:
            return self._verify_faces_by_sorting()

    def _verify_edges(self):
        for i, pos_i in enumerate(zip(self.face_positions[:-2:2], self.face_positions[1:-2:2])):
            for j, pos_j in enumerate(zip(self.face_positions[2 * (i + 1)::2], self.face_positions[(2 * (i + 1) + 1)::2])):
                # get possible combinations
                pos_comb = self._get_edge_combinations(pos_i, pos_j)
                n_comb = len(pos_comb)

                # get sorted nodes for each edge
                nodes = []
                k = self._get_edge_sort_direction(i, i + j + 1)
                for (pos_i_, pos_j_) in pos_comb:
                    nodes.append(self._get_edge_nodes(pos_i_, pos_j_,
                                                      sort_direction=k))

                # verify sizes
                sizes = [len(node_list) for node_list in nodes]
                if len(set(sizes)) > 1:
                    return False

                # verify if tolerance is respected
                for n, node in enumerate(nodes[0]):
                    for m in range(1, n_comb):
                        if abs(node.coordinates[k] - nodes[m][n].coordinates[k]) > self.mesh_tol:
                            # create set with error nodes
                            nodes_ = [node_list[n] for node_list in nodes]
                            set_name = self._verify_set_name('_ERROR_EDGE_NODES')
                            self.part.Set(set_name,
                                          nodes=MeshNodeArray(nodes_))
                            return False

        return True

    def _verify_set_name(self, name):
        new_name = name
        i = 1
        while new_name in self.part.sets.keys():
            i += 1
            new_name = '{}_{}'.format(new_name, i)

        return new_name

    def _verify_faces_by_sorting(self):
        '''
        Notes
        -----
        1. the sort method is less robust to due rounding errors. `mesh_tol`
        is used to increase its robustness, but find by closest should be
        preferred.
        '''
        for i, (pos_i, pos_j) in enumerate(zip(self.face_positions[::2], self.face_positions[1::2])):

            # get nodes
            j, k = self._get_face_sort_directions(i)
            nodes_i = self._get_face_nodes(pos_i, j, k)
            nodes_j = self._get_face_nodes(pos_j, j, k)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for n, (node, node_cmp) in enumerate(zip(nodes_i, nodes_j)):

                # verify tolerance
                if not self._verify_tol_face_nodes(node, node_cmp, j, k):
                    return False

        return True

    def _verify_faces_by_closest(self):
        '''
        Notes
        -----
        1. at first sight, it appears to be more robust than using sort.
        '''
        for i, (pos_i, pos_j) in enumerate(zip(self.face_positions[::2], self.face_positions[1::2])):

            # get nodes
            j, k = self._get_face_sort_directions(i)
            nodes_i = self._get_face_nodes(pos_i)
            nodes_j = self._get_face_nodes(pos_j)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for node in nodes_i:
                node_cmp = nodes_j.getClosest(node.coordinates)

                # verify tolerance
                if not self._verify_tol_face_nodes(node, node_cmp, j, k):
                    return False

        return True

    def _verify_tol_face_nodes(self, node, node_cmp, j, k):
        if abs(node.coordinates[j] - node_cmp.coordinates[j]) > self.mesh_tol or abs(node.coordinates[k] - node_cmp.coordinates[k]) > self.mesh_tol:
            # create set with error nodes
            set_name = self._verify_set_name('_ERROR_FACE_NODES')
            self.part.Set(set_name,
                          nodes=MeshNodeArray((node, node_cmp)))
            return False

        return True

    def _get_edge_combinations(self, pos_i, pos_j):
        comb = []
        for pos_i_ in pos_i:
            for pos_j_ in pos_j:
                comb.append([pos_i_, pos_j_])

        return comb

    def _get_edge_sort_direction(self, i, j):
        if 0 not in [i, j]:
            return 0
        elif 1 not in [i, j]:
            return 1
        else:
            return 2

    def _get_face_sort_directions(self, i):
        if i == 0:
            return 1, 2
        elif i == 1:
            return 0, 2
        else:
            return 0, 1

    def _get_exterior_edges(self):
        exterior_edges = []
        for i, position in enumerate(self.face_positions):
            k = int(i // 2)
            face_name = self._get_face_name(position)
            sign = 1 if '+' in face_name else 0  # 0 to represent negative face
            face_axis = face_name.split('_')[1][0].lower()
            var_name = '{}Min'.format(face_axis) if sign else '{}Max'.format(face_axis)
            kwargs = {var_name: self.dims[k] * sign}
            edges = self.part.edges.getByBoundingBox(**kwargs)
            exterior_edges.extend(edges)

        # unique edges
        edge_indices, unique_exterior_edges = [], []
        for edge in exterior_edges:
            if edge.index not in edge_indices:
                unique_exterior_edges.append(edge)
                edge_indices.append(edge.index)

        return EdgeArray(unique_exterior_edges)

    def _create_bounds_sets(self):

        # faces
        self._create_bound_faces_sets()

        # edges
        self._create_bound_edges_sets()

    def _create_bound_faces_sets(self):

        for i, position in enumerate(self.face_positions):
            k = i // 2
            face_name, var_name, dim = self._get_face_info(k, position)
            kwargs = {var_name: dim}
            faces = self.part.faces.getByBoundingBox(**kwargs)
            self.part.Set(name=face_name, faces=faces)
            # self.part.Surface(name='SUR{}'.format(face_name), side1Faces=faces)

    def _create_bound_edges_sets(self):
        for i, pos_i in enumerate(self.face_positions[:-2]):
            k_i = i // 2
            _, var_name_i, dim_i = self._get_face_info(k_i, pos_i)
            for j, pos_j in enumerate(self.face_positions[2 * (k_i + 1):]):
                k_j = j // 2
                _, var_name_j, dim_j = self._get_face_info(k_j, pos_j)
                edge_name = self._get_edge_name(pos_i, pos_j)
                kwargs = {var_name_i: dim_i, var_name_j: dim_j}
                edges = self.part.edges.getByBoundingBox(**kwargs)
                self.part.Set(name=edge_name, edges=edges)

    def _get_face_info(self, i, position):
        face_name = self._get_face_name(position)
        sign = 1 if '+' in face_name else 0  # 0 to represent negative face
        face_axis = face_name.split('_')[1][0].lower()
        var_name = '{}Min'.format(face_axis) if sign else '{}Max'.format(face_axis)
        dim = self.dims[i] * sign

        return face_name, var_name, dim

    @staticmethod
    def _get_edge_name(pos_i, pos_j):
        return 'EDGE_{}{}'.format(pos_i, pos_j)

    @staticmethod
    def _get_face_name(position):
        return 'FACE_{}'.format(position)

    @staticmethod
    def _get_node_coordinate(node, i):
        return node.coordinates[i]

    @staticmethod
    def _get_node_coordinate_with_tol(node, i, decimal_places):
        return round(node.coordinates[i], decimal_places)

    def _get_decimal_places(self):
        d = 0
        aux = 1
        while aux > self.mesh_tol:
            d += 1
            aux = 10**(-d)

        return d
