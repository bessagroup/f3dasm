'''
Created on 2020-03-24 14:33:48
Last modified on 2020-11-09 15:49:13

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
from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (TWO_D_PLANAR, DEFORMABLE_BODY, ON, FIXED,
                             THREE_D, DELETE, GEOMETRY, TET, FREE,
                             YZPLANE, XZPLANE, XYPLANE)
from part import (EdgeArray, FaceArray)
from mesh import MeshNodeArray

# standard
from abc import ABCMeta
from collections import OrderedDict

# local library
from ..modelling.bcs import DisplacementBC
from ...utils.linalg import symmetricize_vector
from ...utils.solid_mechanics import compute_small_strains_from_green


# object definition

class RVE(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, dims, center):
        self.name = name
        self.dims = dims
        self.center = center
        # mesh definitions
        # TODO: base it on characteristic length
        self.mesh_size = .02
        self.mesh_tol = 1e-5
        self.mesh_trial_iter = 1
        self.mesh_refine_factor = 1.25
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4
        # variable initialization
        self.part = None
        self.particles = []
        self.bounds = []
        # auxiliar variables
        self.var_coord_map = OrderedDict([('X', 0), ('Y', 1), ('Z', 2)])
        self.sign_bound_map = {'-': 0, '+': 1}
        # initial operations
        self._compute_bounds()

    def _compute_bounds(self):
        for dim, c in zip(self.dims, self.center):
            self.bounds.append([c - dim / 2, c + dim / 2])

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_particle(self, particle):
        self.particles.append(particle)

    def _create_RVE_sketch(self, model):

        # rectangle points
        pts = []
        for x1, x2 in zip(self.bounds[0], self.bounds[1]):
            pts.append([x1, x2])

        # sketch
        sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                         sheetSize=2 * self.dims[0])
        sketch.rectangle(point1=pts[0], point2=pts[1])

        return sketch

    def _create_bounds_sets(self):

        # vertices
        self._create_bound_obj_sets('vertices', self.vertex_positions, self._get_vertex_name)

        # edges
        self._create_bound_obj_sets('edges', self.unnest(self.edge_positions), self._get_edge_name)

        # faces
        if len(self.dims) > 2:
            self._create_bound_obj_sets('faces', self.unnest(self.face_positions), self._get_face_name)

    def _create_bound_obj_sets(self, obj, positions, get_name):
        '''
        Parameter
        ---------
        obj : str
            Possible values are 'vertices', 'edges', 'faces'
        '''

        # initialization
        get_objs = getattr(self.part, obj)

        # create sets
        for pos in positions:
            name = get_name(pos)
            kwargs = {}
            for i in range(0, len(pos), 2):
                pos_ = pos[i:i + 2]
                var_name = self._get_bound_arg_name(pos_)
                kwargs[var_name] = self.get_bound(pos_)

            objs = get_objs.getByBoundingBox(**kwargs)
            kwargs = {obj: objs}
            self.part.Set(name=name, **kwargs)

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

    @staticmethod
    def _get_edge_name(position):
        return 'EDGE_{}'.format(position)

    @staticmethod
    def _get_vertex_name(position):
        return 'VERTEX_{}'.format(position)

    @staticmethod
    def _get_face_name(position):
        return 'FACE_{}'.format(position)

    @staticmethod
    def _get_ref_point_name(position):
        return 'REF_POINT_{}'.format(position)

    @staticmethod
    def _define_positions_by_recursion(ref_positions, d):
        def append_positions(obj, i):
            if i == d:
                positions.append(obj)
            else:
                for ref_obj in (ref_positions[i]):
                    obj_ = obj + ref_obj
                    append_positions(obj_, i + 1)

        positions = []
        obj = ''
        append_positions(obj, 0)

        return positions

    @staticmethod
    def unnest(array):
        unnested_array = []
        for arrays in array:
            for array_ in arrays:
                unnested_array.append(array_)

        return unnested_array

    @staticmethod
    def _get_bound_arg_name(pos):
        return '{}Min'.format(pos[0].lower()) if pos[-1] == '+' else '{}Max'.format(pos[0].lower())

    def get_bound(self, pos):
        i, p = self.var_coord_map[pos[0]], self.sign_bound_map[pos[1]]
        return self.bounds[i][p]


class RVE2D(RVE):

    def __init__(self, length, width, center, name='RVE'):
        '''
        Notes
        -----
        -1st reference point represents the difference between right bottom
        and left bottom vertices.
        -2nd reference point represents the difference between left top
        and left bottom vertices.
        '''
        super(RVE2D, self).__init__(name, dims=(length, width), center=center)
        # additional variables
        self.edge_positions = (('X-', 'X+'), ('Y-', 'Y+'))  # order is relevant
        self.ref_points_positions = ('X-X+', 'Y-Y+')  # order is relevant
        # define positions
        # TODO: group vertices? They don't require verification...
        self.vertex_positions = self._define_positions_by_recursion(
            self.edge_positions, len(self.dims))

    def create_part(self, model):
        # TODO: move to parent?

        # create RVE
        sketch = self._create_RVE_sketch(model)

        # create particles geometry in sketch
        for particle in self.particles:
            particle.create_inner_geometry(sketch, self)

        # create part
        self.part = model.Part(name=self.name, dimensionality=TWO_D_PLANAR,
                               type=DEFORMABLE_BODY)
        self.part.BaseShell(sketch=sketch)

        # TODO: create particles by partitions

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
        edge_positions = self.unnest(self.edge_positions)
        edges = [self.part.sets[self._get_edge_name(position)].edges[0] for position in edge_positions]
        self.part.seedEdgeBySize(edges=edges, size=self.mesh_size,
                                 deviationFactor=self.mesh_deviation_factor,
                                 constraint=FIXED)
        # generate mesh
        self.part.generateMesh()

    def generate_mesh_for_pbcs(self, fast=False):
        # TODO: generate error file

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
                if it <= self.mesh_trial_iter:
                    print('Warning: Failed mesh generation. Mesh size will be decreased')
                    self.mesh_size /= self.mesh_refine_factor
                else:
                    print('Warning: Failed mesh generation')

        return success

    def verify_mesh_for_pbcs(self):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''
        # TODO: create set with all error nodes

        for i, positions in enumerate(self.edge_positions):

            # perpendicular direction
            j = (i + 1) % 2

            # get sorted nodes
            nodes = [self._get_edge_nodes(pos, sort_direction=j) for pos in positions]

            # verify if tolerance is respected
            for node_m, node_p in zip(nodes[0], nodes[1]):
                if abs(node_m.coordinates[j] - node_p.coordinates[j]) > self.mesh_tol:
                    return False

        return True

    def verify_mesh_for_pbcs_quick(self):

        for positions in self.edge_positions:
            nodes = [self._get_edge_nodes(pos) for pos in positions]
            if len(nodes[0]) != len(nodes[1]):
                return False

        return True

    def apply_pbcs_constraints(self, model):

        # create reference points
        self._create_pbcs_ref_points(model)

        # get vertices
        rt_vertex, lt_vertex, lb_vertex, rb_vertex = self._get_all_vertices(only_names=True)

        # get reference points
        ref_points = self._get_all_ref_points(only_names=True)

        # apply vertex constraints
        for ndof in range(1, 3):
            # right-top and left-bottom nodes
            model.Equation(name='CONSTRAINT_X+Y+_X-Y-_V_{}'.format(ndof),
                           terms=((1.0, rt_vertex, ndof),
                                  (-1.0, lb_vertex, ndof),
                                  (-1.0, ref_points[0], ndof),
                                  (-1.0, ref_points[1], ndof)))

            # left-top and right-bottom nodesL
            model.Equation(name='CONSTRAINT_X-Y+_X+Y-_V_{}'.format(ndof),
                           terms=((1.0, rb_vertex, ndof),
                                  (-1.0, lt_vertex, ndof),
                                  (-1.0, ref_points[0], ndof),
                                  (1.0, ref_points[1], ndof)))

        # edges constraints
        for i, (positions, ref_point) in enumerate(zip(self.edge_positions, ref_points)):

            # get sorted nodes
            j = (i + 1) % 2
            nodes = [self._get_edge_nodes(pos, sort_direction=j) for pos in positions]

            for k, nodes_ in enumerate(zip(nodes[0][1:-1], nodes[1][1:-1])):

                # create set with individual nodes
                set_names = []
                for pos, node in zip(positions, nodes_):
                    set_name = 'NODE_{}_{}'.format(pos, k)
                    self.part.Set(name=set_name, nodes=MeshNodeArray((node,)))
                    set_names.append('{}.{}'.format(self.name, set_name))

                # create constraint
                for ndof in range(1, 3):
                    model.Equation(name='CONSTRAINT_{}_{}_{}_{}'.format(positions[0], positions[1], k, ndof),
                                   terms=((1.0, set_names[1], ndof),
                                          (-1.0, set_names[0], ndof),
                                          (-1.0, ref_point, ndof)))

        # ???: need to create fixed nodes? why to not apply bcs e.g. LB and RB?

    def apply_bcs_disp(self, model, step_name, epsilon_11, epsilon_22,
                       epsilon_12, green_lagrange_strain=True):
        # TODO: receive epsilon as array-like?

        # initialization
        epsilon = [epsilon_11, epsilon_12, epsilon_22]
        epsilon = symmetricize_vector(epsilon)

        # create strain matrix
        if green_lagrange_strain:
            epsilon = compute_small_strains_from_green(epsilon)

        # apply displacement
        # TODO: transform eps in displacement?
        disp_bcs = []
        for k, position in enumerate(self.ref_points_positions):
            i, j = k % 2, (k + 1) % 2
            region_name = self._get_ref_point_name(position)
            applied_disps = {'u{}'.format(i + 1): epsilon[i, i]}
            disp_bcs.append(DisplacementBC(name='TEST_{}_{},{}'.format(position, i, i),
                                           region=region_name, createStepName=step_name,
                                           **applied_disps))
            applied_disps = {'u{}'.format(j + 1): epsilon[i, j]}
            disp_bcs.append(DisplacementBC(name='TEST_{}_{},{}'.format(position, i, j),
                                           region=region_name, createStepName=step_name,
                                           **applied_disps))

        # TODO: return disp bcs and do apply outside?
        for disp_bc in disp_bcs:
            disp_bc.apply_bc(model)

        # TODO: fix left bottom node? I think Miguel does not it (even though he founds out that "support nodes" -> see line 433)

    def _create_pbcs_ref_points(self, model):
        '''
        Notes
        -----
        Any coordinate for reference points position works.
        '''

        # initialization
        modelAssembly = model.rootAssembly

        # create reference points
        coord = list(self.center) + [0.]
        for position in self.ref_points_positions:
            ref_point = modelAssembly.ReferencePoint(point=coord)
            modelAssembly.Set(name=self._get_ref_point_name(position),
                              referencePoints=((modelAssembly.referencePoints[ref_point.id],)))

    def _get_all_vertices(self, only_names=False):
        '''
        Notes
        -----
        -output is given in the order of position definition.
        '''
        # TODO: move to parent?

        if only_names:
            vertices = ['{}.{}'.format(self.name, self._get_vertex_name(position)) for position in self.vertex_positions]
        else:
            vertices = [self.part.sets[self._get_vertex_name(position)] for position in self.vertex_positions]

        return vertices

    def _get_all_ref_points(self, model=None, only_names=False):
        '''
        Notes
        -----
        Output is given in the order of position definition.
        Model is required if only_names is False.
        '''
        # TODO: move to parent?

        if only_names:
            ref_points = [self._get_ref_point_name(position) for position in self.ref_points_positions]
        else:
            ref_points = [model.rootAssembly.sets[self._get_ref_point_name(position)] for position in self.ref_points_positions]

        return ref_points

    def _get_edge_nodes(self, position, sort_direction=None):
        # TODO: move to parent?
        edge_name = self._get_edge_name(position)
        nodes = self.part.sets[edge_name].nodes
        if sort_direction is not None:
            nodes = sorted(nodes, key=lambda node: self._get_node_coordinate(node, i=sort_direction))

        return nodes


class RVE3D(RVE):

    def __init__(self, dims, name='RVE', center=(0., 0., 0.)):
        super(RVE3D, self).__init__(name, dims, center)
        # additional variables
        self.face_positions = (('X-', 'X+'), ('Y-', 'Y+'), ('Z-', 'Z+'))
        # define positions
        self.edge_positions = self._define_edge_positions()
        # TODO: group vertices?
        self.vertex_positions = self._define_positions_by_recursion(
            self.face_positions, len(self.dims))

    def _define_edge_positions(self):

        # define positions
        positions = []
        for i, posl in enumerate(self.face_positions[:2]):
            for posr in self.face_positions[i + 1:]:
                for posl_ in posl:
                    for posr_ in posr:
                        positions.append(posl_ + posr_)

        # reagroup positions
        edge_positions = []
        for perp_axis in self.var_coord_map.keys():
            grouped_edges = []
            for pos in positions:
                if perp_axis not in pos:
                    grouped_edges.append(pos)
            edge_positions.append(grouped_edges)

        return edge_positions

    def create_part(self, model):

        # create RVE
        sketch = self._create_RVE_sketch(model)

        # create particles geometry in sketch
        for particle in self.particles:
            particle.create_inner_geometry(sketch, self)

        # create particles parts
        for particle in self.particles:
            particle.create_part(model, self)

        # create part
        part_name = '{}_TMP'.format(self.name) if len(model.parts) > 0 else self.name
        self.part = model.Part(name=part_name, dimensionality=THREE_D,
                               type=DEFORMABLE_BODY)
        self.part.BaseSolidExtrude(sketch=sketch, depth=self.dims[2])

        # create part by merge instances
        if len(model.parts) > 1:
            self._create_part_by_merge(model)

        # create PBCs sets (here because sets are required for meshing purposes)
        self._create_bounds_sets()

    def _create_part_by_merge(self, model):
        # TODO: make with voids to extend method (base it on the material); make also combined

        # initialization
        modelAssembly = model.rootAssembly

        # create rve instance
        modelAssembly.Instance(name='{}_TMP'.format(self.name),
                               part=self.part, dependent=ON)

        # create particle instances
        for particle in self.particles:
            particle.create_instance(model)

        # create merged rve
        modelAssembly.InstanceFromBooleanMerge(name=self.name,
                                               instances=modelAssembly.instances.values(),
                                               keepIntersections=ON,
                                               originalInstances=DELETE,
                                               domain=GEOMETRY)
        self.part = model.parts[self.name]  # override part

    def create_instance(self, model):

        # initialization
        modelAssembly = model.rootAssembly

        # verify if already created (e.g. during _create_part_by_merge)
        if len(modelAssembly.instances.keys()) > 0:
            return

        # create assembly
        modelAssembly.Instance(name=self.name, part=self.part, dependent=ON)

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
        # TODO: need to be tested with the use of bounds

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
        # TODO: simplify
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
        # TODO: create set with all error nodes

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
        # TODO: recode
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
