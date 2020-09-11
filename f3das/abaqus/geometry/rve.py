'''
Created on 2020-03-24 14:33:48
Last modified on 2020-09-11 17:00:17
Python 2.7.16
v0.1

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


#%% imports

# abaqus
from abaqusConstants import (TWO_D_PLANAR, DEFORMABLE_BODY, ON, FIXED)
from mesh import MeshNodeArray

# third-party
import numpy as np

# local library
from .utils import transform_point
from .utils import get_orientations_360
from ...utils.linalg import symmetricize_vector
from ...utils.linalg import sqrtm


#%% 2d RVE

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
        sucess = False
        while it <= self.mesh_trial_iter and not sucess:

            # generate mesh
            self.generate_mesh()

            # verify generated mesh
            if fast:
                sucess = self.verify_mesh_for_pbcs_quick()
            else:
                sucess = self.verify_mesh_for_pbcs()

            # prepare next iteration
            it += 1
            if not sucess:
                self.mesh_size /= self.mesh_refine_factor

        return sucess

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


class BertoldiExampleRVE(RVE2D):

    def __init__(self, length, width, center, r_0, c_1, c_2,
                 n_points=100, name='BERTOLDI_RVE'):

        # instantiate parent class
        RVE2D.__init__(self, length, width, center, name=name)

        # store vars
        self.r_0 = r_0
        self.c_1 = c_1
        self.c_2 = c_2
        self.n_points = n_points

    def _create_inner_geometry(self, model):
        '''
        Creates particular geometry of this example, i.e. internal pores.
        '''

        # initialization
        thetas = np.linspace(0., 2 * np.pi, self.n_points)

        # get points for spline
        points = []
        for theta in thetas:
            rr = self.r_0 * (1. + self.c_1 * np.cos(4. * theta) + self.c_2 * np.cos(8. * theta))
            points.append(transform_point((rr * np.cos(theta), rr * np.sin(theta)),
                                          origin_translation=self.center))

        # generate spline
        self.sketch.Spline(points=points)
