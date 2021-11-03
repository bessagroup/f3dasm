from dolfin import *
import numpy as np
from ufl import algorithms
import copy as cp

class PeriodicBoundary(SubDomain):
    """
        GENERAL PERIODIC BOUNDARY CONDITIONS IMPLEMENTATION

                              #-----------# 
                             / |        / |
                            /  |       /  |
                           #----------#   |
                           *   |      |   |
                           *   #----------# 
                     [1]   *  *       |  / 
                           * * [2]    | / 
                           **         |/  
                           ***********# 
                               [0]

            *   : Master edges/nodes
            -   : Slave edges/nodes 
            [i] : directions for periodicity

   """

    def __init__(self,domain,periodicity=None,tolerance=DOLFIN_EPS):

        """ Initialize """

        SubDomain.__init__(self,tolerance)              # Initialize the base-class (Note: the tolerance is needed for the mapping method)

        ################################
        # Get the extrema of the domain for every direction
        ################################
        self.mins = np.array(domain.bounds[0])          
        self.maxs = np.array(domain.bounds[1])

        self.directions = np.flatnonzero(self.maxs - self.mins)     # Mark non-zero directions
        
        ################################
        # Definie periodic directions
        ################################
        if periodicity is None:
            self.periodic_dir = self.directions                 

        else:
            self.periodic_dir = periodicity

        self.tol = tolerance 
        self.master = []                                # Master nodes 
        self.map_master = []                            # Mapped master nodes
        self.map_slave = []                             # Mapped slave nodes

    def inside(self,x,on_boundary):
        ################################
        # Mark the master nodes as True 
        ################################
        x_master=False                                                   
        if on_boundary:                                                         
            for axis in self.periodic_dir:                                      
                if near(x[axis],self.maxs[axis],self.tol):
                    x_master=False                                       
                    break
                elif near(x[axis],self.mins[axis],self.tol):
                    x_master=True                                        

        if x_master:
            self.master.append(cp.deepcopy(x))

        return x_master

    # Overwrite map method of SubDomain class to define the mapping master -> slave
    def map(self,x,y):
        ################################
        # Map the master nodes to slaves
        ################################
        x_slave=False                                       
        for axis in self.directions:                               
            if axis in self.periodic_dir:                          
                if near(x[axis],self.maxs[axis],self.tol):
                    y[axis]=self.mins[axis]                        
                    x_slave=True                            
                else:                                              
                    y[axis]=x[axis]                                
            else:                                                  
                y[axis]=x[axis]                                    

        if x_slave:
            self.map_master.append(cp.deepcopy(y))                            # add y to list mapped master coordinates
            self.map_slave.append(cp.deepcopy(x))                             # add x to list of mapped slave coordinates


#class PeriodicBoundary2D(SubDomain):
#    """
#    Periodic Boundary Conditions implimentation. Given the
#    vertices of the RVE it maps left boundary to the right
#    boundary and bottom boundary to the top boundary.
#
#                          (top)
#                        #-------#
#                        ^       |
#                (left)  |  RVE  |  (right)
#                 [v1]   |       |
#                        *------>#
#                         (bottom)
#                           [v2]
#
#     NOTE: Your RVE should have 0.0 coordinates at the * vertex!
#     TO DO: Generalize!
#    """ 
#
#    def __init__(self, domain, tolerance=DOLFIN_EPS):
#        SubDomain.__init__(self, tolerance)
#        self.tol = tolerance
#        self.v = domain.verticies
#        self.v1 = self.v[1,:]-self.v[0,:] # first vector generating periodicity
#        self.v2 = self.v[3,:]-self.v[0,:] # second vector generating periodicity
#        # check if UC vertices form indeed a parallelogram
#        assert np.linalg.norm(self.v[2, :]-self.v[3, :] - self.v1) <= self.tol
#        assert np.linalg.norm(self.v[2, :]-self.v[1, :] - self.v2) <= self.tol
#        
#    def inside(self, x, on_boundary):
#        # return True if on left or bottom boundary AND NOT on one of the 
#        # bottom-right or top-left vertices
#        return bool((near(x[0], self.v[0,0] + x[1]*self.v2[0]/self.v[3,1], self.tol) or 
#                     near(x[1], self.v[0,1] + x[0]*self.v1[1]/self.v[1,0], self.tol)) and 
#                     (not ((near(x[0], self.v[1,0], self.tol) and near(x[1], self.v[1,1], self.tol)) or 
#                     (near(x[0], self.v[3,0], self.tol) and near(x[1], self.v[3,1], self.tol)))) and on_boundary)
#
#    def map(self, x, y):
#        if near(x[0], self.v[2,0], self.tol) and near(x[1], self.v[2,1], self.tol): # if on top-right corner
#            y[0] = x[0] - (self.v1[0]+self.v2[0])
#            y[1] = x[1] - (self.v1[1]+self.v2[1])
#        elif near(x[0], self.v[1,0] + x[1]*self.v2[0]/self.v[2,1], self.tol): # if on right boundary
#            y[0] = x[0] - self.v1[0]
#            y[1] = x[1] - self.v1[1]
#        else:   # should be on top boundary
#            y[0] = x[0] - self.v2[0]
#            y[1] = x[1] - self.v2[1]



