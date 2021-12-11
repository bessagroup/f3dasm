################################################################################
#       CLASS DEFINITIONS OF GEOMETRIC OBJECTS USED WITHIN THE RVE CLASS       #
################################################################################
# Within this file, classes for different geometric objects used in the GmshModel
# are defined. The geometric objects are used within the geometry generation to
# create Gmsh model from basic geometry entities.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations

################################################################################
#                       Define generic GeometricObject                         #
################################################################################
# This class provides general properties that all geometrical objects within the
# RVE will have. Its child-classes inherit those properties and provide additional,
# more specific properties.
class GeometricObject:
    """Definition of a generic geometric object

    This class is the parent class of all geometric objects and provides general
    information that all objects should have

    Attributes:
    -----------
    dimension: int
        dimension of the RVE

    group: string
        string defining which group the geometric object belongs to
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,dimension=None,group="default"):
        """Initialization method for geometric objects

        Parameters:
        ----------
        dimension: int
            dimension of the geometric object

        group: string
            group the geometric object belongs to
        """
        # error checking for dimension
        if dimension is None:
            raise TypeError("Variable \"dimension\" not defined. For a geometric object of type {}, the dimension must be specified. Check your input data.".format(self.__class__))

        # initialize relevant variables
        self.dimension=dimension                                                # dimension of the geometrical object
        self.group=group                                                        # group tag of the geometrical object (to distinguish different groups of objects)


    ################################################################
    # Method to add the geometric object to a specified Gmsh model #
    ################################################################
    def addToGmshModel(self,gmshModel):
        """Add a representation of the geometric object to the Gmsh model

        Parameters:
        -----------
        gmshModel: class object
            model class of the Gmsh-Python-API that holds the Gmsh-specific
            current model information
        """
        return (self.dimension,self._getGmshRepresentation(gmshModel))


    ############################################################################
    # Internal placeholder method to determine the objects Gmsh representation #
    ############################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Placeholder method to be specified by child classes in order to return
        the correct Gmsh representation of the geometric object under investigation
        """
        pass



################################################################################
#                Define Box as a child of GeometricObject                      #
################################################################################
# This class provides more specified attributes and methods for geometrical
# objects of type "Box". It inherits basic properties from its parent class
# "GeometricObject".
class Box(GeometricObject):
    """Definition of a Box object

    This class is a child class of geometricObject and provides additional
    information for objetcs of type Box

    Attributes:
    -----------
    dimension: int
        dimension of the box object
    origin: array/list
        origin of the domain object -> origin=[Ox, Oy, (Oz)]
    size: array/list
        size of the domain object -> size=[Lx, Ly, (Lz)]
    group: string
        group the box object belongs to
    """


    ##########################
    # Initialization method  #
    ##########################
    def __init__(self,size=None,origin=[0,0,0],group="default"):
        """Initialization method for Box objects

        Parameters:
        -----------
        size: array/list
            size of the domain object -> size=[Lx, Ly, Lz]
        origin: array/list
            origin of the domain object -> origin=[Ox, Oy, Oz]
        group: string
            group the box object belongs to
        """

        # initialize parent classes attributes and methods
        super().__init__(dimension=3,group=group)

        # plausibility checks for input variables:
        for varName, varValue in {"size": size, "origin": origin}.items():
            if varValue is None:                                                # check if variable has a value
                raise TypeError("Variable \"{0}\" not set! For a geometric object of type \"{1}\", the {0} must be specified. Check your input data.".format(varName,self.__class__))
            elif len(np.shape(varValue)) > 1:                                   # check for right amount of array dimensions
                raise ValueError("Wrong amount of array dimensions for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" can only be one-dimensional. Check your input data.".format(varName,self.__class__))
            elif len(varValue) != 3:                                            # check for right amount of values
                raise ValueError("Wrong number of (non-zero) values for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" has to have 3 values. Check your input data.".format(varName,self.__class__))
            elif varName is "size" and np.count_nonzero(varValue) != 3:         # check for right amount of non-zero values (size only)
                raise ValueError("Wrong number of non-zero values for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" has to have 3 non-zero values. Check your input data.".format(varName,self.__class__))

        # set object attributes
        self.origin=np.asarray(origin)                                          # set origin of the domain object
        self.size=np.asarray(size)                                              # set size of the domain object


    ################################################################
    # Internal method to determine the objects Gmsh representation #
    ################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Return a Gmsh OCC entity of type Box"""
        return gmshModel.occ.addBox(*np.r_[self.origin, self.size])



################################################################################
#               Define Rectangle as a child of GeometricObject                 #
################################################################################
# This class provides more specified attributes and methods for geometrical
# objects of type "Rectangle". It inherits basic properties from its parent
# class "GeometricObject".
class Rectangle(GeometricObject):
    """Definition of a Rectangle object

    This class is a child class of geometricObject and provides additional
    information for objetcs of type Rectangle

    Attributes:
    -----------
    dimension: int
        dimension of the box object
    origin: array/list
        origin of the domain object -> origin=[Ox, Oy, (Oz)]
    size: array/list
        size of the domain object -> size=[Lx, Ly, (Lz)]
    group: string
        group the box object belongs to
    """
    ##########################
    # Initialization method  #
    ##########################
    def __init__(self,size=None,origin=[0,0,0],group="default"):
        """Initialization method for Rectangle objects

        Parameters:
        -----------
        size: array/list
            size of the domain object -> size=[Lx, Ly, (Lz)]
        origin: array/list
            origin of the domain object -> origin=[Ox, Oy, (Oz)]
        group: string
            group the domain object belongs to
        """

        # initialize parent classes attributes and methods
        super().__init__(dimension=2,group=group)

        # plausibility checks for input variables:
        for varName, varValue in {"size": size, "origin": origin}.items():
            if varValue is None:                                                # check if variable has a value
                raise TypeError("Variable \"{0}\" not set! For a geometric object of type \"{1}\", the {0} must be specified. Check your input data.".format(varName,self.__class__))
            elif len(np.shape(varValue)) > 1:                                   # check for right amount of array dimensions
                raise ValueError("Wrong amount of array dimensions for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" can only be one-dimensional. Check your input data.".format(varName,self.__class__))
            elif not len(varValue) in [2,3]:                                    # check for right amount of values
                raise ValueError("Wrong number of (non-zero) values for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" has to have 2 or 3 values. Check your input data.".format(varName,self.__class__))
            elif varName is "size" and np.count_nonzero(varValue) != 2:         # check for right amount of non-zero values (size only)
                raise ValueError("Wrong number of non-zero values for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" has to have 2 non-zero values. Check your input data.".format(varName,self.__class__))

        # Correct potentially two-dimensional arrays
        if len(size) != 3:                                                      # check if size is not a three-dimensional array
            size=np.r_[size,0]                                                  # -> append 0
        if len(origin) != 3:                                                    # check if origin is not a three-dimensional array
            newOrigin=np.zeros(3)                                               # -> create new three-dimensional array
            newOrigin[size != 0]=origin                                         # -> assign values of origin to non-zero dimensions of new array
            origin=newOrigin                                                    # -> overwrite origin with new array

        # set object attributes
        self.origin=np.asarray(origin)                                          # set origin of the domain object
        self.size=np.asarray(size)                                              # set size of the domain object


    ################################################################
    # Internal method to determine the objects Gmsh representation #
    ################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Return a Gmsh entity of type Rectangle"""
        return gmshModel.occ.addRectangle(*np.r_[self.origin, self.size[0:self.dimension]])



################################################################################
#                 Define Sphere as child of GeometricObject                    #
################################################################################
# This class provides more specified attributes and methods for geometrical
# objects of type "Sphere". It inherits basic properties from its parent class
# "GeometricObject".
class Sphere(GeometricObject):
    """Definition of a Sphere object

    This class is a child class of geometricObject and provides additional
    information for objetcs of type Sphere

    Attributes:
    -----------
    center: array/list
        array that defines the center of the Sphere object
        -> center=[Cx, Cy, (Cz)]
    radius: float
        radius of the Sphere object
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,center=None,radius=None,group="default"):
        """Initialization method for Sphere objects

        Parameters:
        -----------
        center: array/list
            center of the Sphere object -> center=[Cx, Cy, Cz]
        radius: float
            radius of the Sphere object
        group: string
            group the Sphere object belongs to
        """

        # initialize parent classes attributes and methods
        super().__init__(dimension=3,group=group)

        # error checking
        if radius is None:                                                      # no radius defined -> error
            raise TypeError("Variable \"radius\" not set! For a geometric object of type \"{}\", the radius must be specified. Check your input data.".format(self.__class__))
        if center is None:                                                      # no center defined -> error
            raise TypeError("Variable \"center\" not set! For a geometric object of type \"{}\", the center must be specified. Check your input data.".format(self.__class__))
        elif len(np.shape(center)) > 1:                                         # check if center has the right number of array dimensions
            raise ValueError("Wrong amount of array dimensions for variable \"center\"! For a geometric object of type \"{}\", the variable center can only be one-dimensional. Check your input data.".format(self.__class__))
        elif len(center) != 3:                                                  # check if center has 3 values
            raise ValueError("Wrong number of values for variable \"center\"! For a geometric object of type \"{}\", the variable \"center\" has to have 3 values. Check your input data.".format(self.__class__))

        # set object attributes
        self.radius=radius
        self.center=np.asarray(center)


    ################################################################
    # Internal method to determine the objects Gmsh representation #
    ################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Return a Gmsh entity of type Sphere"""
        return gmshModel.occ.addSphere(*np.r_[self.center, self.radius])



################################################################################
#               Define Cylinder as child of GeometricObject                    #
################################################################################
# This class provides more specified attributes and methods for geometrical
# objects of type "Cylinder". It inherits basic properties from its parent class
# "GeometricObject".
class Cylinder(GeometricObject):
    """Definition of a Cylinder object

    This class is a child class of geometricObject and provides additional
    information for objetcs of type Cylinder

    Attributes:
    -----------
    center: array/list
        array that defines the center of the Cylinder object
        -> center=[Cx, Cy, Cz]
    axis: array/list
        array that defines the axis (direction and length) of the Cylinder object
        -> axis=[Ax, Ay, Az]
    radius: float
        radius of the Cylinder object
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,center=None,radius=None,axis=None,group="default"):
        """Initialization method for Cylinder objects

        Parameters:
        -----------
        center: array/list
            center of the Cylinder object -> center=[Cx, Cy, Cz]
        radius: float
            radius of the Cylinder object
        group: string
            group the Cylinder object belongs to
        """

        # initialize parent classes attributes and methods
        super().__init__(dimension=3,group=group)

        # error checking
        if radius is None:                                                      # no radius defined -> error
            raise TypeError("Variable \"radius\" not set! For a geometric object of type \"{}\", the radius must be specified. Check your input data.".format(self.__class__))
        for varName, varValue in {"center": center, "axis": axis}.items():
            if varValue is None:                                                # check if variable is defined
                raise TypeError("Variable \"{0}\" not set! For a geometric object of type \"{1}\", the variable {0} must be specified. Check your input data.".format(varName,self.__class__))
            elif len(np.shape(varValue)) > 1:                                   # check for correct amount of array dimensions
                raise ValueError("Wrong amount of array dimensions for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" can only be one-dimensional. Check your input data.".format(varName,self.__class__))
            elif len(varValue) != 3:                                            # check for correct amount of variable values
                raise ValueError("Wrong number of values for variable \"{0}\"! For a geometric object of type \"{1}\", the variable \"{0}\" has to have 3 values. Check your input data.".format(center,self.__class__))

        # set object attributes
        self.radius=radius
        self.center=np.asarray(center)
        self.axis=np.asarray(axis)


    ################################################################
    # Internal method to determine the objects Gmsh representation #
    ################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Return a Gmsh entity of type Cylinder"""
        return gmshModel.occ.addCylinder(*np.r_[self.center, self.axis, self.radius])



################################################################################
#                  Define Circle as child of GeometricObject                   #
################################################################################
# This class provides more specified attributes and methods for geometrical
# objects of type "Sphere". It inherits basic properties from its parent class
# "GeometricObject".
class Circle(GeometricObject):
    """Definition of an Circle object

    This class is a child class of geometricObject and provides additional
    information for objetcs of type Circle

    Attributes:
    -----------
    center: array/list
        array that defines the center of the Circle object
        -> center=[Cx, Cy, (Cz)]
    radius: float
        radius of the Circle object
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,center=None,radius=None,group="default"):
        """Initialization method for Circle objects

        Parameters:
        -----------
        center: array/list
            center of the Circle object
            -> center=[Cx, Cy, (Cz)]
        radius: float
            radius of the Circle object
        group: string
            group the Circle object belongs to
        """

        # initialize parent classes attributes and methods
        super().__init__(dimension=2,group=group)

        # error checking
        if radius is None:                                                      # radius not defined -> error
            raise TypeError("Variable \"radius\" not set! For a geometric object of type \"{}\", the radius must be specified. Check your input data.".format(self.__class__))
        if center is None:                                                      # no center defined -> error
            raise TypeError("Variable \"center\" not set! For a geometric object of type \"{}\", the center must be specified. Check your input data.".format(self.__class__))
        elif len(np.shape(center)) > 1:                                         # check for correct amount of array dimensions
            raise ValueError("Wrong amount of array dimensions for variable \"center\"! For a geometric object of type \"{}\", the variable \"center\" can only be one-dimensional. Check your input data.".format(self.__class__))
        elif not len(center) in [2,3]:                                          # check for correct amount of variable values
            raise ValueError("Wrong number of values for variable \"center\"! For a geometric object of type \"{}\", the center has to have 2 or 3 values. Check your input data.".format(self.__class__))

        # Correct potentially two-dimensional arrays
        if len(center) != 3:                                                    # check if center is not a three-dimensional array
            center=np.r_[center,0]                                               # -> append 0

        # set object attributes
        self.radius=radius
        self.center=np.asarray(center)


    ################################################################
    # Internal method to determine the objects Gmsh representation #
    ################################################################
    def _getGmshRepresentation(self,gmshModel):
        """Return a Gmsh entity of type Disk"""
        return gmshModel.occ.addDisk(*np.r_[self.center, self.radius, self.radius])
