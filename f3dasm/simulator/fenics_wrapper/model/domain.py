import numpy as np
import dolfin
import os 

class Domain():
    """
        GENERAL DOMAIN HANDLING 

        * This class reads the mesh file, optionally convert it to xml 
            format (cannot be paralellized) to xml format.
        * Important geometric properties for the periodic boundary 
            conditions are also obtained from the mesh.
    """

    def __init__(self, filename):

        ################################
        # Filename handling
        ################################
        self.fname = filename
        self.name, self.ext =  os.path.splitext(filename)

        ################################
        # Read domain file depending on your extension
        ################################
        if self.ext == ".msh":
            self._dolfin_convert(self.fname)
            self.__read_xml()

        elif self.ext == ".xml":
            self.__read_xml()

        elif self.ext == ".xdmf":
            #xdmf_extract(self.fname)
            self.__read_xdmf()

        self.dim = self.mesh.geometry().dim()           # Dimension of the domain
        self.ele_num = self.mesh.num_cells()            # Number of elements in the domain
        self.phases = np.unique(self.subdomains.array()).astype(int)    # Number of phases in the domain

        #self.verticies, self.vol = self.__get_vertices()
        self.bounds, self.vol = self.__get_bounds()     # Get the bounds and calculate the volume of the domain
        self.__get_volume()                             # Get volume of every element 


    def _dolfin_convert(self, filename):
        """
            Convert the .msh file with msh2 format to xml using dolfin-convert
        *** Legacy format try not to use it!
        """
        name, _ =  os.path.splitext(filename)
        os.system('dolfin-convert '+str(filename)+' '+str(name)+'.xml')

        
    def __read_xml(self):
        """

        Note:Legacy extension try not to use this method. Just here for some tests!

        """

        self.mesh = dolfin.Mesh(self.name+".xml")
        self.subdomains = dolfin.MeshFunction("size_t", self.mesh, self.name+"_physical_region.xml")
        self.facets = dolfin.MeshFunction("size_t", self.mesh, self.name+"_facet_region.xml")

    def __read_xdmf(self):
        """
            To do: name_to_read -> more specific names like subdomain and stuff!
        """

        ################################
        # Read main domain file and put it to self.mesh
        ################################
        self.mesh = dolfin.Mesh()
        with dolfin.XDMFFile(self.name+".xdmf") as infile:
            infile.read(self.mesh)

        ################################
        # Read physical region file and put it to self.subdomains
        ################################
        mvc = dolfin.MeshValueCollection("size_t", self.mesh, 3) 
        with dolfin.XDMFFile(self.name+"_physical_region.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        self.subdomains = dolfin.MeshFunction('size_t',self.mesh, mvc)

        ################################
        # Read facet region file and put it to self.facets
        ################################
        mfc = dolfin.MeshValueCollection("size_t", self.mesh, 3) 
        with dolfin.XDMFFile(self.name+"_facet_region.xdmf") as infile:
            infile.read(mfc, "name_to_read")
        self.facets = dolfin.MeshFunction("size_t", self.mesh, mfc)
        
    def __get_vertices(self):
        """
            Note: Not using it any more!

             (x_min,y_max) #-----# (x_max,y_max)
                           |     |
                           |     |
                           |     |
             (x_min,y_min) #-----# (x_max,y_min)
        """

        if self.dim == 2:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 
            
            vert =  np.array([[x_min,y_min], [x_max,y_min], \
                             [x_max,y_max], [x_min,y_max]])
            vol = x_max * y_max

        elif self.dim == 3:
            raise ("Not implimented yet!")

        return vert, vol

    def __get_bounds(self):
        """
            Method: Get the bounds of your domain
                                        (x_max,y_max,z_max)
                              #-----------# 
                             / |        / |
                            /  |       /  |
                           #----------#   |
                           |   |      |   |
                           |   #----------# 
                           |  /       |  / 
                           | /        | / 
                           |/         |/  
                           #----------# 
         (x_min,y_min,z_min)
        """
        ################################
        # Bounds for 2D domains
        ################################
        if self.dim == 2:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 
            
            vol = x_max * y_max

            bounds = np.array([[x_min, y_min],[x_max, y_max]])

        ################################
        # Bounds for 3D domains
        ################################
        elif self.dim == 3:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 

            z_min = np.min(self.mesh.coordinates()[:,2]) 
            z_max = np.max(self.mesh.coordinates()[:,2]) 
            
            vol = x_max * y_max * z_max

            bounds = np.array([[x_min, y_min, z_min],[x_max, y_max, z_max]])

        return bounds, vol



    def __get_volume(self):
        """
            Method: Get volume/area of all the elements in a numpy array
        """
        
        self.ele_vol = np.zeros(self.ele_num)
        for i in range(self.ele_num):
            cell = dolfin.Cell(self.mesh, i)
            self.ele_vol[i] = cell.volume()

