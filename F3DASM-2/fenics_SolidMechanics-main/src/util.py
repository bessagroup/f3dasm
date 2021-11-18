import os 
import meshio 
import numpy as np

##############################################################################################
# UTILITIES for preparing the domain for fenics
##############################################################################################

def dolfin_convert(filename):
    """
        Convert the .msh file with msh2 format to xml using dolfin-convert
    *** Legacy format try not to use it!
    """
    name, _ =  os.path.splitext(filename)
    os.system('dolfin-convert '+str(filename)+' '+str(name)+'.xml')

def xdmf_extract(filename):
    """
        dolfin-convert like funnction for the output from gmshModel
    *** TO DO: Extend the order of elements that can be extracted.
    
    """
    def extract(mesh,cell_type):
        cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
        data = np.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                               for key in mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
        mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells},
                                   cell_data={"name_to_read":[data]})
        return mesh


    name, _ =  os.path.splitext(filename)
    
    mesh = meshio.read(filename)
    
    dim = (np.sum(np.max(mesh.points, axis=0) - np.min(mesh.points, axis=0) > 1e-15))

    if dim == 2:
        physical = extract(mesh,"triangle")
        facet= extract(mesh,"line")
        meshio.write(name+"_physical_region.xdmf", physical)
        meshio.write(name+"_facet_region.xdmf", facet)
        mesh.remove_lower_dimensional_cells()
        mesh.prune_z_0()
        mesh = extract(mesh,'triangle')
        meshio.write(name+".xdmf", mesh)

    elif dim == 3:
        physical = extract(mesh,"tetra")
        facet= extract(mesh,"triangle")
        meshio.write(name+"_physical_region.xdmf", physical)
        meshio.write(name+"_facet_region.xdmf", facet)
        mesh = extract(mesh,'tetra')
        meshio.write(name+".xdmf", mesh)
    else:
        raise Exception("Sorry, not implimented yet!")



