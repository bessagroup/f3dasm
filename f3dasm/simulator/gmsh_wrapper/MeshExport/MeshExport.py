import numpy as np

def FeapExport(model):
    """
    Method providing a direct mesh export for FEAP using the mesh information
    available from the model.
    """
    # nodal transformation for connectivity needed for 2nd order tetrahedron
    # -> swap last two entries
    tet_2 = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 8]

    # prerequisites - abbreviations for easier access
    model = model.gmshAPI
    mesh = model.mesh
    meshfile = 'drve'

    # necessary information for writing feap mesh file
    dim = model.getDimension()
    print("Physical dimension:",dim)
    meshfile = str(dim)+meshfile

    # get information on nodes
    nodeTags, nodalCoord, paramCoord = mesh.getNodes(dim,-1,includeBoundary=True)
    nodeTags,indices = np.unique(nodeTags, return_index=True)
    nNodes = max(nodeTags)
    print("Number of nodes:",nNodes)

    # sort nodes and coordinates before writing
    nodalX = nodalCoord[0::3]
    nodalY = nodalCoord[1::3]
    nodalZ = nodalCoord[2::3]
    nodalX = nodalX[indices]
    nodalY = nodalY[indices]
    nodalZ = nodalZ[indices]

    nodalCoord = np.zeros(int(nNodes*3))
    nodalCoord[0::3] = nodalX
    nodalCoord[1::3] = nodalY
    nodalCoord[2::3] = nodalZ

    # get information on elements
    elemTypes,elemTags,connectivity = mesh.getElements(dim,-1)
    nElem = 0
    maxNodesPerElem = 0
    print("Element types:",len(elemTypes))
    for i in range(0, len(elemTypes)):
        nElem = nElem + len(elemTags[i])
        currNodesPerElem = len(connectivity[i])/len(elemTags[i])
        if currNodesPerElem > maxNodesPerElem:
        	maxNodesPerElem = int(currNodesPerElem)
    print("Number of elements:", nElem)
    print("Maximum number of nodes per element:",maxNodesPerElem)

    # get information on materials
    phyGroup = model.getPhysicalGroups(dim)
    numberOfMat = len(phyGroup)

    ## build database where each element has a material
    elem2mat = np.zeros((nElem,2),dtype=int)
    startind = 0
    # loop over all entities for each physical Group
    for i in range(0,numberOfMat):
        physEntities = model.getEntitiesForPhysicalGroup(dim,i+1)
        for j in range(0,len(physEntities)):
            currElemTypes,currElemTags,currConnectivity = mesh.getElements(dim,physEntities[j])
            # build array[nElem,2], where each element has number of physical group
            elem2mat[startind:startind+len(currElemTags[0]),0] = currElemTags[0]
            elem2mat[startind:startind+len(currElemTags[0]),1] = i+1
            startind = startind+len(currElemTags[0])

    # write problem type file PROB_XXX with general definitions
    with open('PROB_'+meshfile,'wt') as outfile:
    	outfile.writelines(str(nNodes)+' '+str(nElem)+' '+str(numberOfMat)+' '+str(dim)+' '+str(dim)+' '+str(maxNodesPerElem)+'\n')
    	outfile.writelines('! nodes, elements, material sets, mesh dimension, maximum nodal DOF (default = dim), max nodes/element')

    # write mesh file MESH_XXX with nodal coordinates and element connectivity for every element type
    with open('MESH_'+meshfile,'wt') as outfile:
        outfile.writelines('COORDinate\n')
        for i in range(0,nNodes):
            coordStr = np.array2string(nodalCoord[3*i:3*(i+1)],max_line_width=100000)
            outfile.writelines(str(i+1)+' 0 '+coordStr[1:-1]+'\n')

        totalElemCount = 0
        for i in range(0,len(elemTypes)):
            currNodesPerElem = int(len(connectivity[i])/len(elemTags[i]))
            elemsOfType = len(elemTags[i])
            print('-------------------------------------------------')
            print('Current nodes per element:',currNodesPerElem)
            print('Elements of current type:',elemsOfType)
            outfile.writelines('\n')
            outfile.writelines('ELEMent NODEs='+str(currNodesPerElem)+'\n')

            # due to limited input records in one line (=16), elements with more
            # than 13 nodes have to be split into several rows
            for j in range(0,elemsOfType):
                matNum = int(elem2mat[np.where(elem2mat[:,0]==elemTags[i][j]),1])
                totalElemCount = totalElemCount + 1
                outfile.writelines(str(totalElemCount)+' 0 '+str(matNum)+' ')
                if dim == 3 and currNodesPerElem == 10:
                    currConnectivity = connectivity[i][j*currNodesPerElem:(j+1)*currNodesPerElem]
                    currConnectivity = currConnectivity[tet_2]
                else:
                    currConnectivity = connectivity[i][j*currNodesPerElem:(j+1)*currNodesPerElem]
                connectStr13 = np.array2string(currConnectivity[0:min(len(currConnectivity),13)],max_line_width=100000)
                outfile.writelines(connectStr13[1:-1]+'\n')
                # writing entries 14 - x of connectivity in new lines
                # automatic linebreak every 16th entry
                for cLines in range(0,int(np.ceil(max(currNodesPerElem-13,0)/16))):
                    connectStr14x = np.array2string(currConnectivity[13+cLines*16:min(13+(cLines+1)*16,currNodesPerElem)],max_line_width=100000)
                    outfile.writelines('    '+connectStr14x[1:-1]+'\n')
