'''
Created on 2020-04-22 11:34:59
Last modified on 2020-04-22 11:36:14
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Develop functions to get node and element information from odb files.
'''


#%% function definition

def get_nodes_given_set_names(odb, set_names):
    '''
    Gets array with Abaqus nodes objects given set names.

    Parameters
    ----------
    odb : Abaqus odb object
    set_names : array-like of str

    Returns
    -------
    nodes : array-like of OdbMeshNode objects

    Notes
    -----
    -verifies if nodes are repeated in the different sets.
    '''

    # get all nodes
    nodes_with_reps = []
    for set_name in set_names:
        nodes_with_reps.extend(odb.rootAssembly.nodeSets[set_name].nodes[0])

    # get unique nodes
    if len(set_names) > 1:
        nodes = get_unique_nodes(nodes_with_reps)
    else:
        nodes = nodes_with_reps

    return nodes


def get_unique_nodes(nodes_with_reps):
    '''
    Gets unique nodes in an array of Abaqus OdbMeshNode objects.

    Parameters
    ----------
    nodes_with_reps : array-like of OdbMeshNode objects

    Notes
    -----
    -set data structure does not work with OdbMeshNode objects
    '''

    nodes = []
    for node in nodes_with_reps:
        if node not in nodes:
            nodes.append(node)

    return nodes
