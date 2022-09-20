#!/usr/bin/env python
# coding: utf-8

"""
Python module to analyze and plot neurons from neuprint with igraph.
""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import igraph as ig

from neuprint import Client
from neuprint import fetch_neurons, NeuronCriteria as NC, merge_neuron_properties
from neuprint import fetch_skeleton, fetch_synapses, fetch_synapse_connections, attach_synapses_to_skeleton, skeleton
from neuprint.utils import connection_table_to_matrix


def createPanel(x, y, width, height, figureWidth, figureHeight):
    """
    Creates a panel
    width, and height are normalized to the figure's width and height
    """
    return plt.axes([x, y ,width/figureWidth, height/figureHeight])


def reorderColumns(df, startCols):
    """
    Given a dataframe and list of columns
    Return the dataframe with columns reordered so that given columns are at the start
    """
    endCols = []
    for item in df.columns:
        if item not in startCols:
            endCols.append(item)
    return(df[startCols + endCols])


def dist(point1, point2):
    """
    Given 2 points (list of xyz coords),
    Return the distance between them
    """
    if (len(point1) != 3 or len(point1) != len(point2)):
        raise TypeError('Points must be lists of length 3')
    total = 0
    for i in range(len(point1)):
        total += (point1[i] - point2[i])**2
    return np.sqrt(total)


def neuprint_to_um(num):
    """
    Given a number from neuprint, return it in micrometers
    Assumes 1 unit from neuprint is 8 nanometers
    """
    return num * 8 / 1000


def addEdgeLength(g, edge):
    """
    Given an edge, get its vertices
    and use their coordinates to calculate the length of the edge.
    Then assign the length as an attribute to the edge
    """
    v_in = g.vs[edge.source]
    v_out = g.vs[edge.target]
    length = dist([v_in['x'], v_in['y'], v_in['z']],
                  [v_out['x'], v_out['y'], v_out['z']])
    edge['length'] = length


def vertexDist(g, v1, v2):
    """ Given 2 vertices, return the distance of the shortest path between them """
    try:
        results = g.get_shortest_paths(
            v1,
            to=v2,
            weights="length",
            output="epath",
            )
    except InternalError as e:
        # Invalid vertex ID
        print(f'v1: {v2} and v2: {v2} \nFailed with {e}')
        return None
    except Exception as e:
        print(f'v1: {v2} and v2: {v2} \nFailed with {e}')
        return None
        
    total = 0
    # 1 path since its a undirected acyclic graph
    for edge in results[0]:
        total += g.es[edge]['length']
    return total


def colorPath(g, v1, v2, skeleton, color="orange"):
    """
    Given a graph, 2 vertices and a skeleton,
    Get the path between the vertices and color those edges in the skeleton
    """
    results = g.get_shortest_paths(
        v1,
        to=v2,
        weights="length",
        output="epath",
        )
    for edge in results[0]:
        skeleton.loc[skeleton['rowId'] == edge, 'edge_color'] = color


def initialize_dataframe(bodyId=1140245595):
    my_neuprint_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRhbi50dXJuZXIuZXZhbnNAZ21haWwuY29tIiwibGV2ZWwiOiJyZWFkd3JpdGUiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS0vQU9oMTRHZ29ac3BRcUotbmtaRVhHSl9FNEZpZXBrRkdiSWl2LXRzdk90ako9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MDMxODg0NjV9.-TY4e9CA30ON2j1FMXUnmkcSGNATc8e5v7jzEduF7zA'
    c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=my_neuprint_token)

    # Define your neuron of interest and fetch its skeleton
    skeleton = fetch_skeleton(bodyId, heal=True)

    ### Fetch the synapses onto your neuron
    # Get the synapse locations
    synapses = fetch_synapses(bodyId)

    # Select only the postsynaptic sites
    synapses_post = synapses[synapses['type'] == 'post']

    # Get the synaptic connections (this includes bodyId information about the partners)
    synapse_conns = fetch_synapse_connections(None,bodyId)

    # Find the type of each neuron
    synapse_conns['type_pre'] = fetch_neurons(synapse_conns['bodyId_pre'])[0]['type']

    # Get the bodyIds for each of the postsynaptic sites that will be added to the skeleton
    bodyIds_pre = [synapse_conns['bodyId_pre'][(synapse_conns['x_post'] == x) & 
                                                (synapse_conns['y_post'] == y) &
                                                (synapse_conns['z_post'] == z)].values for 
                    i,[x,y,z] in synapses_post[['x','y','z']].iterrows()]

    # Remove the synapses that are not in the synaptic connections object
    synapses_post = synapses_post.iloc[[i for i, val in enumerate(bodyIds_pre) if len(val) > 0]].reset_index()

    # Attach the synapses to the skeleton
    skeleton_w_syns = attach_synapses_to_skeleton(skeleton, synapses_post)

    # Add bodyIds to skeleton synapses
    synapses_post['bodyId_pre'] = [val[0] for i, val in enumerate(bodyIds_pre) if len(val) > 0]

    # Make a dataframe of all of the neuron types that synapse onto the DNa02s for each bodyId
    pre_neurons = fetch_neurons(list(set(synapses_post['bodyId_pre'])))[0]

    # Assign a type to each bodyId
    synapses_post['type_pre'] = [pre_neurons['type'][pre_neurons['bodyId'] == bId].values[0] for bId in synapses_post['bodyId_pre']]
    synapses_post['instance_pre'] = [pre_neurons['instance'][pre_neurons['bodyId'] == bId].values[0] for bId in synapses_post['bodyId_pre']]

    # Define the skeleton properties for plotting
    skeleton_w_syns['type_pre'] = ['Node']*len(skeleton) + synapses_post['type_pre'].tolist()
    skeleton_w_syns['instance_pre'] = ['Node']*len(skeleton) + synapses_post['instance_pre'].tolist()
    skeleton_w_syns['vertex_color'] = 'black'
    skeleton_w_syns['edge_color'] = 'black'
    skeleton_w_syns['syn_size'] = 0.01

    # Convert units from neuprint voxels (8 nm) to micrometers
    for col in ['x', 'y', 'z', 'radius']:
        skeleton_w_syns[col] = skeleton_w_syns[col].apply(neuprint_to_um)

    return skeleton_w_syns



def createGraph(skeleton_w_syns):
    """Given a skeleton_w_syns dataframe, return an igraph graph."""
    skeleton_edges = skeleton_w_syns[['rowId','link']].iloc[1:] # Ignore first row because the root has no link
    g = ig.Graph.DataFrame(edges = skeleton_edges, directed=False, vertices = skeleton_w_syns)

    # Add edge length
    for edge in g.es:
        addEdgeLength(g, edge)

    return g


def df_syn_dist(g, row, outputSynIdx):
    """
    Given a row of a skeleton with synapses dataframe,
    If the row is part of the skeleton, return None
    If the row is a synapse, return the distance from that synapse to the output synapse
    """
    if row['structure'] == 'neurite':
        return None
    else:
        return vertexDist(g, row['rowId'], outputSynIdx)
        # return None


def calculateSynDistance(g, skeleton_w_syns, outputsyn):
    """
    Given a skeleton_w_syns and an output synapse, 
    add a distance column to the dataframe with the distance from that synapse to the output synapse.
    """
    lastRowId = skeleton_w_syns['rowId'].iat[-1]
    temp_skel_w_syns = skeleton_w_syns.loc[skeleton_w_syns['rowId'] < lastRowId]
    skeleton_w_syns['distance_um'] = temp_skel_w_syns.apply(lambda row: df_syn_dist(g, row, outputsyn), axis=1)


def plotDendrogram(g, rootVertex, skeleton_w_syns, panel, filename):
    """Plot a dendrogram of the graph g at the rootVertex given and save the output as the filename."""
    layout = g.layout_reingold_tilford(root=[rootVertex])
    ig.plot(g, target=panel,
            layout=layout,
            vertex_size = skeleton_w_syns['syn_size'],
            vertex_color = list(skeleton_w_syns['vertex_color']),
            edge_color = 'black',
            )

    plt.savefig(filename)