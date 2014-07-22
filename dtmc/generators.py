import networkx as nx
import numpy
import pydot

def random_walk_graph(nsteps, absorbing = False):
    G = nx.grid_2d_graph(1, nsteps)
    G = G.to_directed()
    G = nx.convert_node_labels_to_integers(G)
    n = [ i for i in G.nodes() if G.out_degree(i) == 1 ]
    M = nx.to_numpy_matrix(G)
    for i in n:
        M[i, :] = 0
        M[i, i] = 1
    return nx.DiGraph(M)


