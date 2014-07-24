import networkx as nx
import numpy
import pydot

def random_walk(nsteps):
    G = nx.grid_2d_graph(1, nsteps)
    G = G.to_directed()
    G = nx.convert_node_labels_to_integers(G)
    n = [ i for i in G.nodes() if G.out_degree(i) == 1 ]
    M = nx.to_numpy_matrix(G)
    for i in n:
        M[i, :] = 0
        M[i, i] = 1
    M = M/numpy.sum(M, axis=1)
    return M

def stock_market():
    """Transition Matrix for the bull-bear-stagnant market from wikipedia
    """
    return numpy.array([ 
        [0.9,  0.075,  0.025], 
        [0.15,   0.8,   0.05], 
        [0.25,  0.25,   0.5 ] 
    ])

def simple_weather():
    """Transition Matrix for the simple weather example on wikipedia
    """
    return numpy.array([
        [0.9, 0.1], 
        [0.5, 0.5], 
    ])

# TODO - land of oz example
