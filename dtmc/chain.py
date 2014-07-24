import numpy
import networkx as nx

class DTMCError(Exception):
    """Serious DTMC Error"""

def is_square(P):
    return P.shape[0] == P.shape[1]

def is_stochastic(P):
    """True if a matrix is stochastic 
   
    All entries must be non-negative and rows must sum to one
    """
    if not numpy.all(P >= 0):
        return False

    # potentially not a good test
    return numpy.count_nonzero(numpy.sum(P, axis = 1) - 1.0) == 0

def is_substochastic(P):
    """True if a matrix is stochastic 

    All entries must be non-negative and rows must sum to less than one
    """
    if not numpy.all(P >= 0):
        return False
    return numpy.all(numpy.sum(P, axis = 1) <= 1)

def vertex_out_component(G, nodes):
    """Find the set of reachable nodes on a graph
    """
    s = set(nodes)
    for n in nodes:
        reachable = nx.shortest_path_length(G, n).keys()
        s = s.union(reachable)
    return s

def transient_class(G, cpt):
    """Find all the components, such that they can escape
    """
    return vertex_out_component(G, cpt) - set(cpt) != set() 

def recurrent_class(G, cpt):
    """Find all the components, such that they will never escape
    """
    return vertex_out_component(G, cpt) - set(cpt) == set() 


class DiscreteTimeMarkovChain(object):
    """A discrete time Markov chain
    """

    def __init__(self, transition_matrix):
        """Initialise a discrete time Markov Chain 

        Parameters
        ----------
        transition_matrix : matrix-like
            Data that describes the transitions between states in the
            chain. The rows of the transition matrix need to sum to one
        """
        # TODO - none of the work precludes the use of sparse 
        # representations, but as a first guess, this is going to
        # simplify the code signficantly
        self._P = numpy.asmatrix( transition_matrix )
        if not is_square(self._P):
            raise DTMCError('Non-square transition matrix')

        if not is_stochastic(self._P):
            raise DTMCError('Transition Matrix is not right stochastic')

    def _as_digraph(self):
        return nx.DiGraph(self._P)

    def _as_dense(self):
        return self._P

    def _num_states(self):
        return self._P.shape[0]

    def irreducible(self):
        G = self._as_digraph()
        return nx.number_strongly_connected_components(G) == 1

    def aperiodic(self):
        G = self._as_digraph()
        return nx.is_aperiodic(G)

    def stationary_distribution(self):
        # in MATLAB, this is simply A\b
        # in Python, things are a little uglier, because numpy.linalg.solve
        #   only deals with square matrices.
        P = self._P
        n = P.shape[0]
        I = numpy.eye(n)
        # the transpose here is because we're working with a stationary row
        # vector rather than a column vector
        P = P.T - I

        # we need to add an extra row of 1s at the bottom
        A = numpy.vstack([P, numpy.ones(n)])

        # everything sums to zero, except the last row, which sums to 1
        b = numpy.zeros(n+1)
        b[-1] = 1

        return numpy.linalg.lstsq(A, b)[0]

    def absorbing(self, i):
        G = self._as_digraph()
        
        return numpy.sum(self._P[i, :]) == self._P[i, i] 

    def absorbing_states(self):
        """Return a list of states that are absorbing

        A state is absorbing, if it is impossible to leave that state. 
        """
        return [ i for i in xrange(self._num_states()) if self.absorbing(i) ]

    def is_absorbing(self):
        # to be absorbing all states must eventually reach an
        # absorbing state
        G = self._as_digraph()
        for cpt in nx.weakly_connected_components(G):
            Gprime = nx.subgraph(G, cpt)
            Gprime.remove_edges_from(Gprime.selfloop_edges())
            akeys = [k for k, v in Gprime.out_degree().iteritems() if v == 0]
            if len(akeys) == 0:
                return False
        return True

    def communicating_classes(self):
        G = self._as_digraph()
        return list(nx.strongly_connected_components(G))

    def recurrent_classes(self):
        G = self._as_digraph()
        classes = self.communicating_classes()
        return [ cpt for cpt in classes if recurrent_class(G, cpt) ]

    def transient_classes(self):
        G = self._as_digraph()
        classes = self.communicating_classes()
        return [ cpt for cpt in classes if transient_class(G, cpt) ]

    def transient_states(self):
        return sum(self.transient_classes(), [])

    def recurrent_states(self):
        return sum(self.recurrent_classes(), [])

    def is_canonical(self):
        """Check if states are in canonical order

        True iff the transient states are before the recurrent states
        """
        return max(self.transient_states()) < min(self.recurrent_states())

    def canonical_permutation(self):
        if self.is_canonical():
            # attempt to be stable, ie/ do not change the
            # order of states unless you need to
            return range(self.num_states())

        lst = []
        for cls in self.transient_classes():
            lst += cls
        for cls in self.recurrent_classes():
            lst += cls
        return lst

    def permute(self, p):
        M = self._P.copy()
        M = M[p, :]
        M = M[:, p]
        return DiscreteTimeMarkovChain(M)

    def canonical_form(self):
        p = self.canonical_permutation()
        return self.permute(p), p

    def Qmatrix(self):
        alt, p = self.canonical_form()
        transient = alt.transient_states()
        Q = alt._P[transient, :]
        Q = Q[:, transient]
        return Q, p[transient]

    def fundamental_matrix(self):
        return None
