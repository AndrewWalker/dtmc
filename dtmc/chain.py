import numpy
import networkx

class DTMCError(Exception):
    """Serious DTMC Error"""

def is_square(P):
    return P.shape[0] == P.shape[1]

def is_right_stochastic(P):
    return numpy.count_nonzero(numpy.sum(P, axis = 1) - 1.0) == 0

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

        if not is_right_stochastic(self._P):
            raise DTMCError('Transition Matrix is not right stochastic')

    def _as_digraph(self):
        return networkx.DiGraph(self._P)

    def _as_dense(self):
        return self._P

    def _num_states(self):
        return self._P.shape[0]

    def irreducible(self):
        G = self._as_digraph()
        return networkx.number_strongly_connected_components(G) == 1

    def aperiodic(self):
        G = self._as_digraph()
        return networkx.is_aperiodic(G)

    def stationary_distribution(self):
        # in MATLAB, this is simply A\b
        # in Python, things are a little uglier, because numpy.linalg.solve
        #   only deals with square matrices.
        
        P = self._as_dense()
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

    def absorbing(self, i, eps = 1e-10):
        # TODO - is this the right way to do this?
        return (self._P[i, i] - 1.0) < eps

    def absorbing_states(self):
        """Return a list of states that are absorbing

        A state is absorbing, if it is impossible to leave that state. 
        """
        return [ i for i in xrange(self._num_states()) if self.absorbing(i) ]

    def is_absorbing(self):
        # TODO - This is a TDD placeholder
        return len(self.absorbing_states()) > 0

    def communicating_classes(self):
        G = self._as_digraph()
        return list(networkx.strongly_connected_components(G))

    def recurrent_classes(self):
        raise NotImplementedError('todo')
        return []

    def transient_classes(self):
        raise NotImplementedError('todo')
        return []


