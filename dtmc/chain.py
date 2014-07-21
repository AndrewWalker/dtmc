import numpy

class DiscreteTimeMarkovChain(object):
    """A discrete time Markov chain
    """

    def __init__(self, transition_matrix):
        """Initialise a discrete time Markov Chain 

        Parameters
        ----------
        transition_matrix : matrix
            Data that describes the transitions between states in the
            chain. 
        """
        self._P = transition_matrix

