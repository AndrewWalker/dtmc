import unittest
import dtmc

class DiscreteTimeMarkovChainTest(unittest.TestCase):

    def test_trivial(self):
        chain = dtmc.DiscreteTimeMarkovChain([[0.]])
