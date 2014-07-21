import unittest
import dtmc

class DiscreteTimeMarkovChainTest(unittest.TestCase):

    def test_trivial(self):
        chain = dtmc.DiscreteTimeMarkovChain([[1.]])
        self.assertTrue(chain is not None)

    def test_non_square_input(self):
        with self.assertRaises(dtmc.DTMCError): 
            dtmc.DiscreteTimeMarkovChain([[1., 0.]])

    def test_non_stochastic_matrix(self):
        with self.assertRaises(dtmc.DTMCError): 
            dtmc.DiscreteTimeMarkovChain([[0.]])

    def test_trivially_irreducible(self):
        chain = dtmc.DiscreteTimeMarkovChain([[1.]])
        self.assertTrue(chain.irreducible())
