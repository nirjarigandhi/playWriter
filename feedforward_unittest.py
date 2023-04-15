import unittest
from feedforward import *

class TestFeedForward(unittest.TestCase):

    def test_expected_dimensionality(self):
        """Sanity"""
        y = torch.rand((2, 3 ,4))
        layer = FeedForward(4, 20, 5)
        result = layer.forward(y)
        assert tuple(result.shape) == (2, 3, 5)



if __name__ == '__main__':
    unittest.main(verbosity=True)