import unittest
from attention import *

class TestAttention(unittest.TestCase):

    def test_passthrough_singlehead(self):
        y = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(1, y, y, 2, 3, 8, 2, 3, 8, 5, None)
        catch = ma.forward()
        assert tuple(catch.shape) == (2, 3, 5)
    
    def test_passthrough_doublehead(self):
        y = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(2, y, y, 2, 3, 8, 2, 3, 8, 5, None)
        catch = ma.forward()
        assert tuple(catch.shape) == (2, 3, 10)
    
    def test_encoder_decoder_pass(self):
        y = torch.rand((2, 5, 16))
        x = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(2, y, x, 2, 5, 16, 2, 3, 8, 5, None)
        catch = ma.forward()
        assert tuple(catch.shape) == (2, 3, 10)
    


if __name__ == '__main__':
    unittest.main(verbosity=True)