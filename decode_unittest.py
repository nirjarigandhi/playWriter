import unittest

from decoderlayer import *

class TestDecoder(unittest.TestCase):

    def test_general_run(self):
        inputs = torch.rand((2, 5, 16))
        encoder_inputs = torch.rand((2, 30, 32))
        de = DecoderLayer(32, 16, 2, 5, 2, 5)
        catch = de.forward(encoder_inputs, 2, 30, inputs, 2, 5)
        assert tuple(catch.shape) == tuple(inputs.shape)



if __name__ == '__main__':
    unittest.main(verbosity=True)
