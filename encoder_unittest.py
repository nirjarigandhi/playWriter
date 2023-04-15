import unittest

from encoderlayer import *

class TestEncoder(unittest.TestCase):

    def test_general_run(self):
        inputs = torch.rand((2, 5, 16))
        en = EncoderLayer(16, 4, 3, 20)
        catch = en.forward(inputs, 2, 5)
        print(catch)
        assert tuple(catch.shape) == tuple(inputs.shape)



if __name__ == '__main__':
    unittest.main(verbosity=True)
