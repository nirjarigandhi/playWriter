import unittest

from encoderlayer import *

class TestEncoder(unittest.TestCase):

    def test_general_run(self):
        inputs = torch.rand((2, 5, 16))
        en = EncoderLayer(inputs, 2, 5, 16, 4, 7, 20)
        catch = en.forward()
        print(catch)
        assert tuple(catch.shape) == tuple(inputs.shape)



if __name__ == '__main__':
    unittest.main(verbosity=True)
    