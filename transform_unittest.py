import unittest

from Transformer import *

class TestTransformer(unittest.TestCase):

    def test_first(self):

        device = torch.device("cpu")
        words = torch.tensor([4, 5, 3, 2, 3, 4])
        onehots = torch.Tensor(torch.nn.functional.one_hot(words, 7)).to(device)
        onehots= onehots.unsqueeze(0)
        decode_in = torch.zeros((1, 1, 7)).to(device)
        decode_in[0, 0, 4] = 1

        engine = Transformer(3, 2, 10, 5, 20, 3, 2, 2, 10, 5, 5, 7, None).to(device)


        print(onehots.is_cuda)

        answers = engine.forward(onehots.float(), decode_in.float())

        print(answers)
        assert True






if __name__ == '__main__':
    unittest.main(verbosity=True)