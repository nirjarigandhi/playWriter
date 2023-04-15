import unittest
from attention import *
from random import randint

class TestAttention(unittest.TestCase):

    def test_passthrough_singlehead(self):
        y = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(1, 8, 8, 5, None)
        catch = ma.forward(y, y, 2, 3, 2, 3, None)
        assert tuple(catch.shape) == (2, 3, 5)
    
    def test_passthrough_doublehead(self):
        y = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(2, 8, 8, 5, None)
        catch = ma.forward(y, y, 2, 3, 2, 3, None)
        assert tuple(catch.shape) == (2, 3, 10)
    
    def test_encoder_decoder_pass(self):
        y = torch.rand((2, 5, 16))
        x = torch.rand((2, 3, 8))
        ma = MultiHeadAttention(2, 16, 8, 5, None)
        catch = ma.forward(y, x, 2, 5, 2, 3, None)
        assert tuple(catch.shape) == (2, 3, 10)


        y = torch.rand((2, 5, 32))
        x = torch.rand((2, 3, 8))
        catch = ma.forward(y, x, 2, 5, 2, 3, None)
        assert tuple(catch.shape) == (2, 3, 10)
    
    def test_random(self):
        y_batch = randint(4, 10)
        y_seqlen = randint(8, 20)
        y_embedding_Size = 20
        heads = 4
        reduced_emb = 4
        y = torch.rand(y_batch, y_seqlen, y_embedding_Size) #decoder input kv
        x = torch.rand(y_batch, 8, 16) #encoder inputkv
        ma = MultiHeadAttention(heads, 16, y_embedding_Size, reduced_emb, None)
        catch = ma.forward(x, y, y_batch, 8, y_batch, y_seqlen, None)
        assert tuple(catch.shape) == (y_batch, y_seqlen ,reduced_emb * heads)

    


if __name__ == '__main__':
    unittest.main(verbosity=True)