from __future__ import annotations
import torch
import torch.nn as nn
import torch.autograd
from attention import *
from feedforward import *


class EncoderLayer(nn.Module):
    """This is the encoder layer it combines the attention plus the feed forward layer of the neural network
    Note that batch_size, sentence_length, and an embedding_size tensor goes into this class and a tensor of the same size
    comes out"""
    def __init__(self, embedding_size: int, heads: int, reduced_emb: int, hidden_dim_ff: int) -> None:
        super(EncoderLayer, self).__init__()
        self.inputs = None
        self.batch_size = None
        self.sentence_length = None
        self.embedding_size = embedding_size
        self.heads = heads
        self.reduced_emb = reduced_emb
        self.hidden_dim_ff = hidden_dim_ff
        self.multiattention = MultiHeadAttention(self.heads, self.embedding_size, self.embedding_size, self.reduced_emb)
        self.feed_forward = FeedForward(self.embedding_size, hidden_dim_ff, self.embedding_size)
        self.w0 = torch.nn.Parameter(torch.rand((1, self.reduced_emb * self.heads, self.embedding_size), requires_grad=True)) # remember that batch_kv = batch_q and in this instance all q,k,v tensors share the same dimensions. What comes out of the multi head attentions is (batch_kv, seq_q, reduced_emb * self.heads)
        torch.nn.init.xavier_normal_(self.w0)
        self.result = None

    def update(self, inputs: torch.Tensor, batch_size: int, sentence_length: int):
        self.inputs = inputs
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.result = None
    
    def forward(self, inputs: torch.Tensor, batch_size: int, sentence_length: int):
        self.update(inputs, batch_size, sentence_length)
        store1 = self.inputs.clone()
        out = self.multiattention.forward(self.inputs, self.inputs, self.batch_size, self.sentence_length, self.batch_size, self.sentence_length)
        out = torch.matmul(out, self.w0) # will now have shape (batch_kv = batch = batch_q, sentence_length = seq_q = seq_kv, embedding_size)
        second = store1 + out
        second = torch.nn.functional.normalize(second, 2, dim=0)
        store2 = second.clone() #(batch, seq, emb) size
        out = self.feed_forward(second) #(batch, seq, emb) size
        out = store2 + out
        result = torch.nn.functional.normalize(out, 2, 0)
        self.result = result.clone() #(batch, seq, emb) size

        return result









    
