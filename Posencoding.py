from __future__ import annotations
import torch
import torch.nn as nn
import torch.autograd
import math

class PositionalEmbeddings(nn.Module):
    def __init__(self, input: torch.Tensor, embedding_size: int, sentence_length: int, batch_size) -> None:
        """Input tensors must be of the form (batch, sentence_length, embedding_size)
        Note the variable i is a bijection from the naturals onto the positions into the vector_word embeddings"""
        super(PositionalEmbeddings, self).__init__()
        self.inputs = input.clone() #(inputs of the form (batch, sentence_length, embedding_size))
        self.embedding_size= embedding_size
        self.output = None # Outputs will later be of size (batch, sentence_length, embedding_size) for the sake of simplicity
        self.sentence_length = sentence_length
        self.batch_size = batch_size

    
    def _sin_equation(self, pos: int, i: int):
        """This is a function that applies the positional encoding formula for sine
        dmodel is the embedding size. Both even and odd i go in here"""
        exp = 2*i / self.embedding_size
        denominator = pow(10000, exp=exp)
        
        return math.sin(pos / denominator)
    
    def _cos_equation(self, pos: int, i: int):
        """This is a positional encoding formula that applies for cosine
        d_model is the embedding size. Both even an odd i go in here"""
        exp = 2*i / self.embedding_size
        denominator = pow(10000, exp=exp)

        return math.cos(pos /denominator)
    
    def _posencoding_maker(self) -> torch.Tensor:
        """sentence_tensor is of form (seq, batch, emb) where we will write a function that creates the sin and
        cos embedings in a similar matrix for (seq, batch emb)"""

        sincos = torch.zeros((self.sentence_length, self.embedding_size)) # Create a matrix that will eventually be broadcasted to employ positional encodings

        for pos in range(self.sentence_length):
            for i in range(self.embedding_size // 2):
                sincos[pos, 2 * i] =self._sin_equation(pos, i)
                if (2 * i) + 1 < self.embedding_size: # Ensure that this last position exists before indexing it
                    sincos[pos, (2 * i) + 1] = self._cos_equation(pos, i)
            
    
        return sincos.unsqueeze(0) #The shape is now (1, seq, emb_size) which will allow for broadcasting with the positional vectors
    
    def concat_posencoding(self) -> torch.Tensor:
        """This will append the positional encoding at the end of every word in the "inputs" tensor if the input tensor has 
        dimension (seq, batch, emb) then the resulting vector will be (seq, batch, 2emb) after creating the class call this method only"""
        ones = torch.ones((self.batch_size, self.sentence_length, self.embedding_size), requires_grad= False)

        dim_match = ones * self._posencoding_maker() #Broadcast the positional encodings to size of the input tensor

        self.output =  torch.concat((self.inputs, dim_match), 2) # This will append the pos encodings onto the end of every word vector for all groups (batches) of sentences

        return self.output

    
    