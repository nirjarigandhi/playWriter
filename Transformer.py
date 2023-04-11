from __future__ import annotations
import torch
import torch.nn as nn
import torch.autograd
import math


def sin_equation(pos: int, i: int, dmodel: int):
    """This is a function that applies the positional encoding formula for sine
    dmodel is the embedding size"""
    exp = 2*i / dmodel
    denominator = pow(10000, exp=exp)
    
    return math.sin(pos / denominator)

def cos_equation(pos: int, i: int, dmodel: int):
    """This is a positional encoding formula that applies for cosine
    d_model is the embedding size"""
    exp = 2*i /dmodel
    denominator = pow(10000, exp=exp)

    return math.cos(pos /denominator)
    

def posencoding_maker(sequence_length: int, embedding_size: int) -> torch.Tensor:
    """sentence_tensor is of form (seq, batch, emb) where we will write a function that creates the sin and
    cos embedings in a similar matrix for (seq, batch emb)"""

    sincos = torch.zeros((sequence_length, embedding_size)) # Create a matrix that will eventually be broadcasted to employ positional encodings

    for pos in range(sequence_length):
        for i in range(embedding_size // 2):
            sincos[pos, 2 * i] =sin_equation(pos, i, embedding_size)
            if (2 * i) + 1 < embedding_size: # Ensure that this last position exists before indexing it
                sincos[pos, (2 * i) + 1] = cos_equation(pos, i, embedding_size)
            
    
    return sincos.unsqueeze(1) #The shape is now (seq, 1, emb_size) which will allow for broadcasting with the positional vectors

def concat_posencoding(inputs: torch.Tensor, batch_size:int, sequence_length: int, embedding_size: int) -> torch.Tensor:
    """This will append the positional encoding at the end of every word in the "inputs" tensor if the input tensor has 
    dimension (seq, batch, emb) then the resulting vector will be (seq, batch, 2emb)"""
    ones = torch.ones((sequence_length, batch_size, sequence_length), requires_grad= False)

    dim_match = ones * posencoding_maker(sequence_length=sequence_length, embedding_size=embedding_size) #Broadcast the positional encodings to size of the input tensor

    return torch.concat((inputs, dim_match), 2) # This will append the pos encodings onto the end of every word vector for all groups (batches) of sentences



class Pay_Attention(nn.Module):
    """ This class will implement exactly one attention head
    There is one query matrix, key matrix, value matrix"""
    def __init__(self, reduced_emb: int, batch_size: int, sequence_length: int, embedding_size: int) -> None:
        super(Pay_Attention, self).__init__()
        self.reduced_emb = reduced_emb
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.query_matrix = torch.rand((1, self.embedding_size, self.reduced_emb), requires_grad=True) # The input vector is of size (seq, batch, emb) input * self.query_vector
        torch.nn.init.xavier_normal_(self.query_matrix, gain=3.5)
        self.key_matrix = torch.rand((1, self.embedding_size, self.reduced_emb), requires_grad=True)
        torch.nn.init.xavier_normal_(self.key_matrix, gain=3.5)
        self.value_matrix = torch.rand((1, self.embedding_size, self.reduced_emb), requires_grad=True)
        torch.nn.init.xavier_normal_(self.value_matrix, gain=3.5)
        self.querys = None
        self.keys = None
        self.values = None
        self.attention = None
    
    def get_querys(self, input_tensor: torch.Tensor):
        """Get query associate with each word". This is a right multiplication for the
        input tensor * self.query_matrix ie (seq, batch, emb) * (1, emb, reduced_emb) = 
        (seq, batch, reduced_emb). This is the case for the other three functions below"""

        self.querys = input_tensor * self.query_matrix

        return self.querys
    
    def get_keys(self, input_tensor: torch.Tensor):
        """Get the keys associated with each word"""

        self.keys = input_tensor * self.key_matrix

        return self.keys
    
    def get_values(self, input_tensor: torch.Tensor):
        """Get the values associated with each word"""

        self.values = input_tensor * self.key_matrix

        return self.values
    
    # The past three functions results in a matrix of size (seq, batch, reduced_emb)
    def get_attention_coefficients(self):
        """This will compute the attention coeffeicents where each word's query will be compare to every key
        all present in the first row of this existing matrix.
        
        **** NOTE that these functions get the coefficients it will first transpose the initial reduced embedding
        of the querys from (seq, batch, reduced_emb) to (batch, seq, reduced_emb)****"""
        querys_transpose = self.querys.transpose(0, 1) # THis is now of shape (batch, seq, reduced_emb)
        keys_transpose = self.keys.transpose(0, 1) # This is now of shape (batch, seq, reduced_emb)
        keys_transpose = keys_transpose.transpose(2, 1) #This is now of shape (batch, reduced_emb ,seq)

        result = querys_transpose * keys_transpose # This is now of shape (batch, seq, seq)
        result = result / math.sqrt(self.reduced_emb)

        self.attention = torch.softmax(result) # This is still of shape (batch, seq, seq)

        return self.attention

    
    def most_likely_values(self):
        """This will multiply the the values against the attention matrix to see which value is associated with which word"""

        self.attention_final = self.attention * self.values.transpose(0, 1) 
        self.attention_final = self.attention_final.transpose(0, 1) #Final shape(seq, batch, reduced_emb)

        return self.attention_final # Attention has been given to a specific word in the given matricies


        

class FeedForward(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(FeedForward, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.first = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.second = torch.relu(self.first)
        self.last = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.result = None

    def forward(self, input_tensor: torch.Tensor):

        self.result = self.first(input_tensor)
        self.result = self.second(self.result)
        self.result = self.last(self.result)

        return self.result



class MultiHeadAttention(nn.Module):
    """This is a multiheaded version of the Pay_Attention class using the pay attention class"""

    def __init__(self, multi_head: int, batch_size: int, sequence_size: int, embedding_size: int, reduced_emb: int):
        super(MultiHeadAttention, self).__init__()
        assert (embedding_size / multi_head == embedding_size // multi_head), "The Embedding size must be divisible by the amount of heads to use"
        assert (embedding_size / multi_head >= reduced_emb), "Reduced emb is bigger than the size of the embedding dimension in the reduced multiheaded "
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.embedding_size = embedding_size
        self.reduced_emb = reduced_emb
        self.multi_head = multi_head
        self.output_tensor = None

    def forward(self, input_tensor: torch.Tensor) -> None:
        """batch_size, sequence, embedding_size, and reduced_emb: all refer to the """
        list_of_attention = [Pay_Attention(self.reduced_emb, self.batch_size, self.sequence_size, self.embedding_size) for i in range(self.multi_head)]
        input_split = input_tensor.split(self.multi_head, 2) # given tensor of (seq, batch, emb) it will be split into (seq, batch, emb/multi_head)
        outputs = [None for i in range(self.multi_head)]
        for i in range(self.multi_head):
            list_of_attention[i].get_querys(input_split[i])
            list_of_attention[i].get_keys(input_split[i])
            list_of_attention[i].get_values(input_split[i])
            list_of_attention[i].get_attention_coefficients()
            outputs[i] = list_of_attention[i].most_likely_values()
        output_tensor = torch.concat(outputs, 2) #This should have size (seq, batch, reduced_emb)
        self.output_tensor = output_tensor

        return output_tensor




class EncoderLayer(nn.Module):

    def __init__(self, batch_size: int, sequence_length: int, embedding_size: int, reduced_emb: int, multi_head_size: int, ff_input: int, ff_hidden, ff_output) -> None:
        super(EncoderLayer, self).__init__()
       
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.reduced_emb = reduced_emb
        self.multi_head_size = multi_head_size
        self.ff_hiden = ff_hidden
        self.ff_input = ff_input
        self.ff_output == ff_output


        self.attention = MultiHeadAttention(multi_head_size, batch_size, sequence_length, embedding_size, reduced_emb)
        self.feed_forward = FeedForward(ff_input, ff_hidden, ff_output)

    
    def forward(self, input_tensor: torch.Tensor) -> None:
        result = self.attention.forward(input_tensor)

        #After the multihead concatination this will reshape all the data that is left to reshape back into (seq, batch, embsize)
        w_0 = torch.ones((1, self.reduced_emb, self.embedding_size), requires_grad=True)
        torch.nn.init.xavier_normal_(w_0, gain = 3.5) ## Initialize the weights in a nice way

        result = result * w_0 # put the vectors back into (seq, batch, emb_size) this will be summed and normalized to the original input tensor

        result += input_tensor #(seq, batch, emb_size)

        add_norm_first = torch.nn.functional.normalize(result, dim=1) #(seq, batch, emb_size)

        result = self.feed_forward.forward(add_norm_first)

        result += add_norm_first # this can only work if the output matches the emb size of the feed forward neural network

        self.final_result = torch.nn.functional.normalize(result, dim=1)





        


        

    
    







    




