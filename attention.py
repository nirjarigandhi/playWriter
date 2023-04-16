from __future__ import annotations
import torch
import torch.nn as nn
import torch.autograd
import math

class SingleAttention(nn.Module):
    """This class is designed to simulate exactly one attention head. 
    It will take a matrix of (batch_size, sentence_lenght, embedding_size)
    batch_size, sentence_length, and embedding size, are all sizes that are fed into this class
    these sizes speficically embedding size can be manipulated by linear layers before entering this class.
    Reduced emb is a variable specific to each query, value, and key matrix
    
    NOTE that embedding_size and reduce emb are fixed for every tensor matrix that enters the encoder they cannot change regardless of the data coming through"""
    def __init__(self, embedding_size_kv: int, embedding_size_q: int, reduced_emb: int, mask=None) -> None:
        super(SingleAttention, self).__init__()
        self.inputs_kv = None #Has the shape (batch_kv, sentence_length_kv, embedding_kv)
        self.inputs_q = None
        self.batch_size_kv = None
        self.sentence_length_kv = None
        self.embedding_size_kv = embedding_size_kv
        self.batch_size_q = None
        self.embedding_size_q = embedding_size_q
        self.sentence_length_q = None
        self.reduced_emb = reduced_emb
        self.mask = mask
        self.query_matrix = torch.rand((1, self.embedding_size_q, self.reduced_emb), requires_grad=True) # The input vector is of size (seq, batch, emb) input * self.query_vector
        torch.nn.init.xavier_normal_(self.query_matrix, gain=3.5)
        self.key_matrix = torch.rand((1, self.embedding_size_kv, self.reduced_emb), requires_grad=True)
        torch.nn.init.xavier_normal_(self.key_matrix, gain=3.5)
        self.value_matrix = torch.rand((1, self.embedding_size_kv, self.reduced_emb), requires_grad=True)
        torch.nn.init.xavier_normal_(self.value_matrix, gain=3.5)
        self.querys = None
        self.keys = None
        self.values = None
        self.attention_matrix = None #This is of shape (batch_q = batch_kv, sentence_q, sentence_kv)
        self.final = None
    
    def update(self, inputs_kv: torch.Tensor, inputs_q:torch.Tensor ,batch_size_kv: int, sentence_length_kv: int, batch_size_q:int, sentence_length_q: int, mask=None):
        """Changes the values that need to be changed with every new set of data flowing through the prediction network"""
        assert batch_size_kv == batch_size_q, "The batch sizes of both the kv and q inputs must be equal"
        self.inputs_kv = inputs_kv #Has the shape (batch_kv, sentence_length_kv, embedding_kv)
        self.inputs_q = inputs_q
        self.batch_size_kv = batch_size_kv
        self.sentence_length_kv = sentence_length_kv
        self.batch_size_q = batch_size_q
        self.sentence_length_q = sentence_length_q
        self.mask = mask
        self.querys = None
        self.keys = None
        self.values = None
        self.attention_matrix = None #This is of shape (batch_q = batch_kv, sentence_q, sentence_kv)
        self.final = None


    def _get_querys(self):
        """Get query associate with each word". This is a right multiplication for the
        input tensor * self.query_matrix ie (batch, sentence_length, emb) * (1, emb, reduced_emb) = 
        (batch, sentence_length, reduced_emb). This is the case for the other three functions below"""

        self.querys = torch.matmul(self.inputs_q, self.query_matrix)

        return self.querys #is of size (batchsize_q, sentence_length_q, reduced_emb)
    
    def _get_keys(self):
        """Get the keys associated with each word"""

        self.keys = torch.matmul(self.inputs_kv, self.key_matrix)

        return self.keys # is of size (batch_size_kv, sentence_lengt_kv, reduced_emb)

    def _get_values(self):
        """Get the values associated with each word"""

        self.values = torch.matmul(self.inputs_kv, self.value_matrix)

        return self.values
    
    def _get_coefficents(self):
        """Get the coefficents of the matrix Note that"""
        keys_transpose = self.keys.transpose(1, 2)

        #Ideally batch_q = batch_kv this is a must
        results = torch.matmul(self.querys, keys_transpose) #(batch_q, sentence_len_q, reduced_emb) * (batch_kv, reduced_emb, sentence_len_kv)
        results = results / math.sqrt(self.reduced_emb)

        if self.mask is not None:
            assert self.mask.shape() == results.shape(), "Masks need to have the same size as the attention_matrix"
            results = results * self.mask
        
        results = torch.softmax(results, 2) #compute softmax on dimension 2

        self.attention_matrix = results
    
    def get_final(self):

        final = torch.matmul(self.attention_matrix, self.values) #this is now a matrix of (batch, sentence_q, reduced_emb)

        self.final = final

        return final
    
    def forward(self, inputs_kv: torch.Tensor, inputs_q:torch.Tensor ,batch_size_kv: int, sentence_length_kv: int,  batch_size_q:int, sentence_length_q: int, mask=None):
        self.update(inputs_kv, inputs_q, batch_size_kv, sentence_length_kv, batch_size_q, sentence_length_q, mask)
        self._get_querys()
        self._get_keys()
        self._get_values()
        self._get_coefficents()

        return self.get_final()


class MultiHeadAttention(nn.Module):
    """This class implements multihead attention using smaller single head attention modules. The main work being done
    here is splitting the inputs by their embedding and creating heads amount of single attention classes to give those split workloads too.
    Remember that the input tensors should have format (batch, sentence_length, embedding_size ) and so should the output tensors"""
    def __init__(self, heads: int, embedding_size_kv: int, embedding_size_q: int,reduced_emb: int, mask=None) -> None:
        super(MultiHeadAttention, self).__init__()
        assert (embedding_size_kv / heads == embedding_size_kv // heads), "The embedding size of the key and value matrices must be divisible by number of heads"
        assert (embedding_size_q / heads == embedding_size_q // heads), "The embedding size of the querys must be divisble by the number of heads"
        self.inputs_kv = None
        self.inputs_q = None
        self.batch_size_kv = None
        self.sentence_length_kv = None 
        self.embedding_size_kv = embedding_size_kv # when placed into the single attention classes this will be divided by heads
        self.batch_size_q = None
        self.embedding_size_q = embedding_size_q
        self.sentence_length_q = None
        self.reduced_emb = reduced_emb #Remeber that reduced emb is not the split that happens with mha but one of the dims of the query, value, and key matrices that occur in the single attention layers
        self.heads = heads
        self.mask = mask
        self.output_tensor = None
        self.attention_list = nn.ModuleList([SingleAttention(self.embedding_size_kv // self.heads , self.embedding_size_q // self.heads , self.reduced_emb, self.mask) for i in range(self.heads)])

    def update(self, inputs_kv: torch.Tensor, inputs_q:torch.Tensor ,batch_size_kv: int, sentence_length_kv: int, batch_size_q:int, sentence_length_q: int, mask=None):
        self.inputs_kv = inputs_kv
        self.inputs_q = inputs_q
        self.batch_size_kv = batch_size_kv
        self.sentence_length_kv = sentence_length_kv
        self.batch_size_q = batch_size_q
        self.sentence_length_q = sentence_length_q
        self.mask = mask
        self.output_tensor = None
    
    def forward(self, inputs_kv: torch.Tensor, inputs_q:torch.Tensor ,batch_size_kv: int, sentence_length_kv: int, batch_size_q:int, sentence_length_q: int, mask=None):
        self.update(inputs_kv, inputs_q, batch_size_kv, sentence_length_kv, batch_size_q, sentence_length_q, mask)
        list_of_input_kvs =torch.split(self.inputs_kv, self.embedding_size_kv // self.heads, 2)
        list_of_input_qs = torch.split(self.inputs_q, self.embedding_size_q // self.heads, 2)
        outputs = [None for i in range(self.heads)]

        for i in range(self.heads):
            outputs[i] = self.attention_list[i].forward(list_of_input_kvs[i], list_of_input_qs[i], self.batch_size_kv, self.sentence_length_kv, self.batch_size_q, self.sentence_length_q, mask)
        
        result = torch.concat(outputs, 2) # This should have size (batch_q = batch_kv, sentence_length_q, reduced_emb * self.heads)
        self.output_tensor = result.clone()

        return result


        




    


    
