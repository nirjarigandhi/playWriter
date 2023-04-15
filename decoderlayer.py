from encoderlayer import *

class DecoderLayer(nn.Module):
    """This is the decoder layer module class with two multi-head attentions followed by a feed forward layer
    the softmax and linear layer as found in the diagram in attention is all you need are not part of the decoder class"""

    def __init__(self, encoder_embedding_size: int, embedding_length_in: int, first_heads: int, reduced_emb_first: int, second_heads: int ,reduced_emb_second: int ,mask=None) -> None:
        super(DecoderLayer, self).__init__()
        self.encoder_tensor = None #This encoder has shape (batch_kv, sentence_length_kv, embedding_length_kv)
        self.input_tensor = None
        self.encoder_batch_size = None
        self.encoder_seqence_length = None
        self.encoder_embedding_size = encoder_embedding_size
        self.first_heads = first_heads
        self.batch_in = None
        self.sentence_length_in = None
        self.embedding_length_in = embedding_length_in
        self.reduced_emb_first = reduced_emb_first
        self.second_heads = second_heads
        self.reduced_emb_second = reduced_emb_second
        self.first_multihead = MultiHeadAttention(self.first_heads, self.embedding_length_in, self.embedding_length_in, self.reduced_emb_first, mask)
        self.second_multihead = MultiHeadAttention(self.second_heads, self.encoder_embedding_size, self.embedding_length_in, self.reduced_emb_second)

        self.w01 = torch.rand((1, self.reduced_emb_first * self.first_heads, self.embedding_length_in), requires_grad=True)
        torch.nn.init.xavier_normal_(self.w01)
        self.w02 = torch.rand((1, self.reduced_emb_second * self.second_heads, self.embedding_length_in), requires_grad=True) #notice that at every add and normalize the input must have the same shape as the output, which forces this second attention do to the same as well
        torch.nn.init.xavier_normal_(self.w02)

        self.feed_forward = FeedForward(self.embedding_length_in, self.embedding_length_in * 2, self.embedding_length_in)

        self.final = None

    def update(self, encoder_tensor: torch.Tensor, encoder_batch_size: int, encoder_sequence_length: int, input_tensor: torch.Tensor, batch_in: int, sentence_length_in: int):
        self.encoder_tensor = encoder_tensor #This encoder has shape (batch_kv, sentence_length_kv, embedding_length_kv)
        self.input_tensor = input_tensor
        self.encoder_batch_size = encoder_batch_size
        self.encoder_seqence_length = encoder_sequence_length
        self.batch_in = batch_in
        self.sentence_length_in = sentence_length_in
        self.final = None
    
    def forward(self, encoder_tensor: torch.Tensor, encoder_batch_size: int, encoder_sequence_length: int, input_tensor: torch.Tensor, batch_in: int, sentence_length_in: int, mask=None):
        self.update(encoder_tensor, encoder_batch_size, encoder_sequence_length, input_tensor, batch_in, sentence_length_in)

        first_multi_out = self.first_multihead.forward(self.input_tensor, self.input_tensor, self.batch_in, self.sentence_length_in, self.batch_in, self.sentence_length_in, mask)
        result1 = torch.matmul(first_multi_out, self.w01) #Now a tensor of dimension (self.batch_in, self.sentence_length_in, self.embedding_in)
        result1 = result1 + self.input_tensor
        result1 = torch.nn.functional.normalize(result1, 2, 0) # Not sure if dim 0 is the right one to normalize on

        second_multi_out = self.second_multihead.forward(self.encoder_tensor, self.input_tensor, self.encoder_batch_size, self.encoder_seqence_length, self.batch_in, self.sentence_length_in)
        result2 = torch.matmul(second_multi_out, self.w02)
        result2 = result2 + result1
        result2 = torch.nn.functional.normalize(result2, 2, 0) # This is of shape (batch_size, self.sentence_length_in, self.embedding_size_in)

        result3 = self.feed_forward.forward(result2)
        result3 = result3 + result2
        result3 = torch.nn.functional.normalize(result3, 2, 0) # The size is still (batch_size_in, self.sentence_length_in, self.embedding_size_in)

        self.final = result3.clone()

        return result3












