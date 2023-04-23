from attention import *
from feedforward import *

class DecodeGPTStyle(nn.Module):
    """This class is a modified decoding layer with one multihead attention layer and a feed forwad neural network. This class
    is designed to replicate the decoding layers found in gpt-2 layers"""
    def __init__(self, multiheads: int, embedding_size_input: int, reduced_emb: int, hidden_ff: int, mask: torch.Tensor) -> None:
        """embedding_size input - This modified decoding layer assumes that the dimension of the words it recieves
        has dimension embedding_size input"""
        super(DecodeGPTStyle, self).__init__()
        assert (embedding_size_input // multiheads) == (embedding_size_input / multiheads), "The embedding size must be divisble by the multi head layer"
        self.embedding_size_input = embedding_size_input
        self.reduced_emb = reduced_emb
        self.hidden_ff = hidden_ff
        self.multiheads = multiheads
        self.mask = mask
        self.first_multi_layer = MultiHeadAttention(multiheads, embedding_size_input, embedding_size_input, reduced_emb, mask)
        self.feed_forward_layer = FeedForward(embedding_size_input, hidden_ff, embedding_size_input)
        self.w0 = torch.nn.Parameter(torch.rand((1, reduced_emb * multiheads, embedding_size_input), requires_grad=True))
    
    def forward(self, inputs: torch.Tensor):
        outputs = self.first_multi_layer.forward(inputs, inputs, inputs.shape[0], inputs.shape[1], inputs.shape[0], inputs.shape[1], self.mask)
        outputs = torch.matmul(outputs, self.w0)
        outputs = outputs + inputs
        outputs1 = torch.nn.functional.normalize(outputs, 2, 0)

        outputs = self.feed_forward_layer.forward(outputs1)
        outputs = outputs + outputs1
        outputs = torch.nn.functional.normalize(outputs, 2, 0)

        return outputs
    



