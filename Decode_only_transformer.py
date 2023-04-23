from decode_custom import *


class DecodeTransformer(nn.Module):
    """This is decoder only transformer using a custom decoder module since there is no longer any encoder. The decoder
    is built in the style of gpt2"""
    def __init__(self, number_of_layers: int, multiheads: int, raw_embedding_size: int, embedding_size_input: int, reduced_emb: int, hidden_ff: int, mask: torch.Tensor) -> None:
        super(DecodeTransformer, self).__init__()
        self.num_layers = number_of_layers
        self.multiheads = multiheads
        self.embedding_size_input = embedding_size_input
        self.reduced_emb = reduced_emb
        self.hidden_ff = hidden_ff
        self.mask = mask
        self.raw_embedding_size = raw_embedding_size
        self.list_of_decoders = torch.nn.ModuleList([DecodeGPTStyle(multiheads,  embedding_size_input, reduced_emb, hidden_ff, mask) for i in range(number_of_layers)])
        self.transform_inputs = torch.nn.Parameter(torch.rand((1, raw_embedding_size, embedding_size_input), requires_grad=True))
        self.transform_outputs = torch.nn.Parameter(torch.rand((1, embedding_size_input, raw_embedding_size), requires_grad=True))

    def forward(self, inputs: torch.Tensor):
        new_inputs = torch.matmul(inputs, self.transform_inputs)
        results = new_inputs
        for i in range(self.num_layers):
            results = self.list_of_decoders[i].forward(new_inputs)
        
        self.pre_softmax = torch.matmul(results, self.transform_outputs)
        softmax = torch.nn.LogSoftmax(2)
        almost = softmax(self.pre_softmax)

        return almost
        
        
