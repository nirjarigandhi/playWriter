from encoderlayer import *
from decoderlayer import *
from Posencoding import *

class Transformer(nn.Module):
    def __init__(self, encoder_amount: int, encoder_head: int, encoder_embedding: int, encoder_reduced_emb: int, encoder_hidden_ff: int, decoder_amount: int, decoder_head_first: int, decoder_head_second: int, decoder_embedding: int, decoder_reduced_emb_first: int, decoder_reduced_emb_second: int, decode_mask = None) -> None:
        super(Transformer, self).__init__()
        assert encoder_amount > 0, "The encoder layers need to be greater than 0"
        assert decoder_amount > 0, "The decoder layers need to be greater than 0"
        self.encoders = [EncoderLayer(encoder_embedding, encoder_head, encoder_reduced_emb, encoder_hidden_ff) for i in range(encoder_amount)]
        self.decoders = [DecoderLayer(encoder_embedding, decoder_embedding, decoder_head_first, decoder_reduced_emb_first, decoder_head_second, decoder_reduced_emb_second, decode_mask) for i in range(decoder_amount)]
        # self.input_one_hot_mutate = torch.rand(())


    def forward(self):
        pass
