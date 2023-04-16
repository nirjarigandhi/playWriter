from encoderlayer import *
from decoderlayer import *
from Posencoding import *

class Transformer(nn.Module):
    def __init__(self, encoder_amount: int, encoder_head: int, encoder_embedding: int, encoder_reduced_emb: int, encoder_hidden_ff: int, decoder_amount: int, decoder_head_first: int, decoder_head_second: int, decoder_embedding: int, decoder_reduced_emb_first: int, decoder_reduced_emb_second: int, onehot_embedding_size: int, decode_mask = None) -> None:
        """encoder_embedding - The embedding size of the data for the encoder and may or may not be the same as decoder embedding
           encoder_amount    - The amount of encoder layers to pass through
           encoder_head      - The amount of multihead attention to give the encoder attention layer
           encoder_reduced_emb - The amount to reduce the q, v, k word representaions to from their original form
           
           Note that in this class the encoder_embedding and decoder_embedding share the same dimension despite being two seperate inputs
           This class will throw an error if that is not the case"""
        super(Transformer, self).__init__()
        assert encoder_amount > 0, "The encoder layers need to be greater than 0"
        assert decoder_amount > 0, "The decoder layers need to be greater than 0"
        self.encoder_amount = encoder_amount
        self.encoder_embedding = encoder_embedding
        self.decoder_amount = decoder_amount
        self.decoder_embedding = decoder_embedding
        self.onehot_embedding_size = onehot_embedding_size
        self.encoders = [EncoderLayer(encoder_embedding, encoder_head, encoder_reduced_emb, encoder_hidden_ff) for i in range(encoder_amount)]
        self.decoders = [DecoderLayer(encoder_embedding, decoder_embedding, decoder_head_first, decoder_reduced_emb_first, decoder_head_second, decoder_reduced_emb_second, decode_mask) for i in range(decoder_amount)]
        self.input_one_hot_mutate = torch.rand((1, onehot_embedding_size, encoder_embedding), requires_grad=True)
        self.output_to_onehot = torch.rand((1, self.decoder_embedding, onehot_embedding_size), requires_grad=True)
        torch.nn.init.xavier_normal_(self.input_one_hot_mutate, 5)
        torch.nn.init.xavier_normal_(self.output_to_onehot, 5)
        self.posencode = PositionalEmbeddings()



    def forward(self, encoder_onehot_inputs_raw_tensor: torch.Tensor, decoder_onehot_inputs_raw_tensor: torch.Tensor):
        encoder_inputs = torch.matmul(encoder_onehot_inputs_raw_tensor, self.input_one_hot_mutate)
        encoder_inputs = self.posencode.add_posencoding(encoder_inputs, self.encoder_embedding, tuple(encoder_inputs.shape)[1], tuple(encoder_inputs.shape)[0])
        # Convert both the onehot vectors fed to the network from both the encoder and decoder
        assert self.encoder_embedding == self.decoder_embedding, "The embeddings are not equal"
        decoder_inputs = torch.matmul(decoder_onehot_inputs_raw_tensor, self.input_one_hot_mutate)
        decoder_inputs = self.posencode.add_posencoding(decoder_inputs, self.encoder_embedding, tuple(decoder_inputs.shape)[1], tuple(decoder_inputs.shape)[0])

        assert tuple(encoder_onehot_inputs_raw_tensor.shape)[0] == tuple(decoder_onehot_inputs_raw_tensor.shape)[0], "The batch sizes of the encoder and decoder tensors must be equal!"

        #Compute encoder pass
        encoder_allpass_outputs = encoder_inputs

        for i in range(self.encoder_amount):
            encoder_allpass_outputs = self.encoders[i].forward(encoder_allpass_outputs, tuple(encoder_allpass_outputs.shape)[0], tuple(encoder_allpass_outputs.shape)[1])

        

        # Complete the decoder pass
        decoder_allpass_outputs = decoder_inputs
        for i in range(self.decoder_amount):
            decoder_allpass_outputs = self.decoders[i].forward(encoder_allpass_outputs, tuple(encoder_allpass_outputs.shape)[0], tuple(encoder_allpass_outputs.shape)[1], decoder_allpass_outputs, tuple(decoder_allpass_outputs.shape)[0], tuple(decoder_allpass_outputs.shape)[1], None)
        

        result = torch.matmul(decoder_allpass_outputs, self.output_to_onehot)

        self.pre_logit = torch.softmax(result, 2)

        self.final = torch.argmax(self.pre_logit, 2)
        self.final  = torch.Tensor(torch.nn.functional.one_hot(self.final, self.onehot_embedding_size))

        return self.final
        

        

        
        

        
