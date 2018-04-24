import numpy as np

import torch
import torch.nn as nn

from models.attention import Attention
import data.utils as U

class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, key_size, value_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_size)

        self.attention_combine = nn.Linear(value_size+embedding_size, hidden_size)

        self.h0 = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.c0 = nn.Parameter(torch.FloatTensor(1, hidden_size))

        for i in range(self.num_layers):
            self.add_module("lstm"+str(i), nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))

        self.attention = Attention(hidden_size, key_size, value_size, output_size)


    def forward_step(self, input_var, decoder_hiddens, context, encoder_keys, encoder_values):
        # input_var = (batch_size, 1)
        # context = (batch_size, 1, value_size)

        embedding = self.embedding(input_var)
        # embedding = (batch_size, 1, embedding_size)
        decoder_input = torch.cat((embedding, context), dim=2)
        # inputs = (batch_size, 1, embedding_size+value_size)

        combined_input = self.attention_combine(decoder_input)
        # combined_input = (batch_size, 1, hidden_size)
        hidden_state = combined_input.squeeze(1)

        new_decoder_hiddens = []
        for i in range(self.num_layers):
            layer_fn = getattr(self, "lstm"+str(i))
            hidden_state, cell_state = layer_fn(hidden_state, decoder_hiddens[i])
            new_decoder_hiddens.append((hidden_state, cell_state))

        outputs = hidden_state.unsqueeze(1)
        context, outputs = self.attention(outputs, encoder_keys, encoder_values)
        outputs = outputs.squeeze(1)

        return outputs, new_decoder_hiddens, context


    def forward(self, decoder_targets, encoder_keys, encoder_values, encoder_lens, teacher_forcing_ratio=1.0):
        # decoder_targets = (batch_size, max_target_len)
        batch_size = decoder_targets.size(0)
        max_target_len = decoder_targets.size(1) - 1 #exclude EOS

        decoder_outputs = []

        timestamp = 0
        decoder_input = decoder_targets[:, timestamp].unsqueeze(1)
        decoder_hiddens = self._init_hidden_state(batch_size)
        decoder_output = decoder_hiddens[self.num_layers-1][0].unsqueeze(1)
        
        mask = U.create_mask(encoder_lens).unsqueeze(1)
        self.attention.set_mask(mask)
        context, _ = self.attention(decoder_output, encoder_keys, encoder_values)
        
        use_teacher_forcing = True 
        #if np.random.random() < teacher_forcing_ratio else False

        for timestamp in range(1, max_target_len): 
            decoder_output, decoder_hiddens, context = self.forward_step(decoder_input, decoder_hiddens, 
                                                            context, encoder_keys, encoder_values)
            step_output = decoder_output.squeeze(1)
            #symbols = decode(step_output)
            decoder_outputs.append(step_output)

            if use_teacher_forcing:
                decoder_input = decoder_targets[:, timestamp].unsqueeze(1)
            else:
                decoder_input = symbols

        return decoder_outputs


    def _init_hidden_state(self, batch_size):
        hiddens = []
        for i in range(self.num_layers):
            hidden = (self.h0.expand(batch_size, -1), self.c0.expand(batch_size, -1))
            hiddens.append(hidden)
        return hiddens


def _test():
    max_input_len = 4
    max_target_len = 10

    batch_size = 4
    hidden_size = 12
    key_size = 10
    value_size = 10
    embedding_size = 4
    output_size = 33
    num_layers = 3
    
    decoder_targets = U.var(torch.ones(batch_size, max_target_len).long())

    outputs = U.var(torch.randn(batch_size, 1, hidden_size))
    encoder_lens = list(range(1, batch_size+1))
    encoder_keys = U.var(torch.randn(max_input_len, batch_size, key_size))
    encoder_values = U.var(torch.randn(max_input_len, batch_size, value_size))

    decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers)
    
    if U.is_cuda():
        decoder = decoder.cuda()
    
    decoder_outputs = decoder(decoder_targets, encoder_keys, encoder_values, encoder_lens)


if __name__ == "__main__":
    _test()


