import numpy as np

import torch
import torch.nn as nn

from models.attention import Attention
import data.utils as U
import pdb

np.random.seed(0)
torch.manual_seed(42)

class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, key_size, value_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.value_size = value_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)

        #self.attention_combine = nn.Linear(value_size+embedding_size, hidden_size)

        #self.params_h0 = nn.ParameterList(
        #   [nn.Parameter(torch.FloatTensor(1, self.hidden_size)) for i in range(self.num_layers)])
        #self.params_c0 = nn.ParameterList(
        #   [nn.Parameter(torch.FloatTensor(1, self.hidden_size)) for i in range(self.num_layers)])

        self.lstmCells = nn.ModuleList(
            [nn.LSTMCell(input_size=value_size+embedding_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)])

        self.attention = Attention(self.hidden_size, key_size, value_size, output_size)
        self.attention.projection.weight = self.embedding.weight


    def forward_step(self, input_var, decoder_hiddens, context, encoder_keys, encoder_values):
        #pdb.set_trace()
        # input_var = (batch_size)
        context = context.squeeze(1)
        # context = (batch_size, value_size)

        embedding = self.embedding(input_var)
        # embedding = (batch_size, embedding_size)
        decoder_input = torch.cat((embedding, context), dim=1)
        # inputs = (batch_size, embedding_size+value_size)

        # combined_input = self.attention_combine(decoder_input)
        # combined_input = (batch_size, 1, hidden_size)
        hidden_state = decoder_input

        new_decoder_hiddens = []
        for i in range(self.num_layers):
            hidden_state, cell_state = self.lstmCells[i](hidden_state, decoder_hiddens[i])
            new_decoder_hiddens.append((hidden_state, cell_state))

        outputs = hidden_state.unsqueeze(1)
        context, outputs = self.attention(outputs, encoder_keys, encoder_values)
        outputs = outputs.squeeze(1)

        return outputs, new_decoder_hiddens, context


    def forward(self, decoder_targets, encoder_keys, encoder_values, encoder_lens, teacher_forcing_ratio=1.0):
        # decoder_targets = (batch_size, max_target_len)
        batch_size = decoder_targets.size(0)
        max_target_len = decoder_targets.size(1)

        decoder_outputs = []

        decoder_hiddens = self._init_hidden_state(batch_size)
        #decoder_output = decoder_hiddens[self.num_layers-1][0].unsqueeze(1)
        
        mask = U.create_mask(encoder_lens).unsqueeze(1)
        self.attention.set_mask(mask)
        #context, _ = self.attention(decoder_output, encoder_keys, encoder_values)
        context = U.var(torch.zeros(batch_size, 1, self.value_size))

        use_teacher_forcing = True 
        #if np.random.random() < teacher_forcing_ratio else False

        for timestamp in range(0, max_target_len-1): 
            decoder_input = decoder_targets[:, timestamp]
            # B x 1
            step_output, decoder_hiddens, context = self.forward_step(decoder_input, decoder_hiddens, 
                                                            context, encoder_keys, encoder_values)
            #symbols = decode(step_output)
            decoder_outputs.append(step_output)

        return decoder_outputs


    def _init_hidden_state(self, batch_size):
        hiddens = []
        for i in range(self.num_layers):
            #hidden = (self.params_h0[i].expand(batch_size, -1), self.params_c0[i].expand(batch_size, -1))
            hidden = (U.var(torch.zeros(batch_size, self.hidden_size)),
                            U.var(torch.zeros(batch_size, self.hidden_size)))
            hiddens.append(hidden)
        return hiddens


def _test():
    max_input_len = 4
    max_target_len = 10

    batch_size = 4
    hidden_size = 12
    key_size = 10
    value_size = 10
    embedding_size = 20
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


