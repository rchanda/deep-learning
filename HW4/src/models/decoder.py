import numpy as np

import torch
import torch.nn as nn

from models.attention import Attention
import data.utils as U
import constants as C
import pdb

class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, key_size, value_size, num_layers, max_len):
        super(DecoderRNN, self).__init__()
        self.value_size = value_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.max_len = max_len

        self.params_h0 = nn.ParameterList(
           [nn.Parameter(torch.zeros(1, self.hidden_size)).float() for i in range(self.num_layers)])
        self.params_c0 = nn.ParameterList(
           [nn.Parameter(torch.zeros(1, self.hidden_size)).float() for i in range(self.num_layers)])

        self.lstmCells = nn.ModuleList(
            [nn.LSTMCell(input_size=value_size+embedding_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)])

        self.attention = Attention(self.hidden_size, key_size, value_size, output_size)
        self.attention.projection.weight = self.embedding.weight


    def forward_step(self, input_var, decoder_hiddens, context, encoder_keys, encoder_values):
        # input_var = (batch_size)
        
        embedding = self.embedding(input_var)
        # embedding = (batch_size, embedding_size)
        #pdb.set_trace()
        hidden_state = torch.cat((embedding, context), dim=1)
        # hidden_state = (batch_size, embedding_size+value_size)

        new_decoder_hiddens = []
        for i in range(self.num_layers):
            hidden_state, cell_state = self.lstmCells[i](hidden_state, decoder_hiddens[i])
            new_decoder_hiddens.append((hidden_state, cell_state))

        context, outputs = self.attention(hidden_state, encoder_keys, encoder_values)

        return outputs, new_decoder_hiddens, context


    def forward(self, decoder_targets, encoder_keys, encoder_values, encoder_lens, teacher_forcing_ratio):
        ret_dict = dict()

        # decoder_targets = (batch_size, max_target_len)
        batch_size = decoder_targets.size(0)
        max_target_len = decoder_targets.size(1) if decoder_targets.size(1) > 1 else self.max_len

        decoder_hiddens = self._init_hidden_state(batch_size)
        #decoder_output = decoder_hiddens[self.num_layers-1][0].unsqueeze(1)
        
        mask = U.create_mask(encoder_lens).unsqueeze(1)
        self.attention.set_mask(mask)

        context = U.var(torch.zeros(batch_size, self.value_size).float())
        print(teacher_forcing_ratio)
        use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_target_len] * batch_size)

        def decode(step, step_output):
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(C.EOS_TOKEN_IDX)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols


        decoder_input = decoder_targets[:, 0]
        # B x 1

        for timestamp in range(1, max_target_len): 
            step_output, decoder_hiddens, context = self.forward_step(decoder_input, decoder_hiddens, 
                                                            context, encoder_keys, encoder_values)
            decoder_outputs.append(step_output)

            if use_teacher_forcing:
                decoder_input = decoder_targets[:, timestamp]
            else:
                symbols = decode(timestamp, step_output)
                decoder_input = symbols.squeeze(1)

        ret_dict[C.KEY_SEQUENCE] = sequence_symbols
        ret_dict[C.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, ret_dict


    def _init_hidden_state(self, batch_size):
        hiddens = []
        for i in range(self.num_layers):
            hidden = (self.params_h0[i].expand(batch_size, -1), self.params_c0[i].expand(batch_size, -1))
            #hidden = (U.var(torch.zeros(batch_size, self.hidden_size)),
            #               U.var(torch.zeros(batch_size, self.hidden_size)))
            hiddens.append(hidden)
        return hiddens


def _test():
    max_input_len = 4
    max_target_len = 10

    batch_size = 4
    hidden_size = 20
    key_size = 10
    value_size = 10
    embedding_size = 10
    output_size = 33
    num_layers = 3
    teacher_forcing_ratio = 1.0

    decoder_targets = U.var(torch.ones(batch_size, max_target_len).long())

    outputs = U.var(torch.randn(batch_size, 1, hidden_size))
    encoder_lens = list(range(1, batch_size+1))
    encoder_keys = U.var(torch.randn(max_input_len, batch_size, key_size))
    encoder_values = U.var(torch.randn(max_input_len, batch_size, value_size))

    decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers, 10)
    
    if U.use_cuda():
        decoder = decoder.cuda()
    
    decoder_outputs = decoder(decoder_targets, encoder_keys, encoder_values, encoder_lens, teacher_forcing_ratio)


if __name__ == "__main__":
    _test()


