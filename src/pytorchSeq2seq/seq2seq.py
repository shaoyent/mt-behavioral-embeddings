import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

class Seq2seqBeh(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax, beh_chop=True):
        super(Seq2seqBeh, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

        input_size = (encoder.hidden_size * 2) if encoder.bidirectional else encoder.hidden_size
        input_size = (input_size // 2) if beh_chop else input_size
        self.beh_size = input_size

        self.beh = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )


    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        batch_size = encoder_outputs.size(0)

        # Only half is used for behavior analysis
        if False : 
        # try : 
           bidirectional = self.encoder.bidirectional  
           if bidirectional :  
               encoder_forward_semi = encoder_hidden[-2, :, (self.encoder.hidden_size // 2):].contiguous().view(batch_size,-1)
               encoder_backward_semi = encoder_hidden[-1,:, (self.encoder.hidden_size // 2):].contiguous().view(batch_size,-1)
               beh_semi = torch.cat((encoder_forward_semi, encoder_backward_semi), 1)
           else : 
               beh_semi = encoder_hidden[-1:,:,int(self.encoder.hidden_size / 2):].permute(1,0,2).contiguous().view(batch_size,-1)
           assert batch_size == beh_semi.size(0), "Input to beh has incorrect batch size {} ({})".format(batch_size, beh_semi.size())
           assert self.beh_size == beh_semi.size(1), "Input to beh has incorrect feature size {} ({})".format(self.beh_size, beh_semi.size())

        else :
        # except :    
           # beh_semi = encoder_hidden[-1:,:,int(self.encoder.hidden_size / 2):].permute(1,0,2).contiguous().view(batch_size,-1)
           beh_semi = encoder_hidden[-2:,:,:].permute(1,0,2).contiguous().view(batch_size,-1)
            
        beh_outputs = self.beh(beh_semi)

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)

        result[-1]['beh'] = F.log_softmax(beh_outputs)

        return result
