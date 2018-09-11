import torch

import torch.nn as nn

from torch.autograd import Variable


def gaussian(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * \
        Variable(logvar.data.new(mean.size()).normal_())


def to_one_hot(label, n_class):
    return Variable(
        label.data.new(label.size(0), label.size(1), n_class)
        .zero_().scatter_(2, label.data.unsqueeze(-1), 2)).float()


def get_mlp_layer(input_size, hidden_size, output_size, n_layer):
    if n_layer == 0:
        layer = nn.Linear(input_size, output_size)
    else:
        layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())

        for i in range(n_layer - 1):
            layer.add_module(
                str(len(layer)),
                nn.Linear(hidden_size, hidden_size))
            layer.add_module(str(len(layer)), nn.ReLU())

        layer.add_module(
            str(len(layer)),
            nn.Linear(hidden_size, output_size))
    return layer


def get_rnn_output(inputs, mask, cell, bidir=False, initial_state=None):
    """
    Args:
    inputs: batch_size x seq_len x n_feat
    mask: batch_size x seq_len
    initial_state: batch_size x num_layers x hidden_size
    cell: GRU/LSTM/RNN
    """
    seq_lengths = torch.sum(mask, dim=-1).squeeze(-1)
    sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
    index_sorted_idx = sorted_idx\
        .view(-1, 1, 1).expand_as(inputs)
    sorted_inputs = inputs.gather(0, index_sorted_idx.long())
    packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
        sorted_inputs, sorted_len.long().cpu().data.numpy(), batch_first=True)
    out, _ = cell(packed_seq, hx=initial_state)
    unpacked, unpacked_len = \
        torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)
    _, original_idx = sorted_idx.sort(0, descending=False)
    unsorted_idx = original_idx\
        .view(-1, 1, 1).expand_as(unpacked)
    output_seq = unpacked.gather(0, unsorted_idx.long())
    idx = (seq_lengths - 1).view(-1, 1).expand(
        output_seq.size(0), output_seq.size(2)).unsqueeze(1)
    final_state = output_seq.gather(1, idx.long()).squeeze(1)
    if bidir:
        hsize = final_state.size(-1) // 2
        final_state_fw = final_state[:, :hsize]
        final_state_bw = output_seq[:, 0, hsize:]
        final_state = torch.cat([final_state_fw, final_state_bw], dim=-1)
    return output_seq, final_state, seq_lengths


def get_rnn(rnn_type):
    if rnn_type.lower() == "lstm":
        return nn.LSTM
    elif rnn_type.lower() == "gru":
        return nn.GRU
    elif rnn_type.lower() == "rnn":
        return nn.RNN
    else:
        NotImplementedError("invalid rnn type: {}".format(rnn_type))


def kl_normal2_normal2(mean1, log_var1, mean2, log_var2):
    return 0.5 * log_var2 - 0.5 * log_var1 + \
        (torch.exp(log_var1) + (mean1 - mean2) ** 2) / \
        (2 * torch.exp(log_var2) + 1e-10) - 0.5


def compute_KL_div(mean_q, log_var_q, mean_prior, log_var_prior):
    kl_divergence = kl_normal2_normal2(
        mean_q, log_var_q, mean_prior, log_var_prior)
    return kl_divergence


def compute_KL_div2(mean, log_var):
    return - 0.5 * (1 + log_var - mean.pow(2) - log_var.exp())


class char_rnn(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_dim, hidden_size):
        super(char_rnn, self).__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.char_cell = get_rnn(rnn_type)(
            input_size=embed_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True)

    def forward(self, chars, chars_mask, data_mask):
        """
        chars: batch size x seq len x # chars
        chars_mask: batch size x seq len x # chars
        data_mask: batch size x seq len
        """
        char_output = []
        batch_size, seq_len, char_len = chars.size()
        for i in range(batch_size):
            char = chars[i, :, :]
            char_mask = chars_mask[i, :, :]
            # trim off extra padding
            word_length = int(data_mask[i, :].sum())
            n_char = int(char_mask.sum(-1).max())
            char = char[:word_length, :n_char]
            char_mask = char_mask[:word_length, :n_char]
            # char: word length x char len
            char_vec = self.char_embed(char.long())
            _, final_state, _ = get_rnn_output(
                char_vec, char_mask, self.char_cell, bidir=True)
            # final_state: word length x hidden size
            padding = final_state.data.new(
                seq_len - word_length, 2 * self.hidden_size).zero_()
            if padding.dim():
                final_output = torch.cat(
                    [final_state, Variable(padding)], dim=0)
            else:
                final_output = final_state
            char_output.append(final_output)
        char_outputs = torch.stack(char_output, 0)
        # batch size x seq len x hidden size
        return char_outputs
