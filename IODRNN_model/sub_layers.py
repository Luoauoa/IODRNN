import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """
    A similar Two-head self-attention layer
    """

    def __init__(self, shape, n_head=2, drop=0.2, mask=False):
        """
        :param shape: shape of input sequence(B, time_step, num_nodes * input_dim)
        """
        super(SelfAttentionLayer, self).__init__()
        self.seq_len = shape[1]
        self.num_dim = shape[-1]
        self.n_head = n_head
        self.query = nn.Linear(self.num_dim, self.n_head * self.num_dim)
        self.key = nn.Linear(self.num_dim, self.n_head * self.num_dim)
        self.value = nn.Linear(self.num_dim, self.n_head * self.num_dim)
        # regularization
        self.att_drop = nn.Dropout(drop)
        self.resid_drop = nn.Dropout(drop)
        # output projection
        self.projection = nn.Linear(self.num_dim, self.num_dim)
        self.register_buffer("mask", torch.tril(torch.ones(self.seq_len, self.seq_len))
                             .view(1, 1, self.seq_len, self.seq_len))

    def forward(self, inputs, mask=False):
        """
        :param mask: Do mask if encoder else don't
        :param inputs: input sequence with shape (batch_size, time_step, num_nodes * input_dim)
        :return: y: y_basis, y_res shape(time_step, n_head=2, batch_size, num_nodes * input_dim)
        """
        B, T, C = inputs.shape  # C = num_dim
        C = self.n_head * C
        # Multi-head
        query = self.query(inputs).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)  # (B, n_h, T, n_dim)
        key = self.key(inputs).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)
        value = self.value(inputs).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)

        att_score = (query @ key.transpose(-2, -1)) * (1.0 / np.sqrt(key.shape[-1]))  # (B, n_h, T, T)
        if mask:
            att_score = att_score.masked_fill(self.mask[:, :, T, T] == 0, -1e-10)
        att_score = F.softmax(att_score, dim=-1)
        y = self.att_drop(att_score @ value)
        y = self.resid_drop(self.projection(y) + y).permute(2, 1, 0, 3)  # skip-connection (T, n_h, B, n_dim)
        return y


class CrossAttentionLayer(nn.Module):
    """
    A similar Two-head cross-attention layer
    """
    def __init__(self, n_dim, out_dim, n_head=2, drop=0.2):
        """
        :param n_dim: total dim = units + output_dim
        :param out_dim: num_nodes * output_dim
        :param n_head: number of heads
        :param drop: dropout prob
        """
        super(CrossAttentionLayer, self).__init__()
        self.n_dim = n_dim  # total dim: N * (output_dim + units)
        self.out_dim = out_dim  # num_nodes * out_dim
        self.n_head = n_head
        self.key = nn.Linear(n_dim, n_head * out_dim)  # n_dim -> n_h * out_dim
        self.query = nn.Linear(n_dim, n_head * out_dim)
        self.value = nn.Linear(n_dim, n_head * out_dim)
        self.att_drop = nn.Dropout(drop)
        self.resid_drop = nn.Dropout(drop)
        self.projection = nn.Linear(out_dim, out_dim)

    def forward(self, inputs: torch.Tensor, hidden_states):
        """
        :param inputs: output of the last decoder with shape (B, N * output_dim)
        :param hidden_states: enc_hidden_states with shape (T, B, N * units)
        :return:
        """
        T, B, _ = hidden_states.shape
        C = self.n_head * self.out_dim
        inputs = inputs.expand(T, B, inputs.shape[-1])  # (T, B, out_dim)
        concat = torch.cat((hidden_states, inputs), dim=-1)  # (T, B, n_dim)
        key = self.key(concat).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)  # (B, n_h, T, out_dim)
        query = self.query(concat).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)
        value = self.value(concat).view(B, T, C // self.n_head, self.n_head).permute(0, 3, 1, 2)
        att_score = query @ key.transpose(-2, -1) * (1.0 / np.sqrt(key.shape[-1]))  # (B, n_h, T, T)
        att_score = F.softmax(att_score, dim=-1)
        y = self.att_drop(att_score @ value)   # (B, n_h, T, out_dim)
        y = self.resid_drop(self.projection(y) + y).permute(2, 1, 0, 3)  # (T, n_h, B, out_dim)
        return y



