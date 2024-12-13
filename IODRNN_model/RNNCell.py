import torch

from utils.data_utils import *
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeculateGRUCell(nn.Module):
    """

    """
    def __init__(self, shape, num_units, diffusion_step, adj_mx, add=True, filter_type='Chebyshev'):
        """
        :param shape: shape of input (batch_size, (1),  num_nodes, num_features)
        :param num_units: num_units, hidden state size
        """
        super(SpeculateGRUCell, self).__init__()
        if filter_type == 'dual_direction':
            num_mat = 2 * diffusion_step + 1
        else:
            num_mat = 1
        self.add = add
        self.num_nodes = shape[1]
        self.num_feature = shape[2]
        self.units = num_units
        self.kernel_r = nn.Parameter(torch.empty(((self.num_feature + self.units) * num_mat, self.units * 2)))
        self.kernel_c = nn.Parameter(torch.empty(((self.num_feature + self.units) * num_mat, self.units)))
        nn.init.kaiming_normal_(self.kernel_c)
        nn.init.kaiming_normal_(self.kernel_r)
        self.proj_out = nn.Linear(self.units + self.num_feature, self.num_feature)
        if add:
            self.proj_basis = nn.Linear(self.num_feature, self.num_feature)
        else:
            self.proj_basis = nn.Linear(self.num_feature * 2, self.num_feature)
        self.diffusion_step = diffusion_step
        self.lambda_ = nn.Parameter(torch.randn(1,), requires_grad=True).to(device)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        if filter_type == 'Chebyshev':
            adj_mx = torch.from_numpy(get_normalized_adj(adj_mx))
            self.supports = [adj_mx]
        elif filter_type == 'dual_direction':
            adj_mx1 = torch.from_numpy(get_asy_adj(adj_mx))
            adj_mx2 = torch.from_numpy(get_asy_adj(adj_mx.T))
            self.supports = [adj_mx1, adj_mx2]

    def forward(self, input, hx, delta):
        """
        :param delta: accumulated delta across time axis with shape (B, N, D)
        :param input:(2, B, num_nodes, input_dim)
        :param hx:(B, num_nodes * units)
        :return:
                o: new input with shape(B, num_nodes, input_dim)
                new_state:(B, num_nodes * units)
        """
        _, B, N, D = input.shape
        input = input.to(device)
        hx = hx.to(device)
        y_basis, y_res = input[0], input[1]  # two (B, N, D)
        if self.add:
            # increment
            delta_value = self.relu(self.proj_basis(y_res))  # (B, N, D)
            p_t = self.sigmoid(y_res + (0.5 * torch.absolute(delta + 1e-9) ** 0.5))  # how about removing y_res ?
            delta_direction = 2 * p_t - 1
            new_delta = self.lambda_ * delta_direction * delta_value  # (B, N, D)
        else:
            y_basis = torch.cat([y_basis, y_res], dim=-1)  # shape(B, N, 2*D)
            y_basis = self.relu(self.proj_basis(y_basis))
            new_delta = torch.zeros_like(y_res)  # set to zero
        assert len(y_basis.shape) == 3

        y_basis = y_basis.reshape(B, N * D)
        value = self.sigmoid(self._gconv(y_basis, hx, self.kernel_r))  # (B ,num_nodes, 2*units)
        r, u = torch.split(value, self.units, dim=-1)  # (B ,num_nodes, units)
        r = r.reshape(-1, self.num_nodes * self.units)
        u = u.reshape(-1, self.num_nodes * self.units)
        c = self._gconv(y_basis, r * hx, self.kernel_c)  # (B ,num_nodes, units)
        c = self.tanh(c).reshape(-1, self.num_nodes * self.units)
        # new state
        new_state = torch.multiply(hx, u) + torch.multiply(c, (1.0 - u))  # (B, N * units)
        # new output with skip connection(concat)
        concat = torch.cat([new_state.reshape(B, N, self.units),
                            y_basis.reshape(B, N, D)], dim=-1)  # (B, N, D+units)
        o = self.proj_out(concat) + new_delta  # (B, N, D)
        return o, new_state, new_delta

    def _gconv(self, input, state, kernel, chebshev=True):
        """
        Graph convolution
        :param input: input with shape(B, num_nodes * num_features)
        :param state: hidden state with shape(B, num_nodes * num_units)
        :return:
        """
        batch_size = input.shape[0]
        input = input.to(device)
        state = state.to(device)
        input = input.reshape(batch_size, self.num_nodes, -1)
        state = state.reshape(batch_size, self.num_nodes, -1)
        x = [torch.cat([input, state], dim=-1)]  # [X^t; state]
        if chebshev:
            for support in self.supports:
                for i in range(self.diffusion_step):
                    x = torch.einsum("ij, jkl -> kil", [support, x[0].permute(1, 0, 2)])
        else:
            x0 = x[0]
            for support in self.supports:
                x1 = torch.einsum("ij, jkl -> kil", [support, x0.permute(1, 0, 2)])  # (B, N, input_size)
                x.append(x1)
                for i in range(1, self.diffusion_step + 1):
                    x2 = torch.einsum("ij, jkl -> kil", [support, x1.permute(1, 0, 2)])
                    x.append(x2)
            # x: (num_matrices, batch_size, num_nodes, input_size)
            x = torch.cat(x, dim=0)
            x = x.permute(1, 2, 3, 0)
            x = x.reshape(batch_size, self.num_nodes, -1)  # (batch_size, num_nodes, input_size * num_matrices)
        state = x @ kernel
        return state


class EncoderModel(nn.Module):
    def __init__(self, shape, num_units, diffusion_step, adj_mx, add):
        """
        :param shape: shape of input (batch_size, num_nodes, input_dim)
        :param num_layer:
        """
        super(EncoderModel, self).__init__()
        self.num_units = num_units
        self.diffusion_step = diffusion_step
        self.shape = shape
        self.layer = SpeculateGRUCell(
            shape=shape, num_units=self.num_units,
            diffusion_step=self.diffusion_step,
            adj_mx=adj_mx, add=add)

    def forward(self, input, hidden_state, delta):
        """
        forward through GRU layers vertically
        :param delta:
        :param input: input at each timestep of sequence with shape(2, B, N, D)
        :param hidden_state:shape(B, num_nodes, units)
        :return:
                output_t:shape (batch_size, num_nodes, input_dim)
                hidden_state: shape (batch_size, num_nodes * units)
                delta: shape (batch_size, num_nodes, input_dim)
        """
        _, batch_size, num_nodes, input_dim = input.shape
        # init h0, p0
        if hidden_state is None:
            hidden_state = torch.zeros(size=(batch_size, num_nodes * self.num_units), device=device)
            delta = torch.zeros(size=(batch_size, num_nodes, input_dim), device=device)
        output_t, hidden_state, new_delta = self.layer(input, hidden_state, delta)
        # add delta
        delta += new_delta
        return output_t, hidden_state, delta


class DecoderModel(nn.Module):
    def __init__(self, shape, num_units, out_dim, diffusion_step, adj_mx, add):
        """
        :param shape: shape of input (batch_size, num_nodes, output_dim)
        """
        super(DecoderModel, self).__init__()
        self.num_units = num_units
        self.out_dim = out_dim
        self.diffusion_step = diffusion_step
        self.layers = SpeculateGRUCell(
            shape=shape, num_units=self.num_units, diffusion_step=self.diffusion_step,
            adj_mx=adj_mx, add=add)

    def forward(self, input, hidden_state, delta):
        """
        forward through GRU layers vertically
        :param delta: past deltas
        :param input: decoder input at each timestep with shape(n_head, B, num_nodes * out_dim=num_nodes)
        :param hidden_state: (B, num_nodes * out_dim)
        :return:output:(B, num_nodes * output_dim)
                next_hidden:(B, num_nodes * units)
        """
        _, batch_size, num_nodes = input.shape
        output = input
        output = output.unsqueeze(-1)  # (n_head, batch_size, num_nodes, out_dim=1)
        output, next_hidden, new_delta = self.layers(output, hidden_state, delta)
        output = output.view(batch_size, num_nodes * self.out_dim)
        delta += new_delta
        return output, next_hidden, delta



