from RNNCell import *
from sub_layers import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeculateGCRNN(nn.Module):
    """
    Implement SpeculateGCRNN Model.
    """
    def __init__(self, encoder_shape, decoder_shape, encode_layer, decode_layer, input_time_step,
                 output_time_step, num_units, adj_mx, diffusion_step, add=True,
                 drop=0.1
                 ):
        """
        :param input_shape(encoder&decoder): (batch_size, num_nodes, dim)
        :param input_time_step:
        :param output_time_step:
        :param num_units: hidden_state_size
        :param diffusion_step: max diffusion step
        """
        super(SpeculateGCRNN, self).__init__()
        self.encode_shape = encoder_shape
        self.decode_shape = decoder_shape
        self.num_nodes = self.encode_shape[1]
        self.encode_layer = encode_layer
        self.decode_layer = decode_layer
        self.input_step = input_time_step
        self.out_dim = decoder_shape[-1]
        self.units = num_units
        self.output_step = output_time_step
        self.cl_decay_steps = 2000
        self.use_curriculum_learning = False
        self.proj_delta = nn.Linear(self.encode_shape[-1], self.decode_shape[-1])
        self.EncoderModel = EncoderModel(self.encode_shape, num_units=self.units,
                                         diffusion_step=diffusion_step, adj_mx=adj_mx, add=add)
        self.DecoderModel = DecoderModel(self.decode_shape, num_units=self.units, out_dim=self.out_dim,
                                         diffusion_step=diffusion_step, adj_mx=adj_mx, add=add)
        self.SelfAttention = SelfAttentionLayer(shape=(*self.encode_shape[:-2],
                                                       self.encode_shape[-2] * self.encode_shape[-1]))
        self.CrossAttention = CrossAttentionLayer(n_dim=self.num_nodes * (num_units + self.out_dim),
                                                  out_dim=self.num_nodes * self.out_dim)
        self.MLP_in = nn.Sequential(
            nn.Linear(self.encode_shape[-1], 12),
            nn.Linear(12, self.encode_shape[-1]),
            nn.Dropout(0.1)
        )
        self.drop_input = nn.Dropout(drop)
        self.ln1 = nn.LayerNorm(normalized_shape=[encoder_shape[-2] * encoder_shape[-1]])  # N * D
        self.ln2 = nn.LayerNorm(normalized_shape=[encoder_shape[-2], encoder_shape[-1]])  # (N, D)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder_block(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (batch_size, input_step, num_nodes, input_dim)
        :return:
                outputs: (batch_size, input_step, num_nodes, input_dim)
                hidden_state: last encoder hidden state
                delta: accumulated encoder delta
        """
        # self-attention layer
        B, T, N, D = inputs.shape
        inputs = inputs.view(*inputs.shape[:-2], -1)  # (B, T, C)
        new_inputs = self.ln1(self.SelfAttention(inputs))  # (T, 2, B, C)
        new_inputs = new_inputs.reshape(T, 2, B, N, D)
        assert new_inputs.shape[0] == self.input_step and new_inputs.shape[1] == 2

        hidden_state = None  # init state
        delta = None
        outputs_all_t = []
        hidden_states_all_t = []
        for t in range(self.input_step):
            output, hidden_state, delta = self.EncoderModel(new_inputs[t], hidden_state, delta)
            outputs_all_t.append(output)
            hidden_states_all_t.append(hidden_state)
        outputs = torch.stack(outputs_all_t).permute(1, 0, 2, 3)
        outputs = self.ln2(outputs)  # layer_norm
        assert outputs.shape == (B, T, N, D)
        hidden_states = torch.stack(hidden_states_all_t)  # (T, B, N * units)
        return outputs, hidden_states, delta

    def decoder_block(self, enc_hidden_states, enc_delta, batches_seen=None, labels=None):
        """
        :param enc_delta: enc_delta with shape (B, N, input_dim)
        :param batches_seen:
        :param enc_hidden_states: hidden states of encoder, (input_step, batch_size, num_nodes * units)
        :param labels:(output_time_step, batch_size, num_nodes, output_dim)
        :return: output:(batch_size, output_time_step, num_node * output_dim)
        """
        dec_hidden_state = enc_hidden_states[-1]
        dec_delta = self.proj_delta(enc_delta)  # (B, N, out_dim)
        start_symbol = torch.zeros(self.decode_shape, device=device)  # (B, N, out_dim)
        dec_input = start_symbol.reshape((*start_symbol.shape[:-2], -1))  # (B, N * out_dim)
        # Cross-attention
        out = dec_input
        new_inputs = self.CrossAttention(out, enc_hidden_states)  # (n_h, B, N * out_dim)
        out, dec_hidden_state, dec_delta = self.DecoderModel(new_inputs[-1], dec_hidden_state, dec_delta)
        outputs = []
        for t in range(self.output_step):
            enc_hidden_states = torch.cat(
                [enc_hidden_states, dec_hidden_state.unsqueeze(0)])  # add a new hidden state T -> T+1
            new_inputs = self.CrossAttention(out, enc_hidden_states)
            out, dec_hidden_state, dec_delta = self.DecoderModel(new_inputs[-1], dec_hidden_state, dec_delta)
            outputs.append(out)
            if self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    out = labels[t]
        outputs = torch.stack(outputs).permute(1, 0, 2)  # (B, T, N * out_dim)
        return outputs

    def forward(self, inputs, batches_seen=None, labels=None, is_train=False):
        """
        :param num_layer: number of blocks
        :param is_train: train or valid
        :param inputs:(batch_size, time_step, num_nodes, num_features)
        :param batches_seen: number of iterations * batch_size
        :param labels:
        :return:outputs:(batch_size, output_time_step, num_nodes * output_dim)
        """
        self.use_curriculum_learning = is_train
        enc_hidden_states = None
        delta = None
        outputs = torch.zeros(size=(1, )).to(device)
        # input MLP
        all_inputs = [inputs]
        inputs = self.MLP_in(inputs)    # dense-connect or MLP, which works ?
        all_inputs.append(inputs)
        # encoder
        for i in range(self.encode_layer):
            # skip-connection
            inputs_t = torch.stack(all_inputs)
            inputs = self.drop_input(torch.sum(input=inputs_t, dim=0))
            # encode
            new_inputs, enc_hidden_states, delta = self.encoder_block(inputs)
            all_inputs.append(new_inputs)

        # decode
        for j in range(self.decode_layer):
            new_outputs = self.decoder_block(enc_hidden_states, delta,
                                             batches_seen=batches_seen, labels=labels)
            outputs = new_outputs
        assert outputs.shape[1] == self.output_step
        return outputs


