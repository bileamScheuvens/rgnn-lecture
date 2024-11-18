import einops
import torch
import math


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = torch.nn.Linear(input_size, hidden_size)
        self.W_hh = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        """
        Performs the forward pass of the RNN cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h (Tensor): Hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: Hidden state of shape (batch_size, hidden_size).
        """
        hidden = self.W_ih(x) + self.W_hh(h)
        return torch.nn.functional.tanh(hidden)


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.cell = torch.nn.RNNCell(input_size, hidden_size)
        self.cell = RNNCell(input_size, hidden_size)
        self.output_size = output_size
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Performs the forward pass of the RNN. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        state = torch.zeros(batch_size, self.hidden_size)
        for i in range(seq_len):
            state = self.cell(x[:,i], state)

        return state[:, -self.output_size:]


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # self.W_ii = torch.nn.Linear(input_size, hidden_size)
        # self.W_hi = torch.nn.Linear(hidden_size, hidden_size)
        # self.W_if = torch.nn.Linear(input_size, hidden_size)
        # self.W_hf = torch.nn.Linear(hidden_size, hidden_size)
        # self.W_ig = torch.nn.Linear(input_size, hidden_size)
        # self.W_hg = torch.nn.Linear(hidden_size, hidden_size)
        # self.W_io = torch.nn.Linear(input_size, hidden_size)
        # self.W_ho = torch.nn.Linear(hidden_size, hidden_size)
        self.W = torch.nn.Linear(input_size + hidden_size, 4*hidden_size)
        self.sigmoid = torch.nn.functional.sigmoid
        self.tanh = torch.nn.functional.tanh
        self.hidden_size = hidden_size



    def forward(self, x, h):
        """
        Performs the forward pass of the LSTM cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).

        Returns:
            Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).
        """
        h, c = h
        # i = self.sigmoid(self.W_ii(x) + self.W_hi(h))
        # f = self.sigmoid(self.W_if(x) + self.W_hf(h))
        # g = self.tanh(self.W_ig(x) + self.W_hg(h))
        # o = self.sigmoid(self.W_io(x) + self.W_ho(h))
        store = self.W(torch.cat((x,h), dim=-1))

        i = self.sigmoid(store[:, :self.hidden_size])
        f = self.sigmoid(store[:, self.hidden_size:2*self.hidden_size])
        g = self.tanh(store[:, 2*self.hidden_size:3*self.hidden_size])
        o = self.sigmoid(store[:, 3*self.hidden_size:])
        c_prime = f * c + i * g
        h_prime = o * self.tanh(c_prime)
        return c_prime, h_prime



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell = LSTMCell(input_size, hidden_size)
        self.output_size = output_size
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Performs the forward pass of the LSTM. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        h_state = torch.zeros(batch_size, self.hidden_size)
        c_state = torch.zeros(batch_size, self.hidden_size)
        for i in range(seq_len):
            h_state, c_state = self.cell(x[:,i], (h_state, c_state))

        return h_state[:, -self.output_size:]


class Conv1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.rand((hidden_size, input_size, kernel_size), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(hidden_size))
        self.hidden_size = hidden_size
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        # unfold input
        unfolded = x.unfold(dimension=2, size=self.kernel_size, step=self.stride)  # [batch_size, input_size, output_length, kernel_size]
        # rearrange dims
        unfolded = einops.rearrange(unfolded, 'b c l k -> b l (c k)')
        kernel = einops.rearrange(self.kernel, 'o c k -> o (c k)')  # [hidden_size, input_size * kernel_size]
        # apply conv
        output = torch.matmul(unfolded, kernel.T)  # [batch_size, output_length, hidden_size]
        # Rearrange output to match expected shape
        output = output.permute(0, 2, 1)  # [batch_size, hidden_size, output_length]
        return output + self.bias.reshape(1, -1, 1)

class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # tcn = torch.nn.Conv1d
        tcn = Conv1d
        layers = [tcn(input_size, hidden_size, kernel_size=3, stride=3)]
        sequence_length = 30
        n_layers = math.ceil(math.log(sequence_length/output_size, 3))
        self.expected_len = output_size * 3**n_layers
        for _ in range(n_layers-2):
            layers.append(tcn(hidden_size, hidden_size, kernel_size=3, stride=3))
        layers.append(tcn(hidden_size, 1, kernel_size=3, stride=3))
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x):
        """
        Performs the forward pass of the TCN.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        padded = torch.zeros(batch_size, self.expected_len, input_size)
        padded[:,-seq_len:, :] = x
        return self.layers(padded.transpose(-1, -2)).squeeze()


