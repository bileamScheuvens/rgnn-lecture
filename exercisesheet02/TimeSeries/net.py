import einops
import torch


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.inputLin = torch.nn.Linear(input_size, hidden_size)
        self.hiddenLin = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        """
        Performs the forward pass of the RNN cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h (Tensor): Hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: Hidden state of shape (batch_size, hidden_size).
        """
        hiddenNext = torch.tanh(self.inputLin(x) + self.hiddenLin(h))
        return hiddenNext
        


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.rnnCell = RNNCell(input_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size, output_size)
        self.hiddenSize = hidden_size

    def forward(self, x):
        """
        Performs the forward pass of the RNN. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batchSize, seqLength, _ = x.size()
        
        hiddenT = torch.zeros(batchSize, self.hiddenSize)
        
        for t in range(seqLength):
            xT = x[:, t, :]
            hiddenT = self.rnnCell(xT, hiddenT)
            
        return self.lin(hiddenT)


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linTransform = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h):
        """
        Performs the forward pass of the LSTM cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).

        Returns:
            Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).
        """
        hiddenT, cellT = h
        combined = torch.cat([x, hiddenT], dim=1)
        gates = self.linTransform(combined)
        inGate, forgetGate, cellGate, outGate = gates.chunk(4, dim=1)
        
        inGate = torch.sigmoid(inGate)
        forgetGate = torch.sigmoid(forgetGate)
        outGate = torch.sigmoid(outGate)
        cellGate = torch.tanh(cellGate)

        cellNext = forgetGate * cellT + inGate * cellGate

        hiddenNext = outGate * torch.tanh(cellNext)
        
        return hiddenNext, cellNext



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstmCell = LSTMCell(input_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size, output_size)
        self.hiddenSize = hidden_size

    def forward(self, x):
        """
        Performs the forward pass of the LSTM. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batchSize, seqLength, _ = x.size()
        hiddenT = torch.zeros(batchSize, self.hiddenSize)
        cellT = torch.zeros(batchSize, self.hiddenSize)
        
        for t in range(seqLength):
            xT = x[:, t, :]
            hiddenT, cellT = self.lstmCell(xT, (hiddenT, cellT))
            
        return self.lin(hiddenT)


class Conv1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super().__init__()
        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.kernelSize = kernel_size
        self.stride = stride
        
        self.weight = torch.nn.Parameter(torch.rand(hidden_size, input_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.rand(hidden_size))

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        batchSize, _, seqLength = x.size()
        
        outputLen = (seqLength - self.kernelSize) // self.stride + 1
        
        output = torch.zeros(batchSize, self.hiddenSize, outputLen)
        
        for i in range(outputLen):
            start = i * self.stride
            end = start + self.kernelSize
            
            slidingWindow = x[:, :, start: end]
            for j in range(self.hiddenSize):
                output[:, j, i] = torch.sum(slidingWindow * self.weight[j, :, :], dim=(1,2)) + self.bias[j]

        return output

class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = Conv1d(input_size, hidden_size, kernel_size=3, stride=3, )
        self.conv2 = Conv1d(hidden_size, hidden_size, kernel_size=3, stride=3,)
        self.conv3 = Conv1d(hidden_size, hidden_size, kernel_size=3, stride=3, )


    def forward(self, x):
        """
        Performs the forward pass of the TCN.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        x = x.transpose(1,2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # x = torch.relu(self.conv3(torch.nn.functional.pad(x, (0,7,0,0), mode='constant',value =0)))
        # x = torch.relu(self.conv4(torch.nn.functional.pad(x, (0,16,0,0), mode='constant',value =0)))


        return x.squeeze(-1)
