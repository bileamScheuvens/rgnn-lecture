import random
import math

# Sigmoid function
def sigmoid(x):
    # Prevent overflow
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# Tanh function
def tanh(x):
    return math.tanh(x)

# Binary Cross Entropy Loss with logits
def bce_with_logits_loss(output, target):
    # Numerically stable implementation
    if output >= 0:
        loss = (1 - target) * output + math.log(1 + math.exp(-output))
    else:
        loss = -target * output + math.log(1 + math.exp(output))
    return loss

class FeedForwardNetwork:
    def __init__(self, input_size=2, hidden_size=16, learning_rate=0.01, momentum=0.9, weight_decay=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Xavier Initialization for weights
        def xavier_init(fan_in, fan_out):
            limit = math.sqrt(6 / (fan_in + fan_out))
            return [[random.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]

        # Initialize weights using Xavier initialization
        self.W1 = xavier_init(self.input_size, self.hidden_size)
        self.W2 = xavier_init(self.hidden_size, self.hidden_size)
        self.W3 = xavier_init(self.hidden_size, 1)  # Output layer has one neuron

        # Initialize biases to zero
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.b2 = [0.0 for _ in range(self.hidden_size)]
        self.b3 = [0.0]  # Bias for output neuron

    def linear(self, inputs, weights, biases):
        outputs = []
        for i in range(len(weights)):
            activation = biases[i]
            for j in range(len(inputs)):
                activation += inputs[j] * weights[i][j]
            outputs.append(activation)
        return outputs

    def sigmoid(self, x):
        return sigmoid(x)

    def tanh(self, x):
        return tanh(x)

    def sigmoid_derivative(self, z):
        return sigmoid(z) * (1-sigmoid(z))

    def tanh_derivative(self, z):
        return 1/(math.cosh(z)**2)
        # return 1 - tanh(z)**2

    def forward(self, x):
        # Store inputs and outputs for backward pass
        self.x0 = x  # Input layer
        self.z1 = self.linear(self.x0, self.W1, self.b1)

        # Apply activation function
        self.a1 = [self.tanh(z) for z in self.z1]

        # Compute output of layer 2
        self.z2 = self.linear(self.a1, self.W2, self.b2)

        # Apply activation function
        self.a2 = [self.sigmoid(z) for z in self.z2]

        # Compute output of layer 3
        self.z3 = self.linear(self.a2, self.W3, self.b3)
        self.output = self.z3[0]  # Scalar output
        return self.output

    def zero_grad(self):
        # zero accumulated version of the gradients
        self.dW1_acc = [[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.db1_acc = [0.0 for _ in range(self.hidden_size)]
        self.dW2_acc = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        self.db2_acc = [0.0 for _ in range(self.hidden_size)]
        self.dW3_acc = [[0.0 for _ in range(self.hidden_size)]]
        self.db3_acc = [0.0]


    def backward(self, target):
        # Compute gradients for the loss function
        error = sigmoid(self.output) - target

        # Gradients for W3 and b3
        dW3 = [[error * self.a2[j] for j in range(self.hidden_size)]]
        db3 = [error]

        # Backpropagate to layer 2
        d2 = [error * self.W3[0][i] * self.sigmoid_derivative(self.z2[i]) for i in range(self.hidden_size)]

        # Gradients for W2 and b2
        dW2 = [[d2[i] * self.a1[j] for i in range(self.hidden_size)] for j in range(self.hidden_size)]
        db2 = d2

        # Backpropagate to layer 1
        d1 = [error * self.W2[0][i] * self.tanh_derivative(self.z1[i]) for i in range(self.hidden_size)]

        # Gradients for W1 and b1
        dW1 = [[d1[i] * self.x0[j] for j in range(2)] for i in range(self.hidden_size)]
        db1 = d1

        # Accumulate gradients
        self.dW1_acc = [[self.dW1_acc[i][j] + dW1[i][j] for j in range(self.input_size)] for i in range(self.hidden_size)]
        self.db1_acc = [self.db1_acc[i] + db1[i] for i in range(self.hidden_size)]
        self.dW2_acc = [[self.dW2_acc[i][j] + dW2[i][j] for j in range(self.hidden_size)] for i in range(self.hidden_size)]
        self.db2_acc = [self.db2_acc[i] + db2[i] for i in range(self.hidden_size)]
        self.dW3_acc = [[self.dW3_acc[i][j] + dW3[i][j] for j in range(self.hidden_size)] for i in range(1)]
        self.db3_acc = [self.db3_acc[i] + db3[i] for i in range(1)]
        pass

    def update_weights(self):
        # initialize momentum
        if not hasattr(self, 'v'):
            self.v = [[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)]
            self.v2 = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
            self.v3 = [[0.0 for _ in range(self.hidden_size)] for _ in range(1)]
            self.vb1 = [0.0 for _ in range(self.hidden_size)]
            self.vb2 = [0.0 for _ in range(self.hidden_size)]
            self.vb3 = [0.0]

        # Update W3 and b3
        for i in range(len(self.W3)):
            for j in range(len(self.W3[0])):
                self.v3[i][j] = self.momentum * self.v3[i][j] - self.learning_rate * self.dW3_acc[i][j]
                self.W3[i][j] += self.v3[i][j]
                self.W3[i][j] -= self.weight_decay * self.learning_rate * self.W3[i][j] 

        for i in range(len(self.b3)):
            self.vb3[i] = self.momentum * self.vb3[i] - self.learning_rate * self.db3_acc[i]
            self.b3[i] += self.vb3[i]
            self.b3[i] -= self.weight_decay * self.learning_rate * self.b3[i]

        # Update W2 and b2
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.v2[i][j] = self.momentum * self.v2[i][j] - self.learning_rate * self.dW2_acc[i][j]
                self.W2[i][j] += self.v2[i][j]
                self.W2[i][j] -= self.weight_decay * self.learning_rate * self.W2[i][j]

        for i in range(len(self.b2)):
            self.vb2[i] = self.momentum * self.vb2[i] - self.learning_rate * self.db2_acc[i]
            self.b2[i] += self.vb2[i]
            self.b2[i] -= self.weight_decay * self.learning_rate * self.b2[i]

        # Update W1 and b1
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.v[i][j] = self.momentum * self.v[i][j] - self.learning_rate * self.dW1_acc[i][j]
                self.W1[i][j] += self.v[i][j]
                self.W1[i][j] -= self.weight_decay * self.learning_rate * self.W1[i][j]

        for i in range(len(self.b1)):
            self.vb1[i] = self.momentum * self.vb1[i] - self.learning_rate * self.db1_acc[i]
            self.b1[i] += self.vb1[i]
            self.b1[i] -= self.weight_decay * self.learning_rate * self.b1[i]


