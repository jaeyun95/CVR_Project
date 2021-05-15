import torch

class Activation():
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def tanh(x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def relu(x):
        return max(x,0)

    def leaky_relu(x):
        return max(0.01*x,x)

    def elu(x, alpha=0.01):
        return alpha * (torch.exp(x) + 1) if x <= 0 else x

    def prelu(x, alpha):
        return max(alpha * x, x)
