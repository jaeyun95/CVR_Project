import torch
import torch.nn as nn
from util.activation import Activation

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input_data):
        x = self.fc(input_data)
        self.dropout(x)
        logit = self.sigmoid(x)
        return logit
