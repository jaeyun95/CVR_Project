import numpy as np
import torch

class Adam:
    def __init__(self, lr=0.01, beta1 = 0.9, beta2 = 0.999, cuda=False):
        if cuda:
            self.lr = lr
            self.epsilon = torch.tensor(10e-8)
            self.beta1 = torch.tensor(beta1)
            self.beta2 = torch.tensor(beta2)
        else:
            self.lr = lr.cuda()
            self.epsilon = torch.tensor(10e-8).cuda()
            self.beta1 = torch.tensor(beta1).cuda()
            self.beta2 = torch.tensor(beta2).cuda()
        self.grad = {}
        self.cuda = cuda
        self.m = {}
        self.v = {}

    def step(self, key, param, grad, loss):
        m_h, v_h = {}, {}
        if key not in self.m.keys() or key not in self.v.keys():
            self.m, self.v, self.grad = {}, {}, {}
            if self.cuda:
                self.m[key] = torch.zeros(param.shape).cuda()
                m_h[key] = torch.zeros(param.shape).cuda()
                self.v[key] = torch.zeros(param.shape).cuda()
                v_h[key] = torch.zeros(param.shape).cuda()
            else:
                self.m[key] = torch.zeros(param.shape)
                m_h[key] = torch.zeros(param.shape)
                self.v[key] = torch.zeros(param.shape)
                v_h[key] = torch.zeros(param.shape)

        self.grad[key] = grad
        self.m[key] += (self.beta1 * self.m[key]) + ((1 - self.beta1) * self.grad[key])
        self.v[key] += (self.beta2 * self.v[key]) + ((1 - self.beta2) * (self.grad[key]**2))
        m_h[key] = self.m[key] / (1 - self.beta1)
        v_h[key] = self.v[key] / (1 - self.beta2)
        param -= self.lr * m_h[key] / torch.sqrt(v_h[key] + self.epsilon)
        return param