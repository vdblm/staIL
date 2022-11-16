# Source: https://github.com/cemanil/LNets

import torch.nn.functional as F


import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """
    # TODO: Make sure the higher order terms can be implemented more efficiently.
    if order == 1:
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w.mm(w_t_w)
                 + (3 / 8) * w.mm(w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w.mm(w_t_w)
                 + (21 / 16) * w.mm(w_t_w_w_t_w)
                 - (5 / 16) * w.mm(w_t_w_w_t_w_w_t_w))

    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)
            w_t_w_w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w_w_t_w)

            w = (+ (315 / 128) * w
                 - (105 / 32) * w.mm(w_t_w)
                 + (189 / 64) * w.mm(w_t_w_w_t_w)
                 - (45 / 32) * w.mm(w_t_w_w_t_w_w_t_w)
                 + (35 / 128) * w.mm(w_t_w_w_t_w_w_t_w_w_t_w))

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w


def get_safe_bjorck_scaling(weight):
    bjorck_scaling = torch.tensor([np.sqrt(weight.shape[0] * weight.shape[1])]).float()
    bjorck_scaling = bjorck_scaling.type_as(weight)

    return bjorck_scaling


class DenseLinear(nn.Module):

    def __init__(self):
        super(DenseLinear, self).__init__()

    def _set_network_parameters(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        # Set weights and biases.
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _set_config(self, config):
        self.config = config

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        raise NotImplementedError

    # def project_weights(self, proj_config):
    #     with torch.no_grad():
    #         projected_weights = project_weights(self.weight, proj_config, self.config.cuda)
    #         # Override the previous weights.
    #         self.weight.data.copy_(projected_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class BjorckLinear(DenseLinear):
    def __init__(self, in_features=1, out_features=1, bias=True, config=None):
        super(BjorckLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias)

    def forward(self, x):
        # Scale the values of the matrix to make sure the singular values are less than or equal to 1.
        if self.config.model.linear.safe_scaling:
            scaling = get_safe_bjorck_scaling(self.weight)
        else:
            scaling = 1.0

        ortho_w = bjorck_orthonormalize(self.weight.t() / scaling,
                                        beta=self.config.model.linear.bjorck_beta,
                                        iters=self.config.model.linear.bjorck_iter,
                                        order=self.config.model.linear.bjorck_order).t()
        # print(ortho_w.shape)
        return F.linear(x, ortho_w, self.bias)
