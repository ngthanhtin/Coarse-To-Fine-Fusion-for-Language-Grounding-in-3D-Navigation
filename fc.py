from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# class FCNet(nn.Module):
#     """Simple class for non-linear fully connect network
#     """
#     def __init__(self, dims, act='PReLU', dropout=0):
#         super(FCNet, self).__init__()

#         layers = []
#         for i in range(len(dims)-2):
#             in_dim = dims[i]
#             out_dim = dims[i+1]
#             if 0 < dropout:
#                 layers.append(nn.Dropout(dropout))
#             layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
#             if ''!=act:
#                 layers.append(getattr(nn, act)())

#         if 0 < dropout:
#             layers.append(nn.Dropout(dropout))
#         # layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
#         layers.append(nn.Linear(dims[-2], dims[-1]))
#         if ''!=act:
#             layers.append(getattr(nn, act)())
        
#         self.main = nn.Sequential(*layers)
#         # self.weight_norm = weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None)
#         # self.linear = nn.Linear(dims[-2], dims[-1])
#         # self.dropout = nn.Dropout(dropout)
#         # self.activation_func = nn.PReLU()


#     def forward(self, x):
#         return self.main(x)
#         # x = self.linear(x)
#         # x = self.dropout(x)
#         # x = self.activation_func(x)
#         return x


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='PReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            # layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
            #                           dim=None))

            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias))
        # layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
        #                           dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
