import os
import imp
import torch
import torch.nn as nn
from torch.autograd import Function
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file, path, description = imp.find_module('libpermutohedral_lattice', [os.path.join(BASE_DIR, 'lib')])
C_ = imp.load_module('C_', file, path, description)


class PermutohedralLatticeFunction(Function):
    """


    :param
           position_in:  (b, n_points_in, d)
           position_out: (b, n_points_out, d) or None
           features_in:  (b, df_in, n_points_in)
           weights:      (df_out, df_in, n_neighbors)
    :return:
           features_out: (b, df_out, n_points_out)
           norm:         (b, 1, n_points_out)
    """

    @staticmethod
    def forward(ctx, features_in, weights, position_in, position_out,
                neighbor_size=1, skip_conv=False):
        if position_out is None:
            features_out, lattices_in, offset_in, barycentric_in, conv_neighbors, norm = \
                C_.permutohedral_lattice_keep_position(position_in, features_in, weights, neighbor_size, skip_conv)
            ctx.save_for_backward(weights, lattices_in, offset_in, barycentric_in, conv_neighbors, norm)
            ctx.keep_position = True

        else:
            features_out, lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm = \
                C_.permutohedral_lattice(position_in, position_out, features_in, weights, neighbor_size, skip_conv)
            ctx.save_for_backward(weights, lattices_in, offset_in, offset_out, barycentric_in, barycentric_out,
                                  conv_neighbors, norm)
            ctx.keep_position = False
        ctx.neighbor_size = neighbor_size
        ctx.skip_conv = skip_conv
        return features_out, norm

    @staticmethod
    def backward(ctx, grad_out, *args):
        if ctx.keep_position:
            weights, lattices_in, offset_in, barycentric_in, conv_neighbors, norm = ctx.saved_tensors
            grad_in, grad_weights = \
                C_.permutohedral_lattice_keep_position_grad(grad_out.contiguous(), weights, lattices_in, offset_in,
                                                            barycentric_in, conv_neighbors, norm, ctx.neighbor_size,
                                                            ctx.skip_conv)
        else:
            weights, lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm = ctx.saved_tensors
            grad_in, grad_weights = \
                C_.permutohedral_lattice_grad(grad_out.contiguous(), weights, lattices_in, offset_in, offset_out,
                                              barycentric_in, barycentric_out, conv_neighbors, norm, ctx.neighbor_size,
                                              ctx.skip_conv)
        return grad_in, grad_weights, None, None, None, None


permutohedral_lattice_ = PermutohedralLatticeFunction.apply


class PermutohedralLattice(nn.Module):

    def __init__(self, df_in, df_out, d, pos_lambda, n=1, bias=True, skip_conv=False):
        super(PermutohedralLattice, self).__init__()
        n_neighbors = int(pow((n + 1), (d + 1)) - pow(n, (d + 1)))

        self.df_in = df_in
        self.df_out = df_out
        self.d = d
        self.pos_lambda = pos_lambda
        self.n = n
        self.skip_conv = skip_conv
        self.n_neighbors = n_neighbors
        self.weights = nn.Parameter(torch.Tensor(df_out, df_in, n_neighbors))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, df_out, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(0, 0.01)
        if self.bias is not None:
            self.bias.data.fill_(0)
        return

    def forward(self, features_in, position_in, position_out=None):
        position_in = position_in * self.pos_lambda
        assert features_in.size(1) == self.df_in
        if position_out is None:
            assert position_in.size(2) == self.d
        else:
            assert position_in.size(2) == position_out.size(2) == self.d
            position_out = position_out * self.pos_lambda

        features_out, norm = \
            permutohedral_lattice_(features_in, self.weights, position_in, position_out, self.n, self.skip_conv)
        features_out = features_out / norm.detach()
        if self.bias is not None:
            features_out = features_out + self.bias
        return features_out


if __name__ == '__main__':
    # speed test
    m = PermutohedralLattice(512, 256, 3, 1, True)
    start = time.time()
    for i in range(10):
        init_time = time.time()
        position_in = torch.randn([32, 2048, 3]).cuda()
        position_out = torch.randn([32, 2048, 3]).cuda()
        features_in = torch.randn([32, 512, 2048]).cuda()
        features_in.requires_grad = True
        start = start + time.time() - init_time
        features_out = m(features_in, position_in, position_out)
        loss = features_out.sum()
        loss.backward()
        print(i)
    end = time.time()
    print((end - start) / 10)
