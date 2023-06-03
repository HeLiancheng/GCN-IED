import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from pygcn.layers import FCNNLayer


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GCNIED(nn.Module):
    def __init__(self, num, nfeat, nhid, nclass, hops, threshold, aggre_range, k_connection,
                 input_droprate, hidden_droprate=0.5):
        super(GCNIED, self).__init__()
        self.prop = Parameter(torch.FloatTensor(num, nhid))
        self.reset_parameters()
        self.layer1 = FCNNLayer(nfeat, nhid)
        self.layer2 = FCNNLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.hops = hops
        self.lambd = 1.0 / (k_connection + 1.0/(threshold-0.3))
        self.threshold = threshold
        self.aggre_range = aggre_range
        self.k_connection = k_connection
        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj):
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))

        new_a = self.cosineSimilarity(x, self.threshold, adj, self.lambd)
        new_feature=self.propagation(x,new_a,self.hops)
        x_out = F.dropout(new_feature, self.hidden_droprate, training=self.training)

        x_out = self.layer2(x_out)

        return x_out, adj

    def aggregation(self, x, adj):
        original_feature = x
        for _ in range(self.aggre_range):
            original_feature = adj @ original_feature
        return original_feature

    def propagation(self, x, adj, hops):
        feature_list = [x]
        original_feature = x
        for _ in range(hops):
            original_feature = self.special_spmm(adj.indices(), adj.values(), adj.size(), original_feature)
            feature_list.append(original_feature)
        newProp = torch.unsqueeze(self.prop, dim=2)
        new_feature = torch.stack(feature_list, dim=1)
        new_feature = F.dropout(new_feature, 0.5, training=self.training)
        new_coef = torch.matmul(new_feature, newProp)
        new_coef = torch.squeeze(new_coef)
        coef = F.softmax(new_coef, dim=1)
        coef = torch.unsqueeze(coef, dim=2)
        result = new_feature * coef
        result = torch.sum(result, dim=1)

        return result

    def cosineSimilarity(self, feature, threshold, adj, alpha):
        x = self.aggregation(feature, adj)
        assert not torch.isnan(feature).any()
        x = F.normalize(x, p=2, dim=-1)
        cosine = torch.matmul(x, x.transpose(-1, -2))
        index = adj.coalesce().indices()
        value = cosine[list(index)]
        valueAdj = adj.coalesce().values()
        zero = -1e9 * torch.ones_like(value)

        eyes = torch.eye(cosine.shape[0]).cuda()
        newCos = cosine - eyes
        val, ind = torch.topk(newCos, self.k_connection, dim=-1)
        newMatirx = torch.zeros_like(cosine).scatter_(-1, ind, val)
        similarity_sparse = newMatirx.to_sparse()
        cosine = torch.where(value < threshold, zero, valueAdj)
        coef = torch.sparse.FloatTensor(index, cosine, adj.size())
        coef = coef + alpha * similarity_sparse

        coef = torch.sparse.softmax(coef, dim=1)
        coef = coef.coalesce()
        return coef
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.prop.size(1))
        self.prop.data.normal_(-stdv, stdv)