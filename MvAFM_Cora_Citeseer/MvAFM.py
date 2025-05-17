import torch.utils.data
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, n_fts, n_hid1, n_hid2, dropout, args):
        super(encoder, self).__init__()
        self.GCN3 = GCNLayer(n_fts, n_hid1, dropout=dropout, args=args)
        self.GCN4 = GCNLayer(n_hid1, n_hid2, dropout=dropout, args=args)
        self.dropout = dropout

    def forward(self, X_o, A_o):
        Z_a = self.GCN3(X_o, A_o, is_sparse_input=True)
        Z_a = F.dropout(Z_a, self.dropout, training=self.training)
        Z_a = self.GCN4(Z_a, A_o)
        return Z_a


class Model(nn.Module):
    def __init__(self, n_nodes, n_fts, n_hid1, n_hid2, dropout, args, train_fts_id):
        super(Model, self).__init__()
        self.dropout = dropout
        self.args = args

        # self.GCN1 = GCNLayer(n_fts, n_hid1, dropout=dropout, args=args)
        self.GCN1 = PaGCNLayer(n_fts, n_hid1, dropout=dropout, args=args, train_fts_id=train_fts_id)
        self.GCN2 = GCNLayer(n_nodes, n_hid1, dropout=dropout, args=args)
        self.GCN3 = GCNLayer(n_hid1, n_hid2, dropout=dropout, args=args)

        self.mlp = MLP(n_fts, n_hid1, n_hid2, args)

        self.GCN4 = GCNLayer(n_hid2, n_hid1, dropout=dropout, args=args)
        self.GCN5 = GCNLayer(n_hid1, n_fts, dropout=dropout, args=args)

    def forward(self, X, X_o, Adj, Diag, diff, train_fts_idx, vali_test_fts_idx, non_norm_adj):
        Z_f1, Z_f2 = self.mlp(X_o)

        X[vali_test_fts_idx].normal_(mean=0, std=1)
        X = F.dropout(X, self.dropout, training=self.training)
        index = torch.cat((train_fts_idx, vali_test_fts_idx), 0).argsort()

        Z_a = self.GCN1(X, Adj, non_norm_adj, train_fts_idx)
        # Z_a = self.GCN1(X, Adj)
        # Z_a[train_fts_idx] = 1.0 * Z_a[train_fts_idx] + 1.0 * Z_f1
        # Z_a = torch.relu(Z_a)
        Z_a = F.dropout(Z_a, self.dropout, training=self.training)
        Z_a = self.GCN3(Z_a, Adj)
        # Z_a[train_fts_idx] = 1.0 * Z_a[train_fts_idx] + 1.0 * Z_f2
        # Z_a = torch.relu(Z_a)

        Z_s = self.GCN2(Diag, diff, is_sparse_input=True)
        Z_s = F.dropout(Z_s, self.dropout, training=self.training)
        Z_s = self.GCN3(Z_s, diff)

        Z_i = Z_a
        Z_i[vali_test_fts_idx] = 1.0 * Z_a[vali_test_fts_idx] + 1.0 * Z_s[vali_test_fts_idx]

        X_hat_pre = self.GCN4(Z_i, diff)
        X_hat = self.GCN5(X_hat_pre, diff)

        A_hat = torch.mm(Z_i, torch.transpose(Z_i, 0, 1))

        return X_hat, Z_a, Z_s, A_hat


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, args):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if args.cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, x, sp_adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class PaGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, args, train_fts_id):
        super(PaGCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if args.cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.M = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(args.n_nodes, in_features).type(
            torch.cuda.FloatTensor if args.cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, x, sp_adj, non_norm_adj, train_fts_id):
        train_fts_id = train_fts_id.cuda()
        fixed_M = torch.zeros_like(self.M)
        for i in train_fts_id:
            fixed_M[i] = 1.0
        fixed_M = fixed_M.detach()
        # self.M.data.fill_(0.5)
        sigmoid_M = torch.sigmoid(self.M)
        self.M.data = sigmoid_M
        self.M.data[train_fts_id] = fixed_M[train_fts_id]

        AM = torch.spmm(non_norm_adj, self.M).pow(-1)
        AM = torch.where(torch.isinf(AM), torch.full_like(AM, 0.), AM)
        H = torch.spmm(sp_adj, self.M*x) * AM
        H_prime = torch.mm(H, self.W)
        return F.elu(H_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, n_fts, n_hid1, n_hid2, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_fts, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.dropout = args.dropout

    def forward(self, X_o):
        X_o = F.dropout(X_o, self.dropout, training=self.training)
        Z_F1 = torch.relu(self.fc1(X_o))
        Z_F1 = F.dropout(Z_F1, self.dropout, training=self.training)
        Z_F2 = self.fc2(Z_F1)
        return Z_F1, Z_F2


class MLPEncoder(nn.Module):
    def __init__(self, n_fts, n_hid1, n_hid2, args):
        super(MLPEncoder, self).__init__()
        self.encoder = MLP(n_fts, n_hid1, n_hid2, args)
        self.decoder = MLP(n_hid2, n_hid1, n_fts, args)
        self.dropout = args.dropout

    def forward(self, X_o):
        X_o = F.dropout(X_o, self.dropout, training=self.training)
        Z_F1, Z_F2 = self.encoder(X_o)
        Z_F2 = F.dropout(Z_F2, self.dropout, training=self.training)
        Z_F1, X_hat2 = self.decoder(Z_F2)
        return Z_F2, X_hat2