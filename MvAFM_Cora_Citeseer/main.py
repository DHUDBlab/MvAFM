import argparse

import torch
from torch import optim
from MvAFM import Model, MLPEncoder
from utils import *
import warnings
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amap')
parser.add_argument('--method_name', type=str, default='Model')
parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
parser.add_argument('--update', type=int, default=30)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--hidden2', type=int, default=64)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--neg_times', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lambda_xr', type=float, default=10)
parser.add_argument('--lambda_ar', type=float, default=0.5)
parser.add_argument('--lambda_cr', type=float, default=10000)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--train_fts_ratio', type=float, default=0.4)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == "cora":
    args.n_nodes = 2708
    args.feat = 1433
    args.hidden1 = 128
    args.epoch = 500
    args.n_cluster = 7
    args.pre_epoch = 200
elif args.dataset == "citeseer":
    args.n_nodes = 3327
    args.feat = 3703
    args.hidden1 = 512
    args.epoch = 400
    args.n_cluster = 6
elif args.dataset == "amap":
    args.n_nodes = 7650
    args.feat = 745
    args.hidden1 = 512
    args.epoch = 1000
    args.n_cluster = 8
elif args.dataset == "amac":
    args.n_nodes = 13752
    args.feat = 767
    args.hidden1 = 512
    args.epoch = 8000
else:
    print("Error!")

if __name__ == "__main__":
    adj, norm_adj, diff, true_features, node_labels, non_norm_adj = load_data(args)
    diff = diff.cuda()
    non_norm_adj = non_norm_adj.cuda()
    # norm_adj, 对角线上为1其余为0的矩阵, true_features, norm_adj
    Adj, Diag, True_features, A_temp = input_matrix(args, adj, norm_adj, true_features)
    # 训练集、验证集、测试集的节点下标
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)
    # observed data
    X_o, A_o = observed_data_process(args, adj, train_id, true_features)

    # BCE loss
    fts_loss_func, pos_weight_tensor, neg_weight_tensor = loss_weight(args, true_features, train_id)
    # norm_adj的零值/非零值的下标和值
    neg_indices, neg_values, pos_values, pos_indices = adj_loss_process(args, norm_adj)

    model, optimizer = model_optimizer(args, Model, optim, train_id)

    print("114*514")
    node_labels = node_labels.cpu().numpy()

    # pre_model, pre_optimizer = pre_model_optimizer(args, MLPEncoder, optim)
    # for epoch in range(1, args.pre_epoch + 1):
    #     pre_model.train()
    #     pre_optimizer.zero_grad()
    #     _, X_hat2 = pre_model(X_o)
    #     L = pre_train_loss(args, X_hat2, True_features, fts_loss_func, train_id, pos_weight_tensor, neg_weight_tensor)
    #     L.backward()
    #     pre_optimizer.step()
    #     if (epoch + 1) % 100 == 0:
    #         print(L)
    # pre_model.eval()
    # encoder_weights1 = pre_model.encoder.fc1.weight.data
    # encoder_weights2 = pre_model.encoder.fc2.weight.data
    # model.mlp.fc1.weight.data.copy_(encoder_weights1)
    # model.mlp.fc2.weight.data.copy_(encoder_weights2)

    best = 0.0
    best_mse = 10000.0
    bad_counter = 0
    best_epoch = 0
    L_list = []
    eva_values_list = []

    if True_features is true_features:
        print("True_feature 和 true_feature 是同一个副本")
    else:
        print("True_feature 和 true_feature 不是同一个副本")

    for epoch in tqdm(range(0, args.epoch)):
        model.train()
        # 梯度归零
        optimizer.zero_grad()
        X_hat, Z_a, Z_s, A_hat = model(True_features, X_o, Adj, Diag, diff, train_id, vali_test_id, non_norm_adj)
        L = train_loss(args, X_hat, True_features, fts_loss_func, train_id, pos_weight_tensor, neg_weight_tensor,
                       A_hat, pos_indices, neg_indices, pos_values, neg_values,
                       Z_a, Z_s, diff, vali_test_id)
        L.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                X_hat, Z_a, Z_s, A_hat = model(True_features, X_o, Adj, Diag, diff, train_id, vali_test_id, non_norm_adj)
            gene_fts = X_hat[vali_id].cpu().numpy()
            gt_fts = true_features[vali_id].cpu().numpy()
            avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=args.topK_list[2])
            eva_values_list.append(avg_recall)
            if eva_values_list[-1] > best:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))
                best = eva_values_list[-1]

                # Z_temp = Z_a
                # Z_temp[vali_test_id] = 1.0 * Z_a[vali_test_id] + 1.0 * Z_s[vali_test_id]
                # Z_temp = X_hat
                # # 创建KMeans对象并设置聚类中心的数量
                # kmeans = KMeans(n_clusters=8, random_state=0).fit(Z_temp.cpu().numpy())
                #
                # # 获取聚类标签和聚类中心
                # temp_labels = kmeans.labels_
                # temp_centers = kmeans.cluster_centers_
                #
                # print("-------------------------------------------------------------")
                # print("Clustering Results: ")
                # print("acc: {:.8f}\t\tnmi: {:.8f}\t\tari: {:.8f}".
                #       format(cluster_acc(node_labels, temp_labels), nmi(node_labels, temp_labels),
                #              ari(node_labels, temp_labels)))
                # print("-------------------------------------------------------------")

    test_model(args, model, True_features, true_features, X_o, Adj, Diag, diff, train_id, vali_id, vali_test_id, test_id, non_norm_adj)