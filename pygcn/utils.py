import sys
import os
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(dataset_str = 'cora', seed = 42, edge_change = 0.75):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot
    label = np.argmax(labels, axis=-1)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])


    D = np.array(adj.sum(axis=1))
    D1 = D**(-0.5)
    D2 = np.array(adj.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:,0], format='csr')
    D2 = sp.diags(D2[0,:], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def full_load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #adj=graph_delete_connections(labels,0.7,42,adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #calculate_homophily(labels, adj)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask

def full_load_data(dataset_name, ratio, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = full_load_citation (dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = normalize(features)

    g = adj
    # calculate_homophily(labels, g)
    # distribu = np.array([[0, 1, 0.0, 0.0, 0.0], [0, 0, 1, 0.0, 0.0], [0.0, 0.0, 0.0, 1, 0],
    #                      [0.0, 0.0, 0, 0.0, 1], [1.0, 0.0, 0, 0, 0]])
    # dis = cal_distribution(labels, g)
    # dis = np.tile(dis, (5, 1)).T
    # distribu = np.multiply(distribu, dis)
    # g = graph_add_Distribution(labels, distribu, ratio, 42, g)
    # calculate_homophily(labels,g)
    """g = graph_delete_connections(ratio, 41, g.toarray())
    g = sp.csr_matrix(g)"""
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    g = sys_normalized_adjacency(g)

    g = sparse_mx_to_torch_sparse_tensor(g)


    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
def sys_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    mx = mx.astype(np.float)
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def label_vector_generator(adj, label_vector, index = [], order=1, shuffle=False, style=0):
    label_vectors = []
    lv = torch.zeros(label_vector.shape)
    lv[index, :] = label_vector[index, :]
    for i in range(order):
        label_vectors.append(lv)
        if i!= (order-1):
            lv = torch.spmm(adj, lv)
    if style == 0:
        return sum(label_vectors)*1.0/order
    return torch.cat(label_vectors, 1)

def feature_generator(adj, features, order=0):
    n = features.shape[0]
    index = np.random.permutation(n)
    index_1 = index[: n//2]
    index_2 = index[n//2 :]
    mask_1 = torch.zeros(n,1)
    mask_2 = torch.zeros(n,1)
    mask_1[index_1] = 1
    mask_2[index_2] = 1

    features_1 = [mask_1.cuda() * features]
    features_2 = [mask_2.cuda() * features]

    alpha = 1
    for i in range(order):
        features_1.append(alpha*torch.spmm(adj, features_1[-1]) + (1-alpha)*features_1[0])
    for i in range(order):
        features_2.append(alpha*torch.spmm(adj, features_2[-1]) + (1-alpha)*features_2[0])

    return sum(features_1)*1./(order+1), sum(features_2)*1./(order+1)


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, input):
        input_c = input.coalesce()
        drop_val = F.dropout(input_c._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_c._indices(), drop_val, input.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


def graph_delete_connections(prob_del, seed, adj):
    rnd = np.random.RandomState(seed)

    del_adj = np.array(adj, dtype=np.float32)
    pre_num_edges = np.sum(del_adj)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape)

    del_adj *= smpl


    cur_num_edges = np.sum(del_adj)
    print('[ Deleted {}% edges ]'.format(100 * (pre_num_edges - cur_num_edges) / pre_num_edges))
    return del_adj

def graph_add_connections(prob_add, seed, adj):
    rnd = np.random.RandomState(seed)
    add_adj = np.array(adj, dtype=np.float32)
    pre_num_edges = np.sum(add_adj)
    add_num = prob_add*pre_num_edges

    all = adj.shape[0]*adj.shape[1]
    add_rotio = add_num/all

    smpl = rnd.choice([0., 1.], p=[1.-add_rotio, add_rotio], size=adj.shape)
    #smpl += smpl.transpose()

    add_adj = np.where(smpl>0, smpl, add_adj)
    cur_num_edges = np.sum(add_adj)
    print('[ Added {}% edges ]'.format(100 * (cur_num_edges - pre_num_edges) / pre_num_edges))
    return add_adj

def graph_add_Distribution(labels, distribution, ratio_add, seed, adj):
    all_idx = np.arange(len(labels))
    rnd = np.random.RandomState(seed)
    label_len=labels.max()+1
    resultRow=[]
    resultCol=[]
    edge_len=adj.data.shape[0]
    chose_edge=int(edge_len*ratio_add)
    print(chose_edge)
    choose = np.ceil(chose_edge * distribution)
    # distribu = np.array([[0, 1, 0.0, 0.0, 0.0], [0, 0, 1, 0.0, 0.0], [0.0, 0.0, 0.0, 1, 0],
    #                      [0.0, 0.0, 0, 0.0, 1], [1.0, 0.0, 0, 0, 0]])
    # choose = choose + distribu #use in Texas and Cornell
    choose = choose.astype(np.int32)

    for i in range(chose_edge):
        allClass=[]
        node = rnd.choice(all_idx, 1, replace=True)
        tempIdx=labels[node]
        resultRow.append(node)
        for j in range(label_len):
            allClass.append(rnd.choice(
                all_idx[np.where(labels == j)], choose[tempIdx,j], replace=True))
        get_all=np.concatenate(allClass)
        resultCol.append(rnd.choice(get_all, 1, replace=False))

    row = np.concatenate(resultRow)
    col =np.concatenate(resultCol)

    data = np.ones_like(row)
    coo = sp.coo_matrix((data, (row, col)), shape=adj.shape)
    #coo_inver= sp.coo_matrix((data, (col, row)), shape=adj.shape)

    coo = coo.tocsr()
    adj = adj+coo
    return adj

def calculate_homophily(labels, adj):

    adj_coo = adj.tocoo()
    row = adj_coo.row
    col = adj_coo.col
    all_sum = row.shape[0]
    print(all_sum)
    homo_num = 0
    for i in range(row.shape[0]):

        c = (labels[col[i]] == labels[row[i]])
        if (c.all()):
            homo_num = homo_num + 1
    print(homo_num / all_sum)

def cal_distribution(labels, adj):
    adj_coo = adj.tocoo()
    row = adj_coo.row
    col = adj_coo.col
    lenghth=labels.max()+1
    label_distribution=np.zeros((lenghth,lenghth))
    for i in range(row.shape[0]):
        rowi=labels[row[i]]
        col1=labels[col[i]]
        label_distribution[rowi,col1]=label_distribution[rowi,col1]+1
        #label_distribution[col1,rowi]=label_distribution[col1,rowi]+1

    label_distribution=label_distribution/row.shape[0]

    node_dis=np.sum(label_distribution,axis=1)
    return node_dis



def cal_class_similarity(labels,adj):
    adj_coo = adj.tocoo()
    row = adj_coo.row
    col = adj_coo.col
    length=len(labels)
    lenlabel=labels.max()+1
    distri=np.zeros([length,lenlabel])
    similarity = np.zeros([lenlabel, lenlabel])
    for i in range(row.shape[0]):
        r=row[i]
        c=col[i]
        t=labels[c]
        distri[r][t]=distri[r][t]+1
        distri[c][t] = distri[c][t] -1
    for i in range(length):
        t=labels[i]
        if(np.all(distri[i]==0)):
            distri[i][t] = distri[i][t] + 1

    denom = np.linalg.norm(distri, axis=1, keepdims=True)
    cos = distri / denom
    cosine = np.matmul(cos, cos.transpose(-1, -2))

    for i in range(lenlabel):
        index=np.array(np.where(labels==i)).flatten()
        iLen=index.shape[0]
        for j in range(lenlabel):
            result=0
            other=np.array(np.where(labels==j)).flatten()
            oLen=other.shape[0]
            for classi in range(iLen):
                for classj in range(oLen):
                    result=result+cosine[index[classi],other[classj]]

            similarity[i][j]=similarity[i][j]+result/(iLen*oLen)
    similarity=np.around(similarity,decimals=2)
    allSum=np.sum(similarity)
    print (allSum)

    return similarity,allSum

