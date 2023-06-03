from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import accuracy, full_load_data
from pygcn.models import GCNIED

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--edge', type=float, default=0.5,
                    help='Edge deletion or addition rate.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--hops', type=int, default=4, help='Propagation step')
parser.add_argument('--threshold', type=float, default=1, help='threshold for deleting edges')
parser.add_argument('--aggre_range', type=int, default=2, help='aggregate local neighborhood')
parser.add_argument('--k_connection', type=int, default=2, help='knn graph')
parser.add_argument('--dataset', type=str, default='chameleon', help='Data set')
parser.add_argument('--model', type=str, default='GCNIED', choices=['GCNPND'])

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda")


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    output, newA = model(features, adj)
    output = torch.log_softmax(output, dim=-1)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate_step(model, features, labels, adj, idx_val):
    model.eval()
    with torch.no_grad():
        output, newA = model(features, adj)
        output = torch.log_softmax(output, dim=-1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test_step(model, features, labels, adj, idx_test):
    model.eval()
    with torch.no_grad():
        output, newA = model(features, adj)
        output = torch.log_softmax(output, dim=-1)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return output[idx_test].detach().cpu().numpy(), loss_test.item(), acc_test.item()


def train(datastr, splitstr):
    g, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr, args.edge,
                                                                                                 splitstr)
    features = features.to(device)
    adj = g.to(device)
    model = GCNIED(
        num=features.shape[0],
        nfeat=num_features,
        nhid=args.hidden,
        nclass=num_labels,
        hops=args.hops,
        threshold=args.threshold,
        aggre_range=args.aggre_range,
        k_connection=args.k_connection,
        input_droprate=args.input_droprate,
        hidden_droprate=args.hidden_droprate)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    model.cuda()
    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, adj, idx_train)
        loss_val, acc_val = validate_step(model, features, labels, adj, idx_val)
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if loss_val <= best:
            best = loss_val
            torch.save(model.state_dict(), args.dataset + '.pkl')
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    model.load_state_dict(torch.load(args.dataset + '.pkl'))
    test_logits, _, acc = test_step(model, features, labels, adj, idx_test)
    return acc * 100


t_total = time.time()
acc_list = []
for i in range(10):
    datastr = args.dataset
    splitstr = 'splits/' + args.dataset + '_split_0.6_0.2_' + str(i) + '.npz'
    acc_list.append(train(datastr, splitstr))
    print(i, ": {:.2f}".format(acc_list[-1]))
print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
