from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import load_data, accuracy
from pygcn.models import GCNIED
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.8,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--edge', type=float, default=0.2,
                    help='Edge deletion or addition rate.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--hops', type=int, default=10, help='Propagation step')
parser.add_argument('--threshold', type=float, default=0.4, help='threshold for deleting edges')
parser.add_argument('--aggre_range', type=int, default=1, help='aggregate local neighborhood')
parser.add_argument('--k_connection', type=int, default=2, help='knn graph')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset, args.seed)
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)

# Model and optimizer
model = GCNIED( num=features.shape[0],
                nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                hops=args.hops,
                threshold=args.threshold,
                aggre_range=args.aggre_range,
                k_connection=args.k_connection,
                input_droprate=args.input_droprate,
                hidden_droprate=args.hidden_droprate)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()

def train(epoch):
    t = time.time()
    X = features
    model.train()
    optimizer.zero_grad()
    output, newA = model(X, A)
    output = torch.log_softmax(output, dim=-1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])


    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, newA = model(X, A)
    output = torch.log_softmax(output, dim=-1)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()
def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0
    loss_mn = np.inf
    acc_mx = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:
            if loss_values[-1] <= loss_best: 
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset +'.pkl'))



def test():
    model.eval()
    X = features
    output, newA = model(X, A)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])

    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
Train()
test()
