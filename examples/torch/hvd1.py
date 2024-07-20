from __future__ import print_function
from multiprocessing import freeze_support
import sys
sys.path.append("/content/")
sys.path.append("/content/grace/")
sys.path.append("/Users/garvitbanga/Downloads/AritraDutta/FedCola-LOCAL/")
sys.path.append("/Users/garvitbanga/Downloads/AritraDutta/FedCola-LOCAL/grace/")
import argparse
import sys
import os
# print(sys.path)
sys.path.append(os.path.join('/Users/garvitbanga/Downloads/AritraDutta/FedCola-LOCAL/', 'grace/'))
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod
import horovod.torch as hvd
from grace.grace_dl.torch.communicator.allgather import Allgather
from grace.grace_dl.torch.communicator.allreduce import Allreduce
from grace.grace_dl.torch.compressor.topk import TopKCompressor
from grace.grace_dl.torch.memory.residual import ResidualMemory
import torch
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--num-proc', type=int)
# parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    

if __name__ == '__main__':
                freeze_support()
                args = parser.parse_args()
                args.cuda = not args.no_cuda and torch.cuda.is_available()
                import grace.examples.torch.pytorch_mnist as pmt
                if args.num_proc:
                    # run training through horovod.run
                    print('Running training through horovod.run1')
                    ans=horovod.run(pmt.mainf,
                                args=(args,),
                                np=args.num_proc,
                                # hosts=args.hosts,
                                use_gloo=True,
                                use_mpi=args.communication == 'mpi')
                    print("ans",ans)
                else:
                    # this is running via horovodrun
                    main(args)