from __future__ import print_function
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
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=2, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=42, metavar='S',
#                     help='random seed (default: 42)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--fp16-allreduce', action='store_true', default=False,
#                     help='use fp16 compression during allreduce')
# parser.add_argument('--num-proc', type=int)
# # parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
# parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')
# parser.add_argument('--use-adasum', action='store_true', default=False,
#                     help='use adasum algorithm to do reduction')

# parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
#                     help='apply gradient predivide factor in optimizer (default: 1.0)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
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

def mainf(args):
    # Horovod: initialize library.
    even_set = hvd.ProcessSet([0,2])
    odd_set = hvd.ProcessSet([1,3])
    hvd.init(process_sets=[even_set, odd_set])
    torch.manual_seed(args.seed)
    # print("hvd.local_rank()",hvd.local_rank())
    # print("hvd.global_rank()",hvd.global_rank())hvd.
    # print("hvd.rank()",hvd.rank())
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            sampler=test_sampler, **kwargs)



    model = Net()
    model1=Net()
    if args.cuda:
        # Move model to GPU.
        model.cuda()
        model1.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_parameters(model1.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)#,process_set=even_set)
    hvd.broadcast_optimizer_state(optimizer1, root_rank=0)#,process_set=even_set)
    # GRACE: compression algorithm.
    # grc = Allgather(TopKCompressor(0.3), ResidualMemory(), hvd.size())

    from grace.grace_dl.torch.helper import grace_from_params
    params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather'}
    grc = grace_from_params(params)
    # print("grc",grc)

    # Horovod: wrap optimizer with DistributedOptimizer.
    # compression = Allgather(TopKCompressor(0.01), NoneMemory())
    # compression = Allreduce(RandomKCompressor(), ResidualMemory())
    # hvd.Compression.fp16
    
    # 
    from grace.grace_dl.torch.compressor.randomk import RandomKCompressor
    cpr=TopKCompressor(0.1)
    # if(hvd.rank()%2==0):
    # optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(),grace=grc)#,process_set=even_set#hvd.Compression.fp16)
    # else:
        # optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(),grace=grc,process_set=odd_set)#hvd.Compression.fp16)


    def train(epoch):
        
        # Horovod: set epoch to sampler for shuffling.
        # opt_params=[]
        # opt_params.append({"params": layer.parameters(), "lr": lr})
        # optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                        # momentum=args.momentum)
        optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)
        compression = Allgather(TopKCompressor(0.01), ResidualMemory(), hvd.size())
        if(hvd.rank()%2==0):
            
            optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)
            optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(),grace=grc,process_set=even_set,op=hvd.Adasum if args.use_adasum else hvd.Average,gradient_predivide_factor=args.gradient_predivide_factor)#hvd.Compression.fp16)
            model.train()
        else:
            
            optimizer1 = optim.SGD(model1.parameters(), lr=args.lr * hvd.size(),
                        momentum=args.momentum)
            optimizer1 = hvd.DistributedOptimizer(optimizer1, model1.named_parameters(),grace=grc,process_set=odd_set,op=hvd.Adasum if args.use_adasum else hvd.Average,gradient_predivide_factor=args.gradient_predivide_factor)#hvd.Compression.fp16)
            model1.train()

        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            
            if(hvd.rank()%2==0):
                optimizer.zero_grad()
                output = model(data)
            else :
                optimizer1.zero_grad()
                output = model1(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if(hvd.rank()%2==0):
                optimizer.step()
            else :
                optimizer1.step()
            
            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()))


    def metric_average(val, name):
        tensor = torch.tensor(val)#.clone().detach().requires_grad_(True)
        # tensor1 = torch.tensor(val).clone().detach().requires_grad_(True)
        # avg_tensor1 = hvd.allreduce(tensor, name=name,process_set=even_set)
        if(hvd.rank()%2==0):
            # print("EVEN reduce")
            avg_tensor = hvd.allreduce(tensor, name=name,process_set=even_set)
            # print("EVEN reduce completed")
            return avg_tensor.item()
        else:
            # print("ODD reduce")

            avg_tensor1 = hvd.allreduce(tensor, name=name,process_set=odd_set)
            # print("ODD reduce completed")
            return avg_tensor1.item()
        #     
        # print(avg_tensor1.item(),avg_tensor.item())

        


    def test():
        if(hvd.rank()%2==0):
            model.eval()
        else:
            model1.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if(hvd.rank()%2==0):
                output = model(data)
            else :
                output = model1(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        
        # if(hvd.rank()%2==0):
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
        # else:
        #     test_loss = metric_average(test_loss, 'avg_loss1')
        #     test_accuracy = metric_average(test_accuracy, 'avg_accuracy1')

        # Horovod: print output only on first rank.
        # if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))
        import time
        time.sleep(5)


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # if(hvd.rank()%2==0):
        test()
    return "1"
if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.num_proc:
        # run training through horovod.run
        print('Running training through horovod.run')
        horovod.run(mainf,
                    args=(args,),
                    np=args.num_proc,
                    hosts=args.hosts,
                    use_gloo=args.communication == 'gloo',
                    use_mpi=args.communication == 'mpi')
    else:
        # this is running via horovodrun
        main(args)