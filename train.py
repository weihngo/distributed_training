#system packages
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
#custom packages
from trian_tools import train_one_epoch,eval_one_epoch
from model import resnet18
from dataset import partition_dataset,create_dataloader
from optimizer_tools import LocalSGD
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--data', default='../data', help='path to dataset')
    parser.add_argument('--checkpoint', default='../checkpoints/best_accuracy.pth', help='path to checkpoint')
    parser.add_argument('--world_size', default=4, help='total gpu num')
    parser.add_argument('--epoches', default=1, help='epoch num')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    parser.add_argument('--tau', default=10, help='how much step to all_reduce')
    parser.add_argument('--batch_size', default=192, help='batch_size')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    args = parser.parse_args()
    return args

def train(rank,nprocs,args):
    print(rank)
    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            rank=rank,
                            world_size=args.world_size)
    # seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    # create dataset.
    #train_loader, test_loader = partition_dataset(rank, args.world_size, args)
    train_loader,test_loader,train_sampler=create_dataloader('../data',args.world_size,args.batch_size)
    print("loading dataset successed!")
    # create model.
    model=resnet18()


    torch.cuda.set_device(rank)
    model.cuda(rank)
    cudnn.benchmark = True
    # define the optimizer.
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = LocalSGD(model.parameters(),
                         lr=args.lr,
                         gmf=0,
                         tau=args.tau,
                         size=args.world_size,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=1e-4)
    # define the criterion and lr scheduler.
    criterion = nn.CrossEntropyLoss().cuda(rank)
    for epoch in range(args.epoches):
        acc=train_one_epoch(model,optimizer,criterion,train_loader,test_loader,epoch,rank)
        print(acc)
        break

def main():
    args =get_args() #get parameters
    print("The config parameters are-> world_size:%d, epoches:%d, lr:%.2f, tau:%d" % (
    args.world_size, args.epoches, args.lr, args.tau))
    import time
    start = time.time()
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))
    end = time.time()
    print("Training time is: " + str(end - start))

if __name__ == '__main__':
    main()