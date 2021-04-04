import torch
def train_one_epoch(model,optimizer,criterion,train_dataloader,test_dataloader,epoch,rank):
    model.train()
    print('begin train!')
    for batch_idx,(data,target) in enumerate(train_dataloader):
        data=data.cuda(rank, non_blocking=True)
        target=target.cuda(rank, non_blocking=True)

        output=model(data)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # torch.cuda.synchronize()
        is_average=optimizer.average()
        if batch_idx%10==0:
            # if  is_average and rank==0:
            #     print("average begin------------------------------")

            accuracy=eval_one_epoch(model,test_dataloader,rank)
            print('[%d, %5d] loss: %.3f  accuracy:%.3f rank:%d' %(epoch + 1, batch_idx, loss,accuracy, rank))

            # if  is_average and rank==0:
            #     print("average over------------------------------")
    return accuracy

def eval_one_epoch(model,test_dataloader,rank):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.cuda(rank, non_blocking = True)
            labels = labels.cuda(rank, non_blocking = True)

            outputs = model(images)
            acc1 = comp_accuracy(outputs,labels)
            top1.update(acc1[0].item(), images.size(0))
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum()
    return top1.avg

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count