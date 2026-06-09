import time
import torch


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        # {name} {val:.3f} ({avg:.3f})
        fmtstr = '{name} {val'+self.fmt+'} ({avg'+self.fmt+'})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix+self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # 输出格式：[1/100] time:,acc1,acc5,loss
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches//1))
        fmt = '{:'+str(num_digits)+'d}'
        # [{:3d}/100]
        return '['+fmt+'/'+fmt.format(num_batches)+']'


def adjust_learning_rate(optimizer, epoch, args):
    '''调整学习率：每30个epoch学习率乘以0.1'''
    lr = args.lr*(0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    '''计算前k个最大值中有多少个正确的'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # values, indices = input_tensor.topk(k, dim=None, largest=True, sorted=True)
        # 生成一个二维的tensor，第一维是每个样本的前k个最大值，第二维是每个最大值的索引
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(pred)  # tensor([[3, 0, 3],[2, 1, 0]])
        # 将target转换成pred的形状
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(target)
        # print(target.view(1, -1).expand_as(pred))
        # print(correct)
        res = []
        for k in topk:
            # 计算前k个最大值中有多少个正确的
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            # tensor([False,  True,  True,  True, False, False])
            # print(correct[:k].contiguous().view(-1))
            res.append(correct_k.mul_(100.0/batch_size))
        return res


def main():
    # tmp = ProgressMeter(100, AverageMeter('loss'))
    # for i in range(100):
    #     tmp.print(i)
    #     time.sleep(0.1)
    #     tmp.meters[0].update(i)

    output = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.5]])
    target = torch.tensor([2, 0, 3])
    print(accuracy(output, target, (1, 2)))


if __name__ == '__main__':
    # main()
    pass
