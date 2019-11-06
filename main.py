from __future__ import division

import os, sys, pdb, shutil, time, random, copy
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from model import resnet20,CifarResNet
from adver_utils import *
from copy import deepcopy
# model_names = sorted(name for name in models.__dict__
#   if name.islower() and not name.startswith("__")
#   and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path',default = './data', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default = 'cifar10',\
  choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
# parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./logs', help='Folder to save checkpoints and log.')
parser.add_argument('--gpu', type=int, default=1, help='.')
parser.add_argument('--resume', type=str, default = './model/save_adv_train_from_pretrained.pth.tar', \
  help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate',\
 action='store_false', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--save_figurename', type=str, default='curve_adv_train.png', help='')
parser.add_argument('--action', type=int, default=0, help='')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def discriminator_function():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(64, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    ).to(device)

    return discriminator

def main():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)

  train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
  test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
  elif args.dataset == 'svhn':
    train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'stl10':
    train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'imagenet':
    assert False, 'Do not finish imagenet code'
  else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

  # print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  net = resnet20(num_classes)
  # print_log("=> network :\n {}".format(net), log)

  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
  discriminator = discriminator_function()
  discriminator_optim = torch.optim.Adam(discriminator.parameters(),1e-4)

  # define loss function (criterion) and optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)
  # optimizer = torch.optim.Adam(net.parameters(), state['learning_rate'],
  #               weight_decay=state['decay'])

  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)
  # optionally resume from a checkpoint
  # if args.resume:
  #   if os.path.isfile(args.resume):
  #     print_log("=> loading checkpoint '{}'".format(args.resume), log)
  #     checkpoint = torch.load(args.resume)
  #     recorder = checkpoint['recorder']
  #     args.start_epoch = checkpoint['epoch']
  #     net.load_state_dict(checkpoint['state_dict'])
  #     optimizer.load_state_dict(checkpoint['optimizer'])
  #     print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
  #   else:
  #     raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
  
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      checkpoint = torch.load(args.resume)
      # recorder = checkpoint['recorder']
      # args.start_epoch = checkpoint['epoch']
      net.load_state_dict(checkpoint['net'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

  if args.evaluate:
    acc_temp, loss_temp = validate(test_loader, net, criterion, log)
    print('the direct evaluation result is: ', acc_temp, loss_temp)

    
  net_initial = deepcopy(net)
  # Main loop
  start_time = time.time()
  epoch_time = AverageMeter()
  for epoch in range(args.start_epoch, args.epochs):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

    # train for one epoch
    if epoch < 0: #and not os.path.exists('./model/save.pth.tar'):
      # train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)
      train_acc, train_los = adv_train(train_loader, net, criterion, optimizer, epoch, log, args)
    else:
      train_acc, train_los = adv_train(train_loader, net, criterion, optimizer,\
     epoch, log, args)
     #  train_acc, train_los = dis_train(train_loader, net, criterion, optimizer,\
     # epoch, log, args, discriminator, discriminator_optim, net_initial)
    # evaluate on validation set
    #val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
    val_acc,   val_los   = validate(test_loader, net, criterion, log)
    is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': net.state_dict(),
      'recorder': recorder,
      'optimizer' : optimizer.state_dict(),
      'args'      : copy.deepcopy(args),
    }, is_best, args.save_path, 'checkpoint.pth.tar')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    recorder.plot_curve( os.path.join(args.save_path, args.save_figurename) )

  log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output,_ = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.cpu().numpy(), input.size(0))
    top1.update(prec1.cpu().numpy(), input.size(0))
    top5.update(prec5.cpu().numpy(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
  state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
  torch.save(state, './model/save.pth.tar')
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
  return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output,_ = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.cpu().numpy(), input.size(0))
    top1.update(prec1.cpu().numpy(), input.size(0))
    top5.update(prec5.cpu().numpy(), input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def validate_perturb():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)

  train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
  test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
  elif args.dataset == 'svhn':
    train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'stl10':
    train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'imagenet':
    assert False, 'Do not finish imagenet code'
  else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

  # print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  net = resnet20(num_classes)
  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
  # define loss function (criterion) and optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)

  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)

  
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      checkpoint = torch.load(args.resume)
      net.load_state_dict(checkpoint['net'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

  if args.evaluate:
    acc_temp, loss_temp = validate(test_loader, net, criterion, log)
    print('the direct evaluation result is: ', acc_temp, loss_temp)

  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  model = net
  # switch to evaluate mode
  model.eval()
  noises = {}
  excluded_params = []
  for _m in model.modules():
      if isinstance(_m, torch.nn.BatchNorm1d) or \
              isinstance(_m, torch.nn.BatchNorm2d) or \
              isinstance(_m, torch.nn.BatchNorm3d):
          for _k, _p in _m.named_parameters():
              excluded_params.append(_p)
  layer_index = 0

  for key, p in model.named_parameters():
    layer_index += 1
    if any([p is pp for pp in excluded_params]):
          continue
    noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.)
    noise_shape = noise.shape
    noise_norms = noise.view([noise_shape[0],-1]).norm(p=2, dim=1) + 1.0e-6
    p_norms = p.view([noise_shape[0], -1]).norm(p=2, dim=1)
    for shape_idx in range(1, len(noise_shape)):
        noise_norms = noise_norms.unsqueeze(-1)
        p_norms = p_norms.unsqueeze(-1)
    noise = noise / noise_norms * p_norms.data
    noise_coef = 1.0
    noise = noise * noise_coef
    noises[key] = noise
    if layer_index == 1:
      break
    else:
      p.data.add_(noise)


  for i, (input, target) in enumerate(test_loader):
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output,_ = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.cpu().numpy(), input.size(0))
    top1.update(prec1.cpu().numpy(), input.size(0))
    top5.update(prec5.cpu().numpy(), input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, \
    top5=top5, error1=100-top1.avg), log)
    
  log.close()



def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])
    pdb.set_trace()

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.cpu().numpy(), input.size(0))
    top1.update(prec1.cpu().numpy(), input.size(0))
    top5.update(prec5.cpu().numpy(), input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  if args.action == 0:
    main()
  else:
    validate_perturb()
