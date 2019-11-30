from __future__ import division

import os, sys, pdb, shutil, time, random, copy
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from model import CifarResNet
from copy import deepcopy
import numpy as np
import math

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

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

# train function (forward, backward, update)
def adv_train(train_loader, model, criterion, optimizer, epoch, log, args):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  noises = {}
  excluded_params = []
  for _m in model.modules():
      if isinstance(_m, torch.nn.BatchNorm1d) or \
              isinstance(_m, torch.nn.BatchNorm2d) or \
              isinstance(_m, torch.nn.BatchNorm3d):
          for _k, _p in _m.named_parameters():
              excluded_params.append(_p)

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)


    for key, p in model.named_parameters():
      # if hasattr(model, 'quiet_parameters') and (key in model_copy.quiet_parameters):
      #       continue
      if any([p is pp for pp in excluded_params]):
            continue
      # if args.adapt_type == 'weight':
      #       noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) *\
      #        args.sharpness_smoothing * torch.abs(p.data) * noise_coef
      # elif args.adapt_type == 'filter':
      # if args.noise_type == 'uniform':
      noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.)
      # elif args.noise_type == 'normal':
        # noise = torch.cuda.FloatTensor(p.size()).normal_()
      # else:
        # raise ValueError('Unkown --noise-type')
      noise_shape = noise.shape
      noise_norms = noise.view([noise_shape[0],-1]).norm(p=2, dim=1) + 1.0e-6
      p_norms = p.view([noise_shape[0], -1]).norm(p=2, dim=1)
      for shape_idx in range(1, len(noise_shape)):
          noise_norms = noise_norms.unsqueeze(-1)
          p_norms = p_norms.unsqueeze(-1)
      noise = noise / noise_norms * p_norms.data
      #for idx in range(0, noise.shape[0]):
      #  if 1 == len(noise.shape):
      #    if np.abs(np.linalg.norm(noise[idx]))>1.0e-6:
      #      noise[idx] = noise[idx] / np.linalg.norm(noise[idx]) * np.linalg.norm(p.data[idx])
      #  else:
      #    if np.abs(noise[idx].norm())>1.0e-6:
      #      noise[idx] = noise[idx] / noise[idx].norm() * p.data[idx].norm()
      noise_coef = 1.0
      noise = noise * noise_coef
      # elif args.adapt_type == 'none':
      #       if args.noise_type == 'uniform':
      #         noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) *\
      #          args.sharpness_smoothing * noise_coef
      #       elif args.noise_type == 'normal':
      #         noise = torch.cuda.FloatTensor(p.size()).normal_() * args.sharpness_smoothing *\
      #          noise_coef
      #       else:
      #         raise ValueError('Unkown --noise-type')
      # else:
      #       raise ValueError('Unkown --adapt-type')
      noises[key] = noise
      p.data.add_(noise)


    # compute output
    output,_= model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.cpu().numpy(), input.size(0))
    top1.update(prec1.cpu().numpy(), input.size(0))
    top5.update(prec5.cpu().numpy(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    for key, p in model.named_parameters():
      if key in noises:
        p.data.sub_(noises[key])
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
  torch.save(state, './model/save_adv_train_from_pretrained.pth.tar')
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
  return top1.avg, losses.avg


# train function (forward, backward, update)
def dis_train(train_loader, model, criterion, optimizer, epoch, log, args, discriminator, discriminator_optim,\
 model_initial):
  dis_step = 1
  clf_step = 1
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion_dis = torch.nn.BCEWithLogitsLoss()

  end = time.time()
  total_loss_dis = 0
  total_accuracy_dis = 0
  iterations = 0
  for i, (input, target) in enumerate(train_loader):
    iterations += 1
    # measure data loading time
    data_time.update(time.time() - end)
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)


    import random
    prob = random.uniform(0, 1)    
    if prob < 1:
      #change parameters.
      model_copy = deepcopy(model)#model_initial
      noises = {}
      excluded_params = []
      for _m in model_copy.modules():
          if isinstance(_m, torch.nn.BatchNorm1d) or \
                  isinstance(_m, torch.nn.BatchNorm2d) or \
                  isinstance(_m, torch.nn.BatchNorm3d):
              for _k, _p in _m.named_parameters():
                  excluded_params.append(_p)

      layer_limit = 0
      for key, p in model_copy.named_parameters():
        layer_limit += 1
        if any([p is pp for pp in excluded_params]):
              # print('mark!')
              continue
        noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.)
        noise_shape = noise.shape
        noise_norms = noise.view([noise_shape[0],-1]).norm(p=2, dim=1) + 1.0e-6
        p_norms = p.view([noise_shape[0], -1]).norm(p=2, dim=1)
        #for boardcasting. 
        for shape_idx in range(1, len(noise_shape)):
            noise_norms = noise_norms.unsqueeze(-1)
            p_norms = p_norms.unsqueeze(-1)
        # print(noise_shape)
        noise = noise / noise_norms * p_norms.data
        noise_coef = 1.0
        noise = noise * noise_coef
        noises[key] = noise
        p.data.add_(noise)
        # if layer_limit > 1:
        #   break
      # model_copy.eval()
      set_requires_grad(model_copy, requires_grad=False)
      # compute output
      output, features_target = model(input_var)
      _, features_source = model_copy(input_var)
      
      set_requires_grad(model, requires_grad=False)
      set_requires_grad(discriminator, requires_grad=True)
      for _ in range(dis_step):
          discriminator_x = torch.cat([features_source, features_target])
          discriminator_y = torch.cat([torch.ones(features_source.shape[0], device=device),
                                       torch.zeros(features_target.shape[0], device=device)])
          preds = discriminator(discriminator_x).squeeze()
          loss = criterion_dis(preds, discriminator_y)
          discriminator_optim.zero_grad()
          loss.backward(retain_graph=True)
          discriminator_optim.step()
          total_loss_dis += loss.item()
          total_accuracy_dis += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
      # Train classifier
      set_requires_grad(model, requires_grad=True)
      set_requires_grad(discriminator, requires_grad=False)
      for _ in range(clf_step):
          # flipped labels
          discriminator_y = torch.ones(features_target.shape[0], device=device)
          preds = discriminator(features_target).squeeze()
          loss = criterion_dis(preds, discriminator_y)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()

    output, _ = model(input_var)
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
    if math.isnan(loss.data.cpu().numpy()):
      import ipdb
      ipdb.set_trace()

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
  mean_loss = total_loss_dis / (iterations*dis_step)
  mean_accuracy = total_accuracy_dis / (iterations*dis_step)
  print(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
             f'discriminator_accuracy={mean_accuracy:.4f}')
  state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
  torch.save(state, './model/save_dis_train_temp.pth.tar')
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, \
    error1=100-top1.avg), log)
  return top1.avg, losses.avg
