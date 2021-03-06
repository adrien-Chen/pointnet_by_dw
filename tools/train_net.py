from __future__ import  print_function
import argparse
import os
import random
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from pointnet.model import point_net_cls, point_net_seg, feature_transform_regularizer
from datasets.dataset import ModelNetDataset, ShapeNetDataset

# set commend params
parse = argparse.ArgumentParser()
parse.add_argument(
    '--batch_size', type=int, default=32, help='input batch size')
parse.add_argument(
    '--num_samples', type=int, default=2500, help='num of sample points from a point set')
parse.add_argument(
    '--workers', type=int, default=4, help='num of data loading workers')
parse.add_argument(
    '--nepoch', type=int, default=250, help='num of epochs to train for')
parse.add_argument('--outf', type=str, default='cls', help='output folder')
parse.add_argument('--model', type=str, default='', help='set model path')
parse.add_argument('--dataset', type=str, required=True, help='set dataset path')
parse.add_argument('--dataset_type', type=str, default='modelnet40', help='type of your dataset')
parse.add_argument('--feature_transform', action='store_true', help='use feature transform')  # True or False
parse.add_argument('--class_choice', type=str, default='Chair', help='class choice for segmentation at one time')

# if try to use shapenet for training, u should first set the '--task' for the model
parse.add_argument('--task', type=str, default='cls', help='the task of the model')
# add logs:
parse.add_argument('--log_dir', type=str, default='eval_logs', help='the dir of log')
# show parse:
opt = parse.parse_args()
print(opt)

# output with blue color:
blue = lambda x: '\033[94m' + x + '\033[0m'

# set random seed for recurrent:
opt.manualSeed = random.randint(1, 10000)  # set a fix seed
print("random seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# check dataset type:
if opt.dataset_type == 'modelnet40':  # data_augmentation default is True!
    dataset = ModelNetDataset(
        root_dir=opt.dataset,
        num_points=opt.num_samples,
        split='trainval'
    )

    test_dataset = ModelNetDataset(
        root_dir=opt.dataset,
        num_points=opt.num_samples,
        split='test',
        data_augmentation=False
    )
elif opt.dataset_type == 'shapenet' and opt.task == 'seg':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        num_points=opt.num_samples,
        class_choice=opt.class_choice
    )

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        num_points=opt.num_samples,
        class_choice=opt.class_choice,
        split='test',
        data_augmentation=False
    )
elif opt.dataset_type == 'shapenet' and opt.task == 'cls':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        num_points=opt.num_samples,
        classification=True
    )

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        num_points=opt.num_samples,
        split='test',
        data_augmentation=False,
        classification=True
    )
else:
    exit("wrong dataset type!")

# create data iterators for training set and test set:
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))  # should be 9843, 2468 or 2658, 704 for ShapeNet
if opt.dataset_type == 'shapenet' and opt.task == 'seg':
    num_classes = dataset.num_seg_classes  # for ShapeNet
elif opt.dataset_type == 'shapenet' and opt.task == 'cls':
    num_classes = len(dataset.classes)
else:
    num_classes = len(dataset.classes)  # ModelNet40

print('classes', num_classes)

try:
    os.makedirs(opt.outf)  # write output ?
except OSError:
    pass

# ========================================
# create model for training cls:
# ========================================
def PointNetCls():
    classifier = point_net_cls(k=num_classes, feature_transform=opt.feature_transform)  # default is False!

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))  # if you have trained model params

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()  # model load to gpu

    num_batch = len(dataset) / opt.batch_size
    # save loss and acc:
    train_loss = {}
    test_loss = {}
    train_acc = {}
    test_acc = {}

    for epoch in range(opt.nepoch):
        # scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]  # [B, 1]->>size([B])
            # print(target.shape)
            points = points.transpose(2, 1)  # [B=32, 3, N]
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()  # for training mode
            # pred, trans, trans_feat = classifier(points)
            try:
                pred, trans, trans_feat = classifier(points)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.01
            loss.backward()
            optimizer.step()
            pred_cls = pred.data.max(1)[1]  # [B, k]->>[B, 1]
            correct = pred_cls.eq(target.data).cpu().sum()  # num of correct predict in batch_i
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                               loss.item(),
                                                               correct.item() / float(opt.batch_size)))
            # add initial train loss and acc in first epoch:
            if epoch == 0 and i == 0:
                train_loss[epoch] = loss.item()
                train_acc[epoch] = correct.item() / float(opt.batch_size)

            # show acc in one batch test_data every 10 batch_size:
            if i % 10 == 0:
                #  add train loss and acc in each epoch:
                if i+10 > num_batch:
                    train_loss[epoch+1] = loss.item()
                    train_acc[epoch+1] = correct.item() / float(opt.batch_size)

                j, data = next(enumerate(test_dataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()  # for evaluation mode
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)

                pred_cls = pred.data.max(1)[1]
                correct = pred_cls.eq(target.data).cpu().sum()
                print(correct.item(), opt.batch_size)
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'),
                                                                loss.item(),
                                                                correct.item() / float(opt.batch_size)))
                # add initial test loss and acc in first epoch:
                if epoch == 0 and i == 0:
                    test_loss[epoch] = loss.item()
                    test_acc[epoch] = correct.item() / float(opt.batch_size)
                # add test loss and acc in each epoch:
                if i+10 > num_batch:
                    test_loss[epoch+1] = loss.item()
                    test_acc[epoch+1] = correct.item() / float(opt.batch_size)

        scheduler.step()
        # save checkpoint every epoch:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))  # default: 'cls/cls_model_0.pth'

    # calculate acc on whole test dataset:
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(test_dataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_cls = pred.data.max(1)[1]
        correct = pred_cls.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]  # add batch_size

    print('final accuracy {}'.format(total_correct / float(total_testset)))
    return train_loss, test_loss, train_acc, test_acc


# ========================================
# only for one class at once:
#   Using ShapeNet dataset!!!
# ========================================
def PointNetSeg():
    classifier = point_net_seg(num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))  # if you have trained model params

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()  # model load to gpu

    num_batch = len(dataset) / opt.batch_size
    # save loss and acc:
    train_loss = {}
    test_loss = {}
    train_acc = {}
    test_acc = {}

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()

            try:
                pred, trans, trans_feat = classifier(points)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            # pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1,  num_classes)  # [B*N, k]
            target = target.view(-1, 1)[:, 0] - 1  # ShapeNet's label is from 1 to k
            loss = F.nll_loss(pred, target)  # -x[class]
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_seg = pred.data.max(1)[1]  # [B*N, k]->>[B*K, 1], max() return [values(probabilities), indices]
            correct = pred_seg.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                               loss.item(),
                                                               correct.item()/float(opt.batch_size * 2500)))
            # add initial train loss and acc in first epoch:
            if epoch == 0 and i == 0:
                train_loss[epoch] = loss.item()
                train_acc[epoch] = correct.item() / float(opt.batch_size * 2500)

            if i%10 == 0:
                #  add train loss and acc in each epoch:
                if i+10 > num_batch:
                    train_loss[epoch+1] = loss.item()
                    train_acc[epoch+1] = correct.item() / float(opt.batch_size * 2500)

                j, data = next(enumerate(test_dataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_seg = pred.data.max(1)[1]
                correct = pred_seg.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                                blue('test'),
                                                                loss.item(),
                                                                correct.item() / float(opt.batch_size * 2500)))
                # add initial test loss and acc in first epoch:
                if epoch == 0 and i == 0:
                    test_loss[epoch] = loss.item()
                    test_acc[epoch] = correct.item() / float(opt.batch_size * 2500)
                # add test loss and acc in each epoch:
                if i+10 > num_batch:
                    test_loss[epoch+1] = loss.item()
                    test_acc[epoch+1] = correct.item() / float(opt.batch_size * 2500)

        scheduler.step()
        # save checkpoint every epoch:
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

    # benchmark mIOU:
    shape_ious = []
    for i, data in tqdm(enumerate(test_dataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_seg = pred.data.max(2)[1]  # [B, N, k]->> [B, N, 1], [0]-max value, [1]-indices

        pred_np = pred_seg.cpu().data.numpy()  # [B, N, 1]
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / U
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print('mIOU for class {}: {}'.format(opt.class_choice, np.mean(shape_ious)))
    return train_loss, test_loss, train_acc, test_acc


# ====================
# show loss curve:
# ====================
import matplotlib.pyplot as plt
def show_loss(train_loss, test_loss):
    x = train_loss.keys()  # idx of each epoch
    x = np.asarray(list(x))
    y_train_loss = np.asarray([train_loss[i] for i in x])
    y_test_loss = np.asarray([test_loss[i] for i in x])

    plt.plot(x, y_train_loss, 'bo-', label='train')
    plt.plot(x, y_test_loss, 'gs-', label='test')
    plt.legend(loc="upper left")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('%s_loss_%s' % (opt.task, datetime.today().date()))
    plt.show()


# ====================
# show acc curve:
# ====================
def show_acc(train_acc, test_acc):
    x = train_acc.keys()
    x = np.asarray(list(x))
    y_train_acc = np.asarray([train_acc[i] for i in x])
    y_test_acc = np.asarray([test_acc[i] for i in x])

    plt.plot(x, y_train_acc, 'bo-', label='train')
    plt.plot(x, y_test_acc, 'gs-', label='test')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig('%s_acc_%s' % (opt.task, datetime.today().date()))
    plt.show()


if opt.task == 'cls':
    train_loss, test_loss, train_acc, test_acc = PointNetCls()
    # print(train_loss.keys(), train_loss.values())
    # print(train_acc.keys(), train_acc.values())
    show_loss(train_loss, test_loss)
    print(train_acc, test_acc)
    show_acc(train_acc, test_acc)
elif opt.task == 'seg':
    train_loss, test_loss, train_acc, test_acc = PointNetSeg()
    show_loss(train_loss, test_loss)
    show_acc(train_acc, test_acc)
else:
    exit("No achieve, maybe update later...")













