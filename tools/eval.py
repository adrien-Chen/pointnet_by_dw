from __future__ import  print_function
import argparse
import os
import random
from tqdm import tqdm
from datetime import datetime

import torch.utils.data
import numpy as np
from pointnet.model import point_net_cls, point_net_seg, feature_transform_regularizer
from datasets.dataset import ModelNetDataset, ShapeNetDataset

# ====================
# set commend params
# ====================
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
parse.add_argument('--model', type=str, required=True, default='', help='set model path')
parse.add_argument('--dataset', type=str, required=True, help='set dataset path')
parse.add_argument('--dataset_type', type=str, required=True, default='modelnet40', help='type of your dataset')
parse.add_argument('--feature_transform', action='store_true', help='use feature transform')  # true of false
parse.add_argument('--class_choice', type=str, default='Chair', help='class choice for segmentation at one time')

# if try to use shapenet for training, u should first set the '--task' for the model
parse.add_argument('--task', type=str, required=True, help='the task of the model')
# add logs:
parse.add_argument('--logs', type=str, default='eval_logs', help='the dir of log')
# show parse:
opt = parse.parse_args()
print(opt)

# ====================
# check dataset type:
# ====================
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

# ====================
# create data iterators for training set and test set:
# ====================
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

try:
    os.makedirs(opt.outf)  # write output ?
except OSError:
    pass


# ====================
# Evaluation
# ====================
# num of classes:
if opt.dataset_type == 'shapenet' and opt.task == 'seg':
    num_classes = dataset.num_seg_classes  # for ShapeNet
elif opt.dataset_type == 'shapenet' and opt.task == 'cls':
    num_classes = len(dataset.classes)
else:
    num_classes = len(dataset.classes)  # ModelNet40

def eval_one_epoch():
    if opt.task == 'cls':
        classifier = point_net_cls(k=num_classes, feature_transform=opt.feature_transform)  # default is False!
        if opt.model == '':
            return "no model loading!"
        classifier.load_state_dict(torch.load(opt.model))  # if you have trained model params
        classifier.cuda()  # model load to gpu

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
        return total_correct / float(total_testset)

    elif opt.task == 'seg':
        classifier = point_net_seg(num_classes, feature_transform=opt.feature_transform)

        if opt.model == '':
            return "no model loading!"
        classifier.load_state_dict(torch.load(opt.model))  # if you have trained model params
        classifier.cuda()  # model load to gpu

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
        return opt.class_choice, np.mean(shape_ious)
    else:
        return "wrong task type!"
        exit("wrong task type!")

def eval():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
    dir = os.path.join(dir, opt.logs)
    print(dir)
    if not os.path.exists(dir):  # if folder no exits
        os.mkdir(dir)
    path = os.path.join(dir, '%s_log_%s.txt' % (opt.task, str(datetime.today().date())))
    print(path)
    with open(path, mode='a+') as log_file:
        if opt.task == 'cls':
            print('====== Logging ACC of Classification ======')
            acc = eval_one_epoch()
            log_file.write(str(datetime.now())+'\n'+str(acc)+'\n\n')
            log_file.close()
        else:
            print('====== Logging ACC of Segmentation ======')
            name_class, m_iou = eval_one_epoch()
            log_file.write(str(datetime.now())+'\n' + name_class + ':\t' + str(m_iou) + '\n')

        print("finished writting!")



if __name__ == '__main__':
    eval()