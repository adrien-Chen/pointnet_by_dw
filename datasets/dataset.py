import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import os.path
from tqdm import tqdm
import json


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'trainval.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ModelNetDataset(Dataset):
    """ ModelNet40 Dataset
    data split using trainval.txt & test.txt
      num of files: 12311
      num of trainval: 9843
      num of test: 2468
    """

    def __init__(self, root_dir, num_points=2500, split='train', data_augmentation=True):
        """

        :param root_dir: dir of data root,
        :param num_points: num of sampled points
        :param split: the name of split file used to train or just test
        """
        self.npts = num_points
        self.root = root_dir
        self.split = split
        self.data_augmentation = data_augmentation
        self.filenames = []
        self.cat = {}
        with open(os.path.join(root_dir, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.filenames.append(line.strip())  # 'car/car_0033.txt'

        # ur project/misc/modelnet_id.txt
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                name, idx = line.strip().split()
                self.cat[name] = int(idx)  # like ['airplane': 0]

        # print("All Classes, [name, index]: \n", self.cat)
        self.classes = list(self.cat.keys())

    def __len__(self):
        return len(self.filenames)  # nums of data files

    def __getitem__(self, index):
        """

        :param index: the index of dataset, dataset[i] can be used to get i_th sample
        :return: point_set, classes
        """
        filename = self.filenames[index]
        cls = self.cat[filename.split('/')[0]]
        # read data of this file:
        points = []
        with open(os.path.join(self.root, filename), 'r') as f:
            for line in f:
                x, y, z = line.strip().split(',')[0:3]
                points.append([x, y, z])
        points = np.asarray(points).astype(np.float32)
        assert points.shape == (len(points), 3)

        sample_id = np.random.choice(len(points), self.npts, replace=True)  # random sampling: default=2500
        point_set = points[sample_id, :] # [N, 3]

        # normalization:
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # minus center point
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)  # [N, 1]->>1
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)  # sample in a uniform distribution
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])  # rotate over y-axis, just change xz
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))  # [N, 3]
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))  # [int]
        return point_set, cls


def gen_shapenet_id(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            name, idx = line.strip().split()
            cat[name] = idx

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        filenames = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in filenames:
            token = os.path.splitext(os.path.basename(fn))[0]
            meta[item].append((os.path.join(dir_point, token+'.pts'),
                              os.path.join(dir_seg, token+'.seg')))

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))  # (car, points_file, seg_file)

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


class ShapeNetDataset(Dataset):
    def __init__(self, root, num_points=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npt = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                name, idx = line.strip().split()
                self.cat[name] = idx

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))

        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, id = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', id+'.pts'),
                                                         os.path.join(self.root, category, 'points_label', id+'.seg')))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))  # cls_1: 0 | cls_2: 1 | ...
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                item, num_seg_classes = line.strip().split()
                self.seg_classes[item] = int(num_seg_classes)
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]  # idx of one class
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        sample_id = np.random.choice(len(seg), self.npt, replace=True)
        # resample:
        point_set = point_set[sample_id, :]

        # normalization:
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])  # rotate over y-axis, just change x-z
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[sample_id]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg


if __name__ == '__main__':
    datapath = '/home/dw/Documents/all_data/data/modelnet40_normal_resampled'
    datapath2 = '/home/dw/Documents/all_data/data/shapenetcore_partanno_segmentation_benchmark_v0'

    # gen_modelnet_id(datapath)
    # data = ModelNetDataset(root_dir=datapath, split='trainval')
    # print(len(data))
    # points_0, cls_0 = data[0]
    # print(points_0.shape)
    # print(cls_0)

    gen_shapenet_id(datapath2)
    data = ShapeNetDataset(root=datapath2, class_choice=['Chair'])
    print(len(data))
    ps, seg = data[0]
    print(ps.size(), ps.type(), seg.size(), seg.type())





