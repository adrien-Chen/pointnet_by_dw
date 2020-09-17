from __future__ import  print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.parallel  # for multi gpus
import numpy as np


class T_Net_kd(nn.Module):
    """T-Net :

    predict an affine transformation matrix
    learning the transform(translation / rotation)
    of point cloud or feature.

    Attributes:
        k: is an integer, set 3 for input point cloud and
          64 for feature after MLP.
    """
    def __init__(self, k=3):
        super(T_Net_kd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)  # matrix [k, k]
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # MaxPooling [B, 1024, 1]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))  # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))  # [B, 256]
        x = self.fc3(x) # [B, k*k]
        # print(x.size())

        # bias = 1 ->> identity
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).\
            view(1, self.k*self.k).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x += iden  # Linear func: wx + bias
        x = x.view(-1, self.k, self.k)
        return x


class point_net_feature(nn.Module):
    """Extract feature using MLP + MaxPooling
    Attributes:
        global_feature: return 1*1024 for classification or
          n*1024 for segmentation.
        feature_transform: do transform by T-net or not.
    """
    def __init__(self, global_feature=True, feature_transform=False):
        super(point_net_feature, self).__init__()
        self.tn3d = T_Net_kd()
        self.conv1 = nn.Conv1d(3, 64, 1)  # 1*1 convolution for lift dims
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feature = global_feature
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.tn_feature = T_Net_kd(k=64)

    def forward(self, x):
        num_points = x.size()[2]  # x: tensor, shape [B, 3, N]
        trans_matrix = self.tn3d(x)
        # do transform for points:
        x = x.transpose(2, 1)  # shape [B, N, 3]
        x = torch.bmm(x, trans_matrix)
        x = x.transpose(2, 1)  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        if self.feature_transform:
            trans_matrix_feat = self.tn_feature(x)
            x = x.transpose(2, 1)  # shape [B, N, 64]
            x = torch.bmm(x, trans_matrix_feat)
            x = x.transpose(2, 1)  # [B, 64N]
        else:
            trans_matrix_feat = None

        points_feature = x  # [B, 64, N]
        # print("points_feature:", points_feature.shape)
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.bn3(self.conv3(x))  # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # MaxPooling,(values, indices)
        x = x.view(-1, 1024)  # global feature [B*1, 1024]
        if self.global_feature:
            return x, trans_matrix, trans_matrix_feat
        else:
            # print(x.size())
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)  # [B, 1024, N]
            # print(x.size())
            return torch.cat([x, points_feature], 1), trans_matrix, trans_matrix_feat


class point_net_cls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(point_net_cls, self).__init__()
        self.feature_transform = feature_transform
        self.feature = point_net_feature(global_feature=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_matrix, trans_matrix_feat = self.feature(x)
        x = F.relu(self.bn1(self.fc1(x)))  # [B, 512]
        x = F.relu(self.bn2(self.fc2(x)))  # [B, 256]
        x = self.fc3(x)  # [B, k=2]
        return F.log_softmax(x, dim=1), trans_matrix, trans_matrix_feat


class point_net_seg(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(point_net_seg, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feature = point_net_feature(global_feature=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.size()[0]
        num_points = x.size()[2]
        x, trans_matrix, trans_matrix_feat = self.feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
        x = self.conv4(x)  # [B, k=2, N]
        x = x.transpose(2, 1).contiguous()  # [B, N, 2]
        x = x.view(-1, self.k)  # [B*N, 2]
        x = F.log_softmax(x, dim=-1)
        x = x.view(batch_size, num_points, self.k)
        return x, trans_matrix, trans_matrix_feat

def feature_transform_regularizer(trans):
    """

    :param trans: [B, k, k]
    """
    dims = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(dims)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = T_Net_kd()
    trans_matrix = trans(sim_data)
    print("T-Net:", trans_matrix.size())

    point_feature = point_net_feature(global_feature=True)
    out, _, _ = point_feature(sim_data)
    print("global feature:", out.size())

    point_feature = point_net_feature(global_feature=False)
    out, _, _ = point_feature(sim_data)
    print("point feature:", out.size())

    cls = point_net_cls(k=5)
    out, _, _ = cls(sim_data)
    print("class:", out.size())

    seg = point_net_seg(k=3)
    out, _, _ = seg(sim_data)
    print("seg:", out.size())




