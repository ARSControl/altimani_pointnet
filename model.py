import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=512):
        super(Tnet, self).__init__()

        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.num_points = num_points

    def forward(self, x):
        bs = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.linear1(x)))
        x = F.relu(self.bn5(self.linear2(x)))
        x = self.linear3(x)

        I = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            I = I.cuda()

        x = x.view(-1, self.dim, self.dim) + I

        return x

class PointNetBackbone(nn.Module):
    def __init__(self, num_points=512, num_global_feats=512, local_feat=False):
        super(PointNetBackbone, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(f"Expected input of shape [batch_size, 3, num_points], got {x.shape}")
        
        bs = x.shape[0]

        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        A_feat = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        local_features = x.clone()
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        x = torch.max(x, 2, keepdim=True)[0]
        global_features = x.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_features, 
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                                  dim=1)
            return features, A_feat
        else:
            return global_features, A_feat

class PointNetClassHead(nn.Module):
    def __init__(self, num_points=512, num_global_feats=1024, k=10):
        super(PointNetClassHead, self).__init__()

        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x, A_feat = self.backbone(x)

        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        return x, A_feat
