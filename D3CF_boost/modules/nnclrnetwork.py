import torch.nn as nn
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            #nn.Softmax(dim=1)
        )

        #NNCLR project head!
        self.projection_head = NNCLRProjectionHead(self.resnet.rep_dim, self.resnet.rep_dim,self.feature_dim)
        self.prediction_head = NNCLRPredictionHead(self.feature_dim, self.resnet.rep_dim, self.feature_dim)

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c=F.softmax(c,dim=1)
        c = torch.argmax(c, dim=1)
        return c
