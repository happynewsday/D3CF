import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim #feature_dim=128
        self.cluster_num = class_num #cluster_num=20
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),#Linear(512,512)
            nn.ReLU(),#ReLU()
            nn.Linear(self.resnet.rep_dim, self.feature_dim),#Linear(512,128)
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),#Linear(512,512)
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),#Linear(512,20)
            nn.Softmax(dim=1)
        )
    def forward(self, x_i, x_j):#x_i={Tensor:(128,3,224,224)}, x_j={Tensor:(128,3,224,224)}
        h_i = self.resnet(x_i)#h_i={Tensor:(128,512)}
        h_j = self.resnet(x_j)#h_j={Tensor:(128,512)}

        z_i = normalize(self.instance_projector(h_i), dim=1)#{Tensor:(128,128)}
        z_j = normalize(self.instance_projector(h_j), dim=1)#{Tensor:(128,128)}

        c_i = self.cluster_projector(h_i)#c_i={Tensor:(128,20)}
        c_j = self.cluster_projector(h_j)#c_j={Tensor:(128,20)}

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
