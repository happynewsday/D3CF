import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class SoftInstanceLoss(nn.Module):
    def __init__(self, batch_size, device, T1=0.1, T2=0.05, type='ascl', nn_num=1):
        super(SoftInstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.T1 = T1
        self.T2 = T2
        self.type = type
        self.nn_num = nn_num

        self.mask = self.mask_correlated_samples(batch_size)
        #self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        with torch.no_grad():
            N = 2 * self.batch_size
            z = torch.cat((z_i, z_j), dim=0)
            # pseudo logits: NxN, logits_pd={Tensor:(256, 4096)}
            logits_pd = torch.einsum('nc,cm->nm', [z, z.T])
            #logits_pd = torch.matmul(z, z.T)
            logits_pd /= self.T2  # logits_pd={Tensor:(256, 4096)}


            negative_samples = logits_pd[self.mask].reshape(N, -1)
            logits_pd = negative_samples #logits={Tensor:(64,63)}

            labels = torch.zeros(logits_pd.size(0), logits_pd.size(1)+1).cuda()  # labels={Tensor:(256, 4097)} all is 0
            self.max_entropy = np.log(logits_pd.size(1))

            if self.type == 'ascl':
                labels[:, 0] = 1.0 #{Tensor:(256, 4097)} first column is 1
                pseudo_labels = F.softmax(logits_pd, 1) #{Tensor:(256, 4096)}
                log_pseudo_labels = F.log_softmax(logits_pd, 1) #{Tensor:(256, 4096)}
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True) #{Tensor:(256, 1)}
                c = 1 - entropy / self.max_entropy #c={Tensor:(256,1)}
                pseudo_labels = self.nn_num * c * pseudo_labels  # num of neighbors * uncertainty * pseudo_labels, {Tensor:(256, 4096)}
                pseudo_labels = torch.minimum(pseudo_labels,torch.tensor(1).to(pseudo_labels.device))  # upper thresholded by 1, {Tensor:(256, 4096)}
                labels[:, 1:] = pseudo_labels  # summation <= c*K <= K, labels={Tensor:(256, 4097)}, first column is 1

            elif self.type == 'ahcl':
                labels[:, 0] = 1.0
                _, nn_index = logits_pd.topk(self.nn_num, dim=1, largest=True)
                hard_labels = torch.zeros_like(logits_pd, device=logits_pd.device).scatter(1, nn_index, 1)
                pseudo_labels = F.softmax(logits_pd, 1)
                log_pseudo_labels = F.log_softmax(logits_pd, 1)
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
                c = 1 - entropy / self.max_entropy
                labels[:, 1:] = hard_labels * c  # summation = c*K <= K

            elif self.type == 'hard':
                labels[:, 0] = 1.0
                _, nn_index = logits_pd.topk(self.nn_num, dim=1, largest=True)
                hard_labels = torch.zeros_like(logits_pd, device=logits_pd.device).scatter(1, nn_index, 1)
                labels[:, 1:] = hard_labels  # summation = K

            else:  # no extra neighbors [moco]
                labels[:, 0] = 1.0

    # label normalization, labels={Tensor:(256, 4097)}
        labels = labels / labels.sum(dim=1, keepdim=True)

    # forward pass
    # positive logits: Nx1
    #l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        sim = torch.matmul(z, z.T)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    # negative logits: Nx(N-2)
    #l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        negative_samples = sim[self.mask].reshape(N, -1)
    # logits: Nx(N-1)
        logits = torch.cat([positive_samples, negative_samples], dim=1) #{Tensor:(256, 4097)}
        logits /= self.T1

        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()

#========================================================================================
        return loss

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j): #z_i=z_j={Tensor:(32, 128)}
        N = 2 * self.batch_size #N:64
        z = torch.cat((z_i, z_j), dim=0) #z={Tensor:(64, 128)}

        sim = torch.matmul(z, z.T) / self.temperature #sim={Tensor:(64, 64)}
        sim_i_j = torch.diag(sim, self.batch_size) #sim_j_j={Tensor:(32,)}
        sim_j_i = torch.diag(sim, -self.batch_size) #sim_j_i={Tensor:(32,)}

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #positive_samples={Tensor:(64,1)}
        negative_samples = sim[self.mask].reshape(N, -1) #negative_samples={Tensor:(64,62)}

        labels = torch.zeros(N).to(positive_samples.device).long() #labels={Tensor:(64,)}, all element is 0
        logits = torch.cat((positive_samples, negative_samples), dim=1) #logits={Tensor:(64,63)}
        loss = self.criterion(logits, labels) #tensor(262.0677)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j): #c_i=c_j={Tensor:(32, 10)}
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, class_num, temperature, device, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.lambd = lambd

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j): #c_i=c_j={Tensor:(128, 10)}
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()  #c_i={Tensor:(10, 128)}
        c_j = c_j.t() #c_j={Tensor:(10, 128)}
        N = 2 * self.class_num #N=20
        c = torch.cat((c_i, c_j), dim=0) ##c={Tensor:(20, 128)}

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature #sim={Tensor:(20,20)}
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)#positive_clusters={Tensor:(20,1)}
        negative_clusters = sim[self.mask].reshape(N, -1)#negative_clusters={Tensor:(20,18)}

        #labels = torch.zeros(N).to(positive_clusters.device).long()
        #logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        #loss = self.criterion(logits, labels)

        on_diag = positive_clusters.add_(-1).pow_(2).sum()#such as tensor(0.0027)
        off_diag = negative_clusters.pow_(2).sum() #such as tensor(352.35)
        loss = on_diag + self.lambd * off_diag #such as tensor(176.179)
        loss /= N*(N-1)

        return loss + ne_loss


class BarlowTwinsLossTest(nn.Module):
    def __init__(self, class_num, temperature, device, lambd):
        super(BarlowTwinsLossTest, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.lambd = lambd

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j): #c_i=c_j={Tensor:(32, 10)}
        '''
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        '''

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        #labels = torch.zeros(N).to(positive_clusters.device).long()
        #logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        #loss = self.criterion(logits, labels)

        on_diag = positive_clusters.add_(-1).pow_(2).sum()
        off_diag = negative_clusters.pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        loss /= N*(N-1)

        return loss

class SelfClassifier(nn.Module):
    def __init__(self, row_tau=0.1, col_tau=0.1, eps=1e-8):
        super(SelfClassifier, self).__init__()
        self.row_tau = row_tau
        self.col_tau = col_tau
        self.eps = eps

    '''
    such as out={list:1}
    [[
    tensor([[ 0.1293,  0.2273,  0.0090, -0.0811,  0.0849],
            [ 0.1811,  0.1877,  0.0697, -0.0602,  0.1040]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1452,  0.2072,  0.0180, -0.0799,  0.1116],
            [ 0.1846,  0.2106, -0.0172, -0.0682,  0.0942]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1562,  0.2371,  0.0354, -0.0400,  0.0576],
            [ 0.1556,  0.2162,  0.0472, -0.0945,  0.0899]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1848,  0.2036,  0.0461, -0.0470,  0.0500],
            [ 0.1037,  0.2374, -0.0042, -0.1062,  0.0880]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1755,  0.1317,  0.0125, -0.1123,  0.0571],
            [ 0.1680,  0.2172,  0.0358, -0.0721,  0.0515]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1138,  0.1702, -0.0604, -0.0951,  0.0663],
            [ 0.1780,  0.2621,  0.0044, -0.0208,  0.0853]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1540,  0.1828,  0.0395, -0.0834,  0.1629],
            [ 0.1583,  0.2122,  0.0200, -0.1205,  0.1105]],grad_fn=<SliceBackward>), 
    tensor([[ 0.1856,  0.1547,  0.0683, -0.0270,  0.1095],
            [ 0.1436,  0.1454,  0.0279, -0.0480,  0.1161]],grad_fn=<SliceBackward>)]]
    '''
    def forward(self, out):
        total_loss = 0.0
        num_loss_terms = 0

        for cls_idx, cls_out in enumerate(out):  # classifiers, cls_idx=0,cls_out={list:8}
            # gather samples from all workers
            # cls_out = [cls_out]
            const = cls_out[0].shape[0] / cls_out[0].shape[1]#N/C,2/5=0.4,batch_size=2,class_num=5
            target = []

            for view_i_idx, view_i in enumerate(cls_out): #such as view_i_idx=0,view_i=Tensor(2,5)
                view_i_target = F.softmax(view_i / self.col_tau, dim=0)#column softmax,Tensor(2,5)
                # view_i_target = utils.keep_current(view_i_target)
                view_i_target = F.normalize(view_i_target, p=1, dim=1, eps=self.eps)#denominator of second term, Tensor(2,5), is (batch_size, class_num)
                target.append(view_i_target) #{list:8}, target[i]=Tensor:(2,5)

            for view_j_idx, view_j in enumerate(cls_out):# such as view_j_idx=0,view_j=Tensor(2,5)
                view_j_pred = F.softmax(view_j / self.row_tau, dim=1)#row softmax,Tensor(2,5)
                view_j_pred = F.normalize(view_j_pred, p=1, dim=0, eps=self.eps)#denominator of first term,such as Tensor:(2,5)
                # view_j_pred = utils.keep_current(view_j_pred)
                view_j_log_pred = torch.log(const * view_j_pred + self.eps)#Tensor:(2,5)

                for view_i_idx, view_i_target in enumerate(target):

                    if view_i_idx == view_j_idx or (view_i_idx >= 2 and view_j_idx >= 2):
                        # skip cases when it's the same view, or when both views are 'local' (small)
                        continue

                    # cross entropy
                    loss_i_j = - torch.mean(torch.sum(view_i_target * view_j_log_pred, dim=1))
                    total_loss += loss_i_j
                    num_loss_terms += 1

        total_loss /= num_loss_terms

        return total_loss





