import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import diffdist
from torch.nn.functional import normalize
import torch.distributed as dist

def find_hard_negatives(logits):
    """Finds the top n_hard hardest negatives in the queue for the query.

    Args:
        logits (torch.tensor)[batch_size, len_queue]: Output dot product negative logits.

    Returns:
        torch.tensor[batch_size, n_hard]]: Indices in the queue.
    """
    # logits -> [batch_size, len_queue]
    #_, idxs_hard = torch.topk(logits.clone().detach(), k=self.n_hard, dim=-1, largest=True, sorted=False)
    n_hard = 32
    _, idxs_hard = torch.topk(logits.clone().detach(), k=n_hard, dim=-1, largest=True, sorted=False)
    # idxs_hard -> [batch_size, n_hard]

    return idxs_hard


def hard_negatives1(out_q, logits, idxs_hard):
    """Concats type 1 hard negatives to logits.

    Args:
        out_q (torch.tensor)[batch_size, d_out]: Output of query encoder.
        logits (torch.tensor)[batch_size, len_queue + ...]: Output dot product logits.
        idxs_hard (torch.tensor)[batch_size, n_hard]: Indices of hardest negatives
            in the queue for each query.

    Returns:
        (torch.tensor)[batch_size, len_queue + ... + s1_hard]: logits concatenated with
            type 1 hard negatives.
    """
    # out_q -> [batch_size, d_out]
    # logits -> [batch_size, len_queue + ...]
    # idxs_hard -> [batch_size, n_hard]
    batch_size, device = out_q.shape[0], out_q.device
    n_hard = int(32)
    s1_hard = int(32/2)
    T = 1.0
    idxs1, idxs2 = torch.randint(0, n_hard, size=(2, batch_size, s1_hard), device=device)
    # idxs1, idxs2 -> [batch_size, s1_hard]
    alpha = torch.rand(size=(batch_size, s1_hard, 1), device=device)
    # alpha -> [batch_size, s1_hard, 1]

    neg1_hard = out_q[torch.gather(idxs_hard, dim=1, index=idxs1)]
    neg2_hard = out_q[torch.gather(idxs_hard, dim=1, index=idxs2)]
    #neg1_hard = out_q[torch.gather(idxs_hard, dim=1, index=idxs1)].clone().detach()
    #neg2_hard = out_q[torch.gather(idxs_hard, dim=1, index=idxs2)].clone().detach()
    # neg1_hard, neg2_hard -> [batch_size, s1_hard, d_out]

    neg_hard = alpha * neg1_hard + (1 - alpha) * neg2_hard
    #neg_hard = F.normalize(neg_hard, dim=-1).detach()
    neg_hard = F.normalize(neg_hard, dim=-1)
    # neg_hard -> [batch_size, s1_hard, d_out]

    logits_hard = torch.einsum('b d, b s d -> b s', out_q, neg_hard) / T
    # logits_hard -> [batch_size, s1_hard]

    logits = torch.cat([logits, logits_hard], dim=1)
    # logits -> [batch_size, len_queue + ... + s1_hard]

    return logits


def hard_negatives2(out_q, logits, idxs_hard):
    """Concats type 2 hard negatives to logits.

    Args:
        out_q (torch.tensor)[batch_size, d_out]: Output of query encoder.
        logits (torch.tensor)[batch_size, len_queue + ...]: Output dot product logits.
        idxs_hard (torch.tensor)[batch_size, n_hard]: Indices of hardest negatives
            in the queue for each query.

    Returns:
        (torch.tensor)[batch_size, len_queue + ... + s2_hard]: logits concatenated with
            type 2 hard negatives.
    """
    # out_q -> [batch_size, d_out]
    # logits -> [batch_size, len_queue + ...]
    # idxs_hard -> [batch_size, n_hard]
    batch_size, device = out_q.shape[0], out_q.device
    n_hard = int(16)
    s2_hard = int(8)
    T = 1.0
    beta_hard = 0.5
    idxs = torch.randint(0, n_hard, size=(batch_size, s2_hard), device=device)
    # idxs -> [batch_size, s2_hard]
    beta = torch.rand(size=(batch_size, s2_hard, 1), device=device) * beta_hard
    # beta -> [batch_size, s2_hard, 1]

    neg_hard = out_q[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
    # neg_hard -> [batch_size, s2_hard, d_out]
    neg_hard = beta * out_q.clone().detach()[:, None] + (1 - beta) * neg_hard
    neg_hard = F.normalize(neg_hard, dim=-1).detach()
    # neg_hard -> [batch_size, s2_hard, d_out]

    logits_hard = torch.einsum('b d, b s d -> b s', out_q, neg_hard) / T
    # logits_hard -> [batch_size, s2_hard]

    logits = torch.cat([logits, logits_hard], dim=1)
    # logits -> [batch_size, len_queue + ... + s2_hard]

    return logits

def C3_loss(z_i, z_j, batch_size, zeta): #

    z = torch.cat((z_i, z_j), dim=0) #
    multiply = torch.matmul(z, z.T) #

    a = torch.ones([batch_size])
    mask = 2 * (torch.diag(a, -batch_size) + torch.diag(a, batch_size) + torch.eye(2 * batch_size))
    mask = mask.cuda()

    exp_mul = torch.exp(multiply)
    numerator = torch.sum(torch.where((multiply + mask) > zeta, exp_mul, torch.zeros(multiply.shape).cuda()), dim=1)

    #method1
    #new_mask = torch.where((multiply + mask) > zeta, torch.ones(multiply.shape).cuda(), torch.zeros(multiply.shape).cuda())
    #my_mask = 1 - new_mask
    #exp_mul = exp_mul * my_mask
    den = torch.sum(exp_mul, dim=1)
    #method2
    #idxs_hard = find_hard_negatives(multiply)
    #logits_neg = hard_negatives1(z, multiply, idxs_hard)
    #logits_neg = torch.exp(logits_neg)
    #den = torch.sum(logits_neg, dim=1)

    #print("numerator====>>>",numerator)
    return -torch.sum(torch.log(torch.div(numerator, den))) / batch_size

def C3_loss_tensorboard(z_i, z_j, batch_size, zeta): #

    z = torch.cat((z_i, z_j), dim=0) #
    multiply = torch.matmul(z, z.T) #

    a = torch.ones([batch_size])
    mask = 2 * (torch.diag(a, -batch_size) + torch.diag(a, batch_size) + torch.eye(2 * batch_size))
    mask = mask.cuda()

    exp_mul = torch.exp(multiply)
    numerator = torch.sum(torch.where((multiply + mask) > zeta, exp_mul, torch.zeros(multiply.shape).cuda()), dim=1)

    positive_pairs = torch.where((multiply + mask) > zeta, torch.ones(multiply.shape).cuda(), torch.zeros(multiply.shape).cuda())

    positive_pairs_num = torch.sum(positive_pairs,dim=1)
    positive_pairs_mean = torch.mean(positive_pairs_num)
    #method1
    #new_mask = torch.where((multiply + mask) > zeta, torch.ones(multiply.shape).cuda(), torch.zeros(multiply.shape).cuda())
    #my_mask = 1 - new_mask
    #exp_mul = exp_mul * my_mask
    den = torch.sum(exp_mul, dim=1)
    #method2
    #idxs_hard = find_hard_negatives(multiply)
    #logits_neg = hard_negatives1(z, multiply, idxs_hard)
    #logits_neg = torch.exp(logits_neg)
    #den = torch.sum(logits_neg, dim=1)

    #print("numerator====>>>",numerator)
    return -torch.sum(torch.log(torch.div(numerator, den))) / batch_size, positive_pairs_mean



class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=1, distributed=False, cluster_num=10, alpha=0.9, gamma=0.5):
        super().__init__() #alpha=0.99, cluster_num=10, distributed=False, gamma=0.5
        self.multiplier = multiplier #multiplier=1
        self.distributed = distributed
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma


    @torch.no_grad() #c=Tensor:(128,10), pseudo_label_cur=Tensor:(128,) [-1,-1,...-1],
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        if self.distributed: #index={Tensor:(128,)} tensor([23416, 51728, 53741, ..., 40933])
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pseudo_label_cur_list = [torch.zeros_like(pseudo_label_cur) for _ in range(dist.get_world_size())]
            index_list = [torch.zeros_like(index) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            pseudo_label_cur_list = diffdist.functional.all_gather(pseudo_label_cur_list, pseudo_label_cur)
            index_list = diffdist.functional.all_gather(index_list, index)
            c = torch.cat(c_list, dim=0,)
            pseudo_label_cur = torch.cat(pseudo_label_cur_list, dim=0,)
            index = torch.cat(index_list, dim=0,)
        batch_size = c.shape[0] #batch_size=128
        device = c.device #device=cuda:0, pseudo_label_nxt=Tensor:(128,) [-1,-1,...-1]
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device) #tmp={Tensor:(128,)} [0,1,2...27]

        prediction = c.argmax(dim=1) #prediction={Tensor:(128,)} [9, 0, 9, 5, 9, 5, 0, 7, 4,]
        confidence = c.max(dim=1).values #confidence=Tensor:(128,) [0.9477, 0.4841,...0.9945]
        #print("4confidence================>>>>",confidence)
        unconfident_pred_index = confidence < self.alpha #Tensor:(128,) [True,False,True...]
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(int)
        for i in range(self.cluster_num): #pseudo_per_class=7
            class_idx = prediction == i #class_idx=Tensor:(128,) [False,False,True,False....]
            if class_idx.sum() == 0:
                continue #confidence_class=Tensor:(10,)[0.4447, 0.5127... 0.5379,0.2294])
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class) #num=7
            confident_idx = torch.argsort(-confidence_class) #tensor(10,) [7, 6, 4, 5,...9])
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]] #such as idx=72,49,33,45,14
                pseudo_label_nxt[idx] = i
        #pseudo_label_nxt=Tensor:(128,) [-1, -1, 5, -1, -1, ....0, 7, 4,-1, 0, -1, 1, 9]
        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    #c = Tensor:(128,10), pseudo_label=Tensor:(128,)
    def forward(self, c, pseudo_label, pesudo_label_all):
        if self.distributed:
            # c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pesudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # c_list = diffdist.functional.all_gather(c_list, c)
            pesudo_label_list = diffdist.functional.all_gather(
                pesudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            # c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
            pesudo_label_list = [
                chunk for x in pesudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            # c_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    # c_sorted.append(c_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pesudo_label_list[i * self.multiplier + m]
                    )#idx=Tensor:(10,) [0,1,2,3...9],counts=Tensor:(10,) [ 318,71, ...1742])
            # c = torch.cat(c_sorted, dim=0)
            pesudo_label_all = torch.cat(pesudo_label_sorted, dim=0)
        pseudo_index = pesudo_label_all != -1 #Tensor:(60000,) [False,False,True,...]
        pesudo_label_all = pesudo_label_all[pseudo_index] #Tensor:(3427,), [9,5,9,0,4...]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()#Tensor:(10,) [10.7767,48.2676...1.9673]
        weight = torch.ones(self.cluster_num).to(c.device)#Tensor:(10,) [1.,1.,...1.]
        weight[idx] = freq.to(c.device)#Tensor:(10,) [10.7767, 48.2676, 17.6649,...1.9673]
        pseudo_index = pseudo_label != -1 #Tensor:(128,) [False,False,...True,...False]
        #print("2weight===========>>>", weight)
        if pseudo_index.sum() > 0:
            #print("3pseudo_index.sum()===========>>>", pseudo_index.sum())
            criterion = nn.CrossEntropyLoss(weight=weight).to(c.device)
            loss_ce = criterion(c[pseudo_index], pseudo_label[pseudo_index].to(c.device))
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(c.device)
        return loss_ce

class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        if self.distributed:
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pseudo_label_cur_list = [torch.zeros_like(pseudo_label_cur) for _ in range(dist.get_world_size())]
            index_list = [torch.zeros_like(index) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            pseudo_label_cur_list = diffdist.functional.all_gather(pseudo_label_cur_list, pseudo_label_cur)
            index_list = diffdist.functional.all_gather(index_list, index)
            c = torch.cat(c_list, dim=0,)
            pseudo_label_cur = torch.cat(pseudo_label_cur_list, dim=0,)
            index = torch.cat(index_list, dim=0,)
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(int)
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z, pseudo_label):
        n = z.shape[0]
        assert n % self.multiplier == 0

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            pseudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            pseudo_label_list = diffdist.functional.all_gather(
                pseudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            pseudo_label_list = [
                chunk for x in pseudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pseudo_label_list[i * self.multiplier + m]
                    )
            z_i = torch.cat(
                z_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            z_j = torch.cat(
                z_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )
            pseudo_label = torch.cat(pesudo_label_sorted, dim=0,)
            n = z_i.shape[0]

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(z_i.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).to(z_i.device),
            0,
        )
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss
