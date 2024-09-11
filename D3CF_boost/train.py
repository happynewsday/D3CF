import numpy as np
import torch
import torch.nn as nn
import diffdist
from modules import transform, resnet, network, contrastive_loss
from torch.nn.functional import normalize
import torch.distributed as dist
import modules.misc as misc
import math
import sys
import torch.nn.functional as F


def train_net(model, data_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels, batch_size, zeta):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20


    #for data_iter_step, ((x_w, x_s, x), _, index) in enumerate(metric_logger.log_every(data_loader,print_freq, header)):
    for data_iter_step, ((x_w, x_s), _, index) in enumerate(data_loader):
        optimizer.zero_grad()

        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        #x = x.to(device, non_blocking=True)

        model.eval()
        with torch.cuda.amp.autocast(), torch.no_grad():
            _, _, c,_ = model(x_w, x_w)
            c = F.softmax(c / clu_temp, dim=1)
            pseudo_labels_cur, index_cur = criterion_clu.generate_pseudo_labels(
                c, pseudo_labels[index].to(c.device), index.to(c.device))
            pseudo_labels[index_cur] = pseudo_labels_cur
            pseudo_index = pseudo_labels != -1
            #print("1pseudo_index.sum().item()===================>>>>",pseudo_index.sum().item())
            metric_logger.update(pseudo_num=pseudo_index.sum().item())
            metric_logger.update(pseudo_cluster=torch.unique(pseudo_labels[pseudo_index]).shape[0])
        if epoch == 0:
            continue

        model.train(True)
        for param in model.parameters():
            param.requires_grad = True

        with torch.cuda.amp.autocast():
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            #loss_ins = criterion_ins(torch.concat((z_i, z_j), dim=0), pseudo_labels[index].to(x_s.device))
            loss_ins = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)
            loss_clu = criterion_clu(c_j, pseudo_labels[index].to(x_s.device), pseudo_labels)
            if epoch<=20:
                loss = loss_ins
            else:
                loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print("Loss is {}, {}, stopping training".format(loss_ins_value, loss_clu_value))
            sys.exit(1)
        #loss_scaler(loss,optimizer,parameters=model.parameters(),create_graph=False,update_grad=True,)
        loss.backward()
        optimizer.step()

        #torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (model, optimizer,{k: meter.global_avg for k, meter in metric_logger.meters.items()},pseudo_labels)




