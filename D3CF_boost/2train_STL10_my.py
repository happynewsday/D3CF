import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict

from modules import transform, resnet, network,contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation
from modules.contrastive_loss import ClusterLossBoost
from modules.misc import NativeScalerWithGradNormCount as NativeScaler
import modules.misc as misc
from modules.data import build_dataset, CIFAR10,CIFAR100,ImageFolder,STL10
import math
import sys
import torch.nn.functional as F

def train_net(model, unlabel_loader, data_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels, batch_size, zeta):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    for data_iter_step, ((x_w, x_s), _, index) in enumerate(unlabel_loader):
        optimizer.zero_grad()
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        model.train(True)
        for param in model.parameters():
            param.requires_grad = True

        with torch.cuda.amp.autocast():
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            #loss_ins = criterion_ins(torch.concat((z_i, z_j), dim=0), pseudo_labels[index].to(x_s.device))
            loss_unlabel = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)
            loss = loss_unlabel

        loss_unlabel_value = loss_unlabel.item()

        if not math.isfinite(loss_unlabel_value):
            print("Loss is {}, stopping training".format(loss_unlabel_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        metric_logger.update(loss_unlabel=loss_unlabel_value)

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


#attention dataset_size, batch_size, test_batch_size, max_epochs
def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook.yaml_config_hook("config/config.yaml") #config={dict:18} {'seed': 42, 'workers': 8, 'dataset_dir': './datasets', 'dataset_size': 60000, 'class_num': 10, 'batch_size': 128, 'test_batch_size': 500, 'image_size': 224, 'start_epoch': 0, 'max_epochs': 20, 'dataset': 'CIFAR-10', 'resnet': 'ResNet34', 'feature_dim': 128, 'model_path': 'save/CIFAR-10', 'reload': False, 'learning_rate': 1e-05, 'weight_decay': 0.0, 'zeta': 0.6}

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')
    clu_temp = args.clu_temp

    # prepare data start==========================================================
    if args.dataset == "CIFAR-10":
        args.class_num = 10
        train_dataset = CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

        test_dataset = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

        train_dataset = data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        train_dataset_test = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )

        test_dataset_test = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = data.ConcatDataset([train_dataset_test, test_dataset_test])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size=len(test_dataset)

    elif args.dataset == "CIFAR-100":
        args.class_num = 20
        train_dataset = CIFAR100(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

        test_dataset = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

        train_dataset = data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        train_dataset_test = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )

        test_dataset_test = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = data.ConcatDataset([train_dataset_test, test_dataset_test])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)
    elif args.dataset == "ImageNet-10":
        args.class_num = 10
        train_dataset = ImageFolder(root=args.dataset_dir + "ImageNet-10", transform=transform.Transforms(size=args.image_size, blur=True))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "ImageNet-10", transform=transform.Transforms(size=args.image_size).test_transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)
    elif args.dataset == "ImageNet-dogs":
        args.class_num = 15
        train_dataset = ImageFolder(root=args.dataset_dir + "imagenet-dogs", transform=transform.Transforms(size=args.image_size, blur=True))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "imagenet-dogs", transform=transform.Transforms(size=args.image_size).test_transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)
    elif args.dataset == "tiny-ImageNet":
        args.class_num = 200
        train_dataset = ImageFolder(root=args.dataset_dir + "tiny-imagenet-200/train", transform=transform.Transforms(s=0.5, size=args.image_size))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "tiny-imagenet-200/train", transform=transform.Transforms(size=args.image_size).test_transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)

    elif args.dataset == "STL-10":
        args.class_num=10
        train_dataset = STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        test_dataset = STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        train_dataset_test = STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset_test = STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )

        test_dataset = torch.utils.data.ConcatDataset([train_dataset_test, test_dataset_test])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        unlabel_dataset = STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )

        unlabel_loader = torch.utils.data.DataLoader(
            dataset=unlabel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True)

        args.dataset_size = len(test_dataset)
    else:
        raise NotImplementedError

    # Initializing our network with a network trained with CC -------------------------------------------------------------------------------------------------------
    res = resnet.get_resnet(args.resnet)
    net = network.Network(res, args.feature_dim, args.class_num)
    net = net.to('cuda')
    model_fp = os.path.join(args.model_path, "checkpoint_1000.tar")
    checkpoint = torch.load(model_fp, map_location=torch.device('cuda:0'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    #net.load_state_dict(new_state_dict)
    net.load_state_dict(checkpoint['net'])
    # optimizer ---------------------------------------------------------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion_clu = ClusterLossBoost(distributed=False, cluster_num=args.class_num,alpha=0.99, gamma=0.5)
    pseudo_labels = -torch.ones(train_dataset.__len__(), dtype=torch.long)
    loss_scaler = NativeScaler()
    # train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.max_epochs):
        print("epoch:", epoch)

        evaluation.net_evaluation(net,test_loader,args.dataset_size, args.test_batch_size)
        net, optimizer,train_stats, pseudo_labels = train_net(net, unlabel_loader, train_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels,args.batch_size,args.zeta)

        '''
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        with open('{}_C3_SL_checkpoint_{}'.format(args.model_path,epoch), 'wb') as out:
            torch.save(state, out)
        '''

        save_model.save_model(args, net, optimizer, epoch)

if __name__ == "__main__":
    main()
