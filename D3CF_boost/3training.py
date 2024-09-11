import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict

from modules import transform,tcltransforms,gcctransform, resnet, network
from utils import yaml_config_hook, save_model
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation, ccevaluation
from modules.contrastive_loss import ClusterLossBoost,C3_loss,C3_loss_tensorboard
from modules.misc import NativeScalerWithGradNormCount as NativeScaler
import modules.misc as misc
from modules.data import build_dataset, CIFAR10,CIFAR100,ImageFolder,STL10
import math
import sys
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
import modules.misc as misc

writer = SummaryWriter(log_dir='cifar10_logs2/', comment='ablation', filename_suffix='b128_z0.4_a0.99_g0.5')

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y,_) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def train_net(model, data_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels, batch_size, zeta,clu_begin_epoch):

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
            loss_ins, positive_pairs_mean = C3_loss_tensorboard(z_i, z_j, batch_size, zeta)
            loss_clu = criterion_clu(c_j, pseudo_labels[index].to(x_s.device), pseudo_labels)
            if epoch<=clu_begin_epoch:
                loss = loss_ins
            else:
                loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()
        positive_pairs_mean = positive_pairs_mean.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print("Loss is {}, {}, stopping training".format(loss_ins_value, loss_clu_value))
            sys.exit(1)
        #loss_scaler(loss,optimizer,parameters=model.parameters(),create_graph=False,update_grad=True,)
        loss.backward()
        optimizer.step()

        #torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)
        metric_logger.update(positive_pairs_mean=positive_pairs_mean)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (model, optimizer,{k: meter.global_avg for k, meter in metric_logger.meters.items()},pseudo_labels)

#attention dataset_size, batch_size, test_batch_size, max_epochs
def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook.yaml_config_hook("config/config_cifar10_2.yaml") #config={dict:18} {'seed': 42, 'workers': 8, 'dataset_dir': './datasets', 'dataset_size': 60000, 'class_num': 10, 'batch_size': 128, 'test_batch_size': 500, 'image_size': 224, 'start_epoch': 0, 'max_epochs': 20, 'dataset': 'CIFAR-10', 'resnet': 'ResNet34', 'feature_dim': 128, 'model_path': 'save/CIFAR-10', 'reload': False, 'learning_rate': 1e-05, 'weight_decay': 0.0, 'zeta': 0.6}

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
    args.alpha = 0.99
    args.gamma = 0.5
    args.clu_begin_epoch = 20
    # prepare data start==========================================================
    if args.dataset == "CIFAR-10":
        args.class_num = 10
        train_dataset = CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            #transform=transform.Transforms(size=args.image_size, s=0.5),
            transform=tcltransforms.build_transform(True, args),
        )

        test_dataset = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            #transform=transform.Transforms(size=args.image_size, s=0.5),
            transform=tcltransforms.build_transform(True, args),
        )

        train_dataset = data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)


        train_dataset_test = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )

        test_dataset_test = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
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
            #transform=transform.Transforms(size=args.image_size, s=0.5),
            transform=tcltransforms.build_transform(True,args),
        )

        test_dataset = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            #transform=transform.Transforms(size=args.image_size, s=0.5),
            transform=tcltransforms.build_transform(True,args),
        )

        train_dataset = data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)


        train_dataset_test = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )

        test_dataset_test = CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )
        test_dataset = data.ConcatDataset([train_dataset_test, test_dataset_test])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)

    elif args.dataset == "ImageNet-10":
        args.class_num = 10
        train_dataset = ImageFolder(
            root=args.dataset_dir + "/imagenet-10",
            #transform=transform.Transforms(size=args.image_size, blur=True),
            transform=tcltransforms.build_transform(True, args),
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)


        test_dataset = ImageFolder(
            root=args.dataset_dir + "/imagenet-10",
            #transform=transform.Transforms(size=args.image_size).test_transform
            transform=tcltransforms.build_transform(False, args),
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)

    elif args.dataset == "ImageNet-dogs":
        args.class_num = 15
        train_dataset = ImageFolder(
            root=args.dataset_dir + "/imagenet-dogs",
            #transform=transform.Transforms(size=args.image_size, blur=True),
            transform = tcltransforms.build_transform(True, args),
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)


        test_dataset = ImageFolder(
            root=args.dataset_dir + "/imagenet-dogs",
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)

    elif args.dataset == "tiny-ImageNet":
        args.class_num = 200
        #temp = gcctransform.GCCTransforms()
        train_dataset = ImageFolder(
            root=args.dataset_dir + "/tiny-imagenet-200/train",
            #transform=transform.Transforms(s=0.5, size=args.image_size),
            transform=gcctransform.GCCTransforms(),
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)


        test_dataset = ImageFolder(
            root=args.dataset_dir + "/tiny-imagenet-200/train",
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=gcctransform.GCCTransforms().test_transform,
        )
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
            #transform=transform.Transforms(size=args.image_size),
            transform=tcltransforms.build_transform(True, args),
        )
        test_dataset = STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            #transform=transform.Transforms(size=args.image_size),
            transform=tcltransforms.build_transform(True, args),
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        train_dataset_test = STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )
        test_dataset_test = STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            #transform=transform.Transforms(size=args.image_size).test_transform,
            transform=tcltransforms.build_transform(False, args),
        )

        test_dataset = torch.utils.data.ConcatDataset([train_dataset_test, test_dataset_test])
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)

    else:
        raise NotImplementedError

    # prepare data end==========================================================
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
    criterion_clu = ClusterLossBoost(distributed=False, cluster_num=args.class_num,alpha=args.alpha, gamma=args.gamma)
    pseudo_labels = -torch.ones(train_dataset.__len__(), dtype=torch.long)
    loss_scaler = NativeScaler()
    # train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.max_epochs):
        print("epoch:", epoch)
        X, Y = inference(test_loader, net, device)
        if args.dataset == "CIFAR-100":  # super-class
            super_label = [
                [72, 4, 95, 30, 55],
                [73, 32, 67, 91, 1],
                [92, 70, 82, 54, 62],
                [16, 61, 9, 10, 28],
                [51, 0, 53, 57, 83],
                [40, 39, 22, 87, 86],
                [20, 25, 94, 84, 5],
                [14, 24, 6, 7, 18],
                [43, 97, 42, 3, 88],
                [37, 17, 76, 12, 68],
                [49, 33, 71, 23, 60],
                [15, 21, 19, 31, 38],
                [75, 63, 66, 64, 34],
                [77, 26, 45, 99, 79],
                [11, 2, 35, 46, 98],
                [29, 93, 27, 78, 44],
                [65, 50, 74, 36, 80],
                [56, 52, 47, 59, 96],
                [8, 58, 90, 13, 48],
                [81, 69, 41, 89, 85],
            ]
            Y_copy = copy.copy(Y)
            for i in range(20):
                for j in super_label[i]:
                    Y[Y_copy == j] = i
        nmi, ari, f, acc = ccevaluation.evaluate(Y, X)
        print('CC=>NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

        metric_dict = {
            'nmi': nmi,
            'ari': ari,
            'f': f,
            'acc': acc
        }
        writer.add_scalars("Metric", metric_dict, global_step=epoch)

        #evaluation.net_evaluation(net,test_loader,args.dataset_size, args.test_batch_size)
        net, optimizer,train_stats, pseudo_labels = train_net(net, train_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels,args.batch_size,args.zeta, args.clu_begin_epoch)

        if epoch == 0:
            train_stats['loss_ins']=0
            train_stats['loss_clu']=0
            train_stats['positive_pairs_mean']=0

        real_loss_dict = {
            'loss_ins': train_stats['loss_ins'],
            'loss_clu': train_stats['loss_clu'],
            'loss_total': train_stats['loss_ins'] + train_stats['loss_clu']
        }

        writer.add_scalars("real_loss", real_loss_dict, global_step=epoch)

        writer.add_scalar("positive_pairs_mean", train_stats['positive_pairs_mean'], global_step=epoch)

        all_pseudo_index = pseudo_labels != -1
        all_pseudo_num = all_pseudo_index.sum().item()
        writer.add_scalar("conf.num", all_pseudo_num, global_step=epoch)

        if epoch == 0 or epoch == args.max_epochs or epoch == args.max_epochs/2:
            writer.add_hparams({'dataset':args.dataset, 'max_epochs':args.max_epochs,'batch_size':args.batch_size,'zata':args.zeta,'clu_begin_epoch':args.clu_begin_epoch,'alpha':args.alpha,'gamma':args.gamma},{'acc':acc})

        '''
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        with open('{}_C3_sl_loss_epoch_{}'.format(args.dataset,epoch), 'wb') as out:
            torch.save(state, out)
        '''
        #if epoch % 10 == 0:
        save_model.save_model(args, net, optimizer, epoch)

    #save_model.save_model(args, net, optimizer, args.max_epochs)

if __name__ == "__main__":
    main()
