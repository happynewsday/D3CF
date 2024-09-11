import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict

from modules import transform, resnet, network
from utils import yaml_config_hook, save_model
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation
from train import train_net
from modules.contrastive_loss import ClusterLossBoost
from modules.misc import NativeScalerWithGradNormCount as NativeScaler
import modules.misc as misc
from modules.data import build_dataset, CIFAR10,CIFAR100,ImageFolder,STL10

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
    # prepare data---------------------------------------------------------------------------------------------------------------------------------------------------
    #train data
    '''
    train_dataset_original = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

    train_dataset =  CIFAR10(
                    root=args.dataset_dir,
                    train=True,
                    download=True,
                    transform=transform.Transforms(size=args.image_size, s=0.5),
                )

    test_dataset_original = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

    test_dataset = CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )

    dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,pin_memory=True)


    # test data

    test_dataset_original1 = torchvision.datasets.CIFAR10(
        root=args.dataset_dir,
        download=True,
        train=True,
        transform=transform.Transforms(size=args.image_size).test_transform,
    )

    test_dataset_1 = CIFAR10(
        root=args.dataset_dir,
        download=True,
        train=True,
        transform=transform.Transforms(size=args.image_size).test_transform,
    )
    
    test_dataset_original2 = torchvision.datasets.CIFAR10(
        root=args.dataset_dir,
        download=True,
        train=False,
        transform=transform.Transforms(size=args.image_size).test_transform,
    )

    test_dataset_2 = CIFAR10(
        root=args.dataset_dir,
        download=True,
        train=False,
        transform=transform.Transforms(size=args.image_size).test_transform,
    )
    dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False)
    '''
    #=======================================================================
    # prepare data start==========================================================
    if args.dataset == "CIFAR-10":
        '''
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        '''

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
        '''
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
        '''

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
        '''
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
        '''

        args.class_num = 10#args.dataset_dir + "ImageNet-10"
        train_dataset = ImageFolder(root=args.dataset_dir + "/imagenet-10", transform=transform.Transforms(size=args.image_size, blur=True))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "/imagenet-10", transform=transform.Transforms(size=args.image_size).test_transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)
    elif args.dataset == "ImageNet-dogs":
        '''
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
        '''

        args.class_num = 15
        train_dataset = ImageFolder(root=args.dataset_dir + "/imagenet-dogs", transform=transform.Transforms(size=args.image_size, blur=True))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "/imagenet-dogs", transform=transform.Transforms(size=args.image_size).test_transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False)

        args.dataset_size = len(test_dataset)
    elif args.dataset == "tiny-ImageNet":
        '''
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
        '''

        args.class_num = 200
        train_dataset = ImageFolder(root=args.dataset_dir + "/tiny-imagenet-200/train", transform=transform.Transforms(s=0.5, size=args.image_size))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)

        test_dataset = ImageFolder(root=args.dataset_dir + "/tiny-imagenet-200/train", transform=transform.Transforms(size=args.image_size).test_transform)
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

        args.dataset_size = len(test_dataset)
    else:
        raise NotImplementedError

    '''
    #this is from CC Model
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    '''

    # prepare data end==========================================================
    '''
    #this is from Twin CC
    dataset_train = build_dataset(type="train", args=args)
    dataset_pseudo = build_dataset(type="pseudo", args=args)
    dataset_val = build_dataset(type="val", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_pseudo = torch.utils.data.DistributedSampler(
        dataset_pseudo, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_ps = torch.utils.data.DataLoader(
        dataset_pseudo,
        sampler=sampler_pseudo,
        batch_size=1000,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    '''

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
        net, optimizer,train_stats, pseudo_labels = train_net(net, train_loader, optimizer,criterion_clu,clu_temp, device,epoch,loss_scaler,pseudo_labels,args.batch_size,args.zeta)

        '''
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        with open('{}_C3_SL_checkpoint_{}'.format(args.model_path,epoch), 'wb') as out:
            torch.save(state, out)
        '''

        save_model.save_model(args, net, optimizer, epoch)


if __name__ == "__main__":
    main()
