#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.lenet import LeNet
from models.mlp import MLP
from models.bcmlenet8 import BCMLeNet8
# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

torch.manual_seed(args.seed)

def main():
    # load dataset and split users
    dataset_train = datasets.MNIST('temp_nn/data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    img_size = dataset_train[0][0].shape
    dataset_test = datasets.MNIST('temp_nn/data/mnist/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    # build model
    if args.model == 'lenet':
        net_glob = LeNet().to(args.device)
    elif args.model == 'bcmlent8':
        net_glob = BCMLeNet8().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64).to(args.device)
    else:
        print("error")
        exit('Error: unrecognized model')
    #for name,weight in net_glob.named_parameters():
    #    print(name)
    #    print(weight)
    #print(net_glob)
    #exit()
    if args.resume:
        model_path = os.path.join(args.save_dir, args.resume)
        print("Path is:{}".format(model_path))
        try:
            net_glob.load_state_dict(torch.load(model_path))
        except:
            try:
                net_glob.load_state_dict(torch.load(model_path)["net"])
            except:
                print("Can't load model!")
                return
    if args.evaluate:
        test(net_glob, test_loader)
        return

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # masks = {}
            # for name, W in (model.named_parameters()):
            #     weight = W.cpu().detach().numpy()
            #     non_zeros = weight != 0
            #     non_zeros = non_zeros.astype(np.float32)
            #     zero_mask = torch.from_numpy(non_zeros).cuda()
            #     W = torch.from_numpy(weight).cuda()
            #     W.data = W
            #     masks[name] = zero_mask
            # for name, W in (model.named_parameters()):
            #     W.grad *= masks[name]
            #

            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    #plt.figure()
    #plt.plot(range(len(list_loss)), list_loss)
    #plt.xlabel('epochs')
    #plt.ylabel('train loss')
    #plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
    torch.save(net_glob.state_dict(), "trained/lenet_mnist.pt")



def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    main()

