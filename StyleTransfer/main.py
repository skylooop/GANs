from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

dtype = torch.cuda if use_cuda else torch.FloatTensor
imsize = 512 if use_cuda else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])


def imshow(tensor):
    image = tensor.clone().cpu().squeeze(0)
    image = unloader(image)
    plt.imshow(image)


def image_loader(image):
    image = loader(Image.open(image)).unsqueeze(0)
    return image


content_image = image_loader("/Users/m.bobrin/Desktop/trash_pycharm/StyleTransfer/images/dancing.jpg")
style_image = image_loader("/Users/m.bobrin/Desktop/trash_pycharm/StyleTransfer/images/picasso.jpg")
unloader = transforms.ToPILImage()


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, x):
        self.loss = self.criterion(x * self.weight, self.target)
        self.out = x
        return self.out

    def backward(self, retain_vars=True):
        self.loss.backward(retain_vars=retain_vars)
        return self.loss


class Gramm(nn.Module):
    def forward(self, x):
        a, b, c, d = x.size()
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = Gramm()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        self.out = x.clone()
        self.G = self.gram(x)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)

    def backward(self, retain_variables):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


cnn = models.vgg19(pretrained=True).features
if use_cuda:
    cnn = cnn.cuda()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1, content_layers=content_layers_default,
               style_layers=style_layers_default):

    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    gram = Gramm()
    if use_cuda:
        cnn = model.cuda()
        gram = gram.cuda()
    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                print(model)
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1
            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses

input_img = content_image

def get_input_param_optimizer(input_img):
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_losses(cnn,
                                                                     style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data

output = run_style_transfer(cnn, content_image, style_image, input_img)

plt.figure()
imshow(output)

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()