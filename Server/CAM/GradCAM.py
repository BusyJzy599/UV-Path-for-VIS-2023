from __future__ import absolute_import
import torch
# import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
import torch.nn as nn
from datasets import HDF5Dataset
from PIL import Image
import models
import os.path as osp
import copy
import h5py
import pandas as pd
import pickle


# resnet = models.create('resnet50', num_features=2048, num_classes=2).cuda()


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        for name, module in self.model._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
                break
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers,use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda
    def get_gradients(self):
        return self.feature_extractor.gradients
    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = self.model.gap(output)
        output = output.view(output.size(0), -1)
        if self.cuda:
            # output = output.cpu()
            # output = resnet.gap(output)
            # output = resnet.feat_bn(output)
            output = self.model.classifier(output).cuda()

        else:
            # output = resnet.gap(output)
            output = self.model.classifier(output)

        return target_activations, output


class GradCam:
   def __init__(self, model, target_layer_names, img_size,use_cuda=True):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.img_size=img_size
        if self.cuda:
           self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

   def forward(self, input):
        return self.model(input)

   def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
           index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        if self.cuda:
           one_hot = torch.sum(one_hot.cuda() * output)
        else:
           one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()

        one_hot.backward(retain_graph=True) ## 这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    # preprocessed_img = np.ascontiguousarray(preprocessed_img)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img
    input.requires_grad = True
    return input

def show_cam_on_image(img, mask, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # cam = np.transpose(np.float32(heatmap), (2, 0, 1)) + img
    cam = img + heatmap
    cam = cam / np.max(cam)
    # print(cam.shape)
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam)
    cam.save(save_path)

