import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
from torch.autograd import Variable
import random
import os
from torch.nn.utils.weight_norm import WeightNorm
import sys
from collections import OrderedDict
import cv2

sys.path.append('../../')
import third_party
import timm
from functools import partial


def l2_norm(input, axis=1):
  norm = torch.norm(input, 2, axis, True)
  output = torch.div(input, norm)
  return output

class Spp_patch2(nn.Module): 
  def __init__(self, m:nn.Module,patch,pixel):
    super(Spp_patch2, self).__init__()
    self.proj = None
    # self.patch = nn.Parameter(torch.randn(1))
    self.patch = patch
    self.pixel = pixel   # nn.Parameter(torch.randn(1))  #
    for name,module in m.named_children():
      if name == 'proj':
        self.proj = module   
        print('patch_embed proj replace successfully')

  
    self.layer2 = nn.Identity()
    self.pool_zlf_1 = nn.AvgPool2d(kernel_size=(1,768),stride=(1,768),padding=0)  
    self.pool_zlf_2 = nn.AvgPool2d(kernel_size=(196,1),stride=(196,1),padding=0)  
    
    self.SE_zlf_1 = nn.Sequential(OrderedDict([
                      ('SE1_fc1',  nn.Linear(196,16)),
                      ('SE1_ReLU',nn.ReLU()),
                      ('SE1_fc2',nn.Linear(16,196)),
                      ('SE1_Sigmoid',nn.Sigmoid())
                          ]))
    self.SE_zlf_2 = nn.Sequential(OrderedDict([
                      ('SE2_fc1',  nn.Linear(768,16)),
                      ('SE2_ReLU',nn.ReLU()),
                      ('SE2_fc2',nn.Linear(16,768)),
                      ('SE2_Sigmoid',nn.Sigmoid())
                          ])) 
    pass
  def forward(self,input):
    if self.proj == None:
      print('fas.py code have a problem in Spp_patch')
      exit()
    out_proj = self.proj(input)                    
    out_proj_resize = out_proj.flatten(2).transpose(1, 2)  
    out_layer2 = self.layer2(out_proj_resize)   
    out_layer2_reshape = out_layer2.unsqueeze(1) 
    

    # # patch
    out_pool_1 = self.pool_zlf_1(out_layer2_reshape)      
    out_pool_11 = out_pool_1.squeeze(1).squeeze(2)    
    out_SE1 = self.SE_zlf_1(out_pool_11)
    out_SE11 = out_SE1.unsqueeze(-1)  
    out_SE12 = out_SE11*self.patch
    
    # # pixel
    out_pool_2 = self.pool_zlf_2(out_layer2_reshape)   
    out_pool_21 = out_pool_2.squeeze(1).squeeze(1)
    out_SE2 = self.SE_zlf_2(out_pool_21)
    out_SE21 = out_SE2.unsqueeze(1) 
    out_SE22 = out_SE21*self.pixel

    out = out_layer2 + out_layer2*out_SE12 + out_layer2*out_SE22
    return out


class feature_generator_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(feature_generator_adapt, self).__init__()
    self.vit = third_party.create_model(
        'vit_base_patch16_224', pretrained=True, gamma=gamma, beta=beta)

  def forward(self, input):
    feat, total_loss = self.vit.forward_features(input)
    return feat, total_loss


class feature_generator_fix(nn.Module):

  def __init__(self):
    super(feature_generator_fix, self).__init__()
    self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
  
  def forward(self, input):
    feat = self.vit.forward_features(input).detach()
    return feat


class feature_embedder(nn.Module): 

  def __init__(self):
    super(feature_embedder, self).__init__()
    self.bottleneck_layer_fc = nn.Linear(768, 512)
    self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
    self.bottleneck_layer_fc.bias.data.fill_(0.1)
    self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                          nn.Dropout(0.5))

  def forward(self, input, norm_flag=True):
    feature = self.bottleneck_layer(input)
    if (norm_flag): 
      feature_norm = torch.norm(
          feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
      feature = torch.div(feature, feature_norm)
    return feature


class classifier(nn.Module):

  def __init__(self):
    super(classifier, self).__init__()
    self.classifier_layer = nn.Linear(512, 2)           
    self.classifier_layer.weight.data.normal_(0, 0.01)  
    self.classifier_layer.bias.data.fill_(0.0)          

  def forward(self, input, norm_flag=True):
    if (norm_flag):  
      self.classifier_layer.weight.data = l2_norm(      
          self.classifier_layer.weight, axis=0)
      classifier_out = self.classifier_layer(input)
    else:
      classifier_out = self.classifier_layer(input)
    return classifier_out

class fas_model_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(fas_model_adapt, self).__init__()
    self.backbone = feature_generator_adapt(gamma, beta)
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature, total_loss = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature, total_loss


class fas_model_fix(nn.Module):

  def __init__(self):
    super(fas_model_fix, self).__init__()
    self.backbone = feature_generator_fix()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature


class fas_model_weighting2(nn.Module):
  def __init__(self, gamma, beta,tgt_data,patch,pixel):
    super(fas_model_weighting2, self).__init__()
    self.model = fas_model_adapt(gamma,beta)
    net_ = torch.load(tgt_data + "_baseline.pth.tar")
    self.model.load_state_dict(net_["state_dict"])
    cout = 0
    for name, module in self.model.backbone.vit.named_children():
      if name == 'patch_embed' and cout == 0:
        replaced_module = Spp_patch2(module,patch,pixel) 
        self.model.backbone.vit.add_module(name,replaced_module)
        cout = 1

  def forward(self, input, norm_flag=True):
    feature768, total_loss = self.model.backbone(input)
    feature512 = self.model.embedder(feature768, norm_flag)    
    classifier_out = self.model.classifier(feature512, norm_flag)
    return classifier_out, feature512, total_loss
