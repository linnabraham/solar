#!/bin/env python
"""
Vishal's utility script
"""
import torch 
import torch.nn as nn 
from sklearn import metrics
from scipy.stats import pearsonr
from captum.attr import IntegratedGradients, GradientShap, GuidedGradCam

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device 

def correlation(pred,true):
    try:
        return pearsonr(true.detach().cpu().numpy(),pred.detach().cpu().numpy())[0].item()
    except:
        return pearsonr(true.detach().cpu().numpy(),pred.detach().cpu().numpy())[0]

def accuracy(pred,true):
    return metrics.accuracy_score(true.detach().cpu().numpy(),pred.detach().cpu().numpy())
def precision_score(pred,true,**kwargs):
    avg = kwargs.pop('average','micro')
    return metrics.precision_score(true.detach().cpu().numpy(),pred.detach().cpu().numpy(),average=avg,**kwargs)
def recall_score(pred,true,**kwargs):
    avg = kwargs.pop('average','micro')
    return metrics.recall_score(true.detach().cpu().numpy(),pred.detach().cpu().numpy(),average=avg,**kwargs)
def f1_score(pred,true,**kwargs):
    avg = kwargs.pop('average','micro')
    return metrics.f1_score(true.detach().cpu().numpy(),pred.detach().cpu().numpy(),average=avg,**kwargs)
def get_attribution_images(model, images,labels,gradcam=True):
    model.eval()
    device = next(model.parameters()).device
    images = images.clone().to(device)
    num_classes = torch.unique(model(images))

    torch.manual_seed(0)
    
    # Generate baselines
    baseline_zero = torch.zeros_like(images)
    baseline_one = torch.ones_like(images)
    
    # Compute attributions using Integrated Gradients
    ig = IntegratedGradients(model)
    attributions_ig, _ = ig.attribute(images, baseline_zero, target=labels, n_steps=100, internal_batch_size=1,
                                        return_convergence_delta=True)
    attributions_ig_baseline1, _ = ig.attribute(images, baseline_one, target=labels, n_steps=100,internal_batch_size=1,
                                        return_convergence_delta=True)
    
    # Compute attributions using GradientShap
    gs = GradientShap(model)
    attributions_gs_baseline1, _ = gs.attribute(images, baseline_one, target=labels, n_samples=50, 
                                                stdevs=0.0001,return_convergence_delta=True)
    attributions_gs_baseline0, _ = gs.attribute(images, baseline_zero, target=labels, n_samples=50, 
                                                stdevs=0.0001,return_convergence_delta=True)
    if gradcam:
        # Compute attributions using Guided Grad CAM
        last_layer = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)][-1]
        ggc = GuidedGradCam(model, last_layer)
        attributions_ggc = ggc.attribute(images, target=labels).detach().cpu().numpy()
    else: 
        attributions_ggc = None
        
    images = images.detach()
    return attributions_gs_baseline1.detach().cpu().numpy(), attributions_gs_baseline0.detach().cpu().numpy(), \
           attributions_ggc,attributions_ig.detach().cpu().numpy(),attributions_ig_baseline1.detach().cpu().numpy()

Losses={}
Losses['crossentropy'] = nn.CrossEntropyLoss()
Losses['weighted_crossentropy'] = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.3, 1.0],dtype=torch.float64))
Losses['weighted_crossentropy_binary'] = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1.0],dtype=torch.float64))
Losses['accuracy'] = accuracy
Losses['precision'] = precision_score
Losses['recall'] = recall_score
Losses['f1score'] = f1_score
Losses["MSE"] = nn.MSELoss()
Losses['MAE'] = nn.L1Loss()
Losses['correlation'] = correlation

