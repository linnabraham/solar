#!/bin/env python
"""
Vishal's base model script
"""
import torch 
import torch.nn as nn 
import numpy as np
import pytorch_lightning as pl 
import sys 

from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau,CosineAnnealingWarmRestarts, LambdaLR
from t_utils import Losses

class BaseModel(pl.LightningModule):
    """
        This is the base model, with generic initialization, training 
        and validation steps. It is model agnostic, and wraps up all of the 
        necessary (but annoying) functions!
    """
    def __init__(self,**kwargs):
        super().__init__()
        self.lr = kwargs.pop('lr',1e-4)
        self.l2reg = kwargs.pop('l2reg',1e-5)
        self.l1reg = kwargs.pop('l1reg',1e-5)
        self.schedule = kwargs.pop("schedule","LRPlateau")
        loss = kwargs.pop('loss','crossentropy')
        self.lossfun = Losses[loss]
        self.val_loss = Losses[loss]
        self.classify = kwargs.pop('classify',True)
        mlist =  kwargs.pop('metrics',[])
        if len(mlist)==0:
            if self.classify:
                mlist = ['accuracy']
            else:
                mlist = ['MSE']
        self.metrics = {m:Losses[m] for m in mlist}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = self.lr,weight_decay=self.l2reg)
        if self.schedule=="Cosine_warm":
            lr_scheduler = {"scheduler":CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=1),"monitor":"val_loss"}
        elif self.schedule=="LRPlateau":
            lr_scheduler = {"scheduler":ReduceLROnPlateau(optimizer,patience=5),"monitor":"val_loss"}
        else:
            lamb = lambda epoch: self.lr
            lr_scheduler = {"scheduler":LambdaLR(optimizer,lr_lambda=lamb),"monitor":"val_loss"}
        return [optimizer], lr_scheduler
    def get_labels(self,aia):
        logits=self(aia)
        return self.logit_to_label(logits)
    def logit_to_label(self,logits):
        if self.classify:
            return torch.argmax(logits,dim=1)
        else:
            return logits.flatten()
    def training_step(self,tr_batch,tr_idx):
        image,labels = tr_batch
        logits = self(image)
        loss = self.lossfun(logits.flatten(),labels.flatten())
        self.log("train_loss",np.float(loss.detach().cpu().numpy().item()))
        for m in self.metrics.keys():
            self.log(f"train_{m}",self.metrics[m](self.logit_to_label(logits),labels),
                        on_step=False, on_epoch=True, logger=True)
        
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return loss+l1_norm*self.l1reg

    def validation_step(self,tr_batch,tr_idx):
        image,labels = tr_batch
        logits = self(image)
        loss = self.val_loss(logits.flatten(),labels.flatten())
        self.log("val_loss",np.float(loss.detach().cpu().numpy().item()))
        for m in self.metrics.keys():
            self.log(f"val_{m}",self.metrics[m](self.logit_to_label(logits),labels),
                    on_step=False, on_epoch=True, logger=True)
        
        return loss 
    def test_step(self,tr_batch,tr_idx):
        image,labels = tr_batch
        logits = self(image)
        loss = self.lossfun(logits.flatten(),labels.flatten())
        self.log("test_loss",np.float(loss.detach().cpu().numpy().item()))
        for m in self.metrics.keys():
            self.log(f"test_{m}",self.metrics[m](self.logit_to_label(logits),labels),on_step=False, on_epoch=True, logger=True)
        return loss 

