import os
import yaml
import math

import numpy as np
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn 
from einops import rearrange
from transformers import AdamW
from torch.cuda.amp import autocast, GradScaler

from classes.losses import TripletLoss 


class Util:
    scaler = GradScaler()
    
    
class TripletLossTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 train_dataloader,
                 valid_dataloader,
                 config,
                 scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = config['device']
        self.config = config
        self.train_steps_in_epoch = len(train_dataloader)
        
        self.is_wandb = False
        self.is_tensorboard = False
        self.is_plotting = False
        
        if self.config['log_to'] == 'wandb':
            self.is_wandb = True
            self._init_wandb()
            
        elif self.config['log_to'] == 'tensorboard':
            self.is_tensorboard = True
        elif self.config['log_to'] == 'plot':
            self.is_plotting = True
        
        
    def _init_wandb(self):
        wandb.init(project=self.config['project_name'],
                   entity=self.config['entity'],
                   reinit=True,
                   config=self.config,
                   name=self.config['experiment_name'])
        
    def _log_losses(self,
                    step: int,
                    loss: float,
                    mode: str='train',): #or 'eval'
        if self.is_wandb:
            wandb.log({f'{mode}_loss': loss}, step=step)
        
        
    def yield_loss(self, anchor, positive, negative):
        return TripletLoss(margin=self.config['loss_params']['margin'])(anchor, positive, negative)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        epoch_steps = epoch * self.train_steps_in_epoch
        
        for step, (anchor_features,
                   positive_features,
                   negative_features,
                   anchor_len,
                   positive_len,
                   negative_len) in enumerate(tqdm(self.train_dataloader, desc="Training", leave=False)):
            global_step = epoch_steps + step
            
            anchor_features = rearrange(anchor_features, 'b h s -> b s h').to(self.device)
            positive_features = rearrange(positive_features, 'b h s -> b s h').to(self.device)
            negative_features = rearrange(negative_features, 'b h s -> b s h').to(self.device)
            
            with autocast():
                anchor_out = self.model(anchor_features)[0][-1]
                positive_out = self.model(positive_features)[0][-1]
                negative_out = self.model(negative_features)[0][-1]
            loss = self.loss_fn(anchor_out, positive_out, negative_out)
            train_loss += loss.item()
            
            self._log_losses(global_step, loss, 'train')
            
            Util.scaler.scale(loss).backward()
            Util.scaler.step(self.optimizer)
            Util.scaler.update()
            
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            
        res_loss = train_loss / len(self.train_dataloader)
        return res_loss
    
    def valid_one_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        epoch_steps = epoch * self.val_steps_in_epoch
        with torch.no_grad():
            for step, (anchor_features,
                       positive_features,
                       negative_sample,
                       anchor_len,
                       positive_len,
                       negative_len) in enumerate(tqdm(self.valid_dataloader, desc='Evaluation', leave=False)):
                
                anchor_features = rearrange(anchor_features, 'b h s -> b s h').to(self.device)
                positive_features = rearrange(positive_features, 'b h s -> b s h').to(self.device)
                negative_features = rearrange(negative_features, 'b h s -> b s h').to(self.device)
                
                anchor_out = self.model(anchor_features)[0][:-1]
                positive_out = self.model(positive_features)[0][:-1]
                negative_out = self.model(negative_features)[0][:-1]
                
                loss = self.loss_fn(anchor_out, positive_out, negative_out)
                test_loss += loss.item()
            
            global_step = (epoch + 1) * self.train_steps_in_epoch        
            res_loss = test_loss/len(self.valid_dataloader)
            self._log_losses(global_step, res_loss, 'val')
            return res_loss
        
    def get_model(self):
        return self.model
    
    def train(self):
        experiment_dir = os.path.join(self.config['save_dir'], self.config['experiment_name'])
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            
        if not os.path.exists(self.config['save_dir']):
            os.makedirs(self.config['save_dir'])
            
        best_loss = 100
        losses_train = []
        losses_val = []
        for epoch in tqdm(range(0, self.config['epochs']), desc='Epochs'):
            loss_train = self.train_one_epoch(epoch)
            losses_train.append(loss_train)
            if self.config['save_every_k_epochs'] is not None:
                if epoch % self.config['save_every_k_epochs'] == 0:
                    torch.save(self.get_model().state_dict(), os.path.join(experiment_dir, f'model-{epoch}-epoch.pt'))
            
            if self.valid_dataloader is not None:
                loss_val = self.valid_one_epoch(epoch)
                losses_val.append(losses_val)
                
                if loss_val < best_loss:                    
                    torch.save(self.get_model().state_dict(), os.path.join(experiment_dir, 'best_model.pt'))
                    best_loss = loss_val
            
            if self.config['verbose_loss_on_epoch']:
                print(f'Epoch:{epoch}, train loss: {loss_train}')
                if self.valid_dataloader is not None:
                    print(f'Epoch:{epoch}, val loss: {loss_val}')
                    
            
                
        
        torch.save(self.get_model().state_dict(), os.path.join(experiment_dir, 'last_model.pt'))
        
        params_dir = os.path.join(experiment_dir,  'params.yaml')
        with open(params_dir, 'w') as f:
            documents = yaml.dump(self.config, f)
            
                
        