import os
import yaml
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import torch.optim as optim

from classes.models import get_bnlstm_model
from classes.datasets import get_dataloader
from classes.trainers import TripletLossTrainer



def main():
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    login(token=config['hf_token'])
    
    model = get_bnlstm_model(config)
    model = model.to(config['device'])
    
    if config['do_train']:
        train_dl = get_dataloader(config, 'train', sample=config['train_sample_size'])
    else:
        print('do_train is False, exit training...')
        exit()
    if config['do_eval']:
        val_dl = get_dataloader(config, 'test', sample=config['val_sample_size'])
    else:
        val_dl = None
    
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    trainer = TripletLossTrainer(model,
                                 optimizer,
                                 train_dl,
                                 val_dl,
                                 config)
    trainer.train()


    
if __name__ == '__main__':
    main()