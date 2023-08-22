import os
import yaml
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import torch.optim as optim

from classes.models import get_bnlstm_model
from classes.trainers import TripletLossTrainer
from classes.datasets import get_dataloader, get_dataloader_with_bucketing



def main():
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    login(token=config['hf_token'])
    
    model = get_bnlstm_model(config)
    model = model.to(config['device'])
    
    if config['do_train']:
        if config['padding'] == 'constant':
            print(f"Using constant padding of len {config['model_params']['max_length']}")
            train_dl = get_dataloader(config, 'train', sample=config['train_sample_size'])
        elif config['padding'] == 'collate':
            print(f"Using batch padding")
            train_dl = get_dataloader_with_bucketing(config, 'train', sample=config['train_sample_size'])
    else:
        print('do_train is False, exit training...')
        exit()
    if config['do_eval']:
        if config['padding'] == 'constant':
            val_dl = get_dataloader(config, 'test', sample=config['val_sample_size'])
        elif config['padding'] == 'collate':
            val_dl = get_dataloader_with_bucketing(config, 'test', sample=config['val_sample_size'])
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