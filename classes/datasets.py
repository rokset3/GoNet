import random
from typing import List


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from einops import rearrange
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader




def get_dataset_for_bucketing(config, split='train', sample=None):
    ds = load_dataset(config['dataset_path'], split=split)
    if sample is not None:
        ds = ds.select(range(0, sample))
    
    ds = ds.to_pandas()
    return TripletLossDatasetFromPandasDataFrame(ds, config['model_params']['max_length'])
    
def get_dataloader_with_bucketing(config, split='train', sample=None):
    ds = get_dataset_for_bucketing(config, split, sample)
    if split =='train':
        dl = DataLoader(ds,
                        batch_size=config['train_batch_size'],
                        shuffle=True,
                        collate_fn=bucket_collate_fn,
                        num_workers=config['dataloader_num_workers'])
    if split =='test':
        dl = DataLoader(ds,
                        batch_size=config['val_batch_size'],
                        collate_fn=bucket_collate_fn,
                        num_workers=config['dataloader_num_workers'])
    return dl
    

def get_dataset(config, split='train', sample=None):
    ds = load_dataset(config['dataset_path'], split=split)
    if sample is not None:
        ds = ds.select(range(0, sample))
    
    ds = ds.to_pandas()
    return CustomTripletLossDatasetFromPandasDataFrame(ds, config['model_params']['max_length'])

def get_dataloader(config, split='train', sample=None):
    ds = get_dataset(config, split, sample)
    if split =='train':
        dl = DataLoader(ds,
                        batch_size=config['train_batch_size'],
                        shuffle=True,
                        num_workers=config['dataloader_num_workers'])
    if split =='test':
        dl = DataLoader(ds,
                        batch_size=config['val_batch_size'],
                        num_workers=config['dataloader_num_workers'])
    return dl



class CustomTripletLossDatasetFromPandasDataFrame(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 max_length: int=128,
                 label_col_name: str='participant_id',
                 features_col_names: List[str]=['keycode_ids', 'hl', 'il', 'pl', 'rl']):
        assert len(features_col_names) == 5, f"currently working with 5 features, got {len(features_col_names)}"
        self.df = df
        self.max_length = max_length
        self.label_col_name = label_col_name
        self.features_col_names = features_col_names
        
        #for Triplet Loss 
        self.index = df.index.values
        self.labels = df[label_col_name].values
    
    
    def __len__(self, ):
        return len(self.df)
    
    
    def _pad_sequences(self, sequence):
        padded_sequence = np.zeros(self.max_length)
        padded_sequence[:len(sequence)] = sequence
        return padded_sequence
    
    
    def _getitem(self, idx):
        label = torch.tensor(self.df.iloc[idx][self.label_col_name])
        keycode_ids = self.df.iloc[idx][self.features_col_names[0]].astype(np.float32)
        hl = self.df.iloc[idx][self.features_col_names[1]].astype(np.float32)
        il = self.df.iloc[idx][self.features_col_names[2]].astype(np.float32)
        pl = self.df.iloc[idx][self.features_col_names[3]].astype(np.float32)
        rl = self.df.iloc[idx][self.features_col_names[4]].astype(np.float32)
        
        if keycode_ids.shape[0] > self.max_length:
            keycode_ids = keycode_ids[:self.max_length] 
            hl = hl[:self.max_length]
            il = il[:self.max_length]
            pl = pl[:self.max_length]
            rl = rl[:self.max_length]
        else:
            keycode_ids = self._pad_sequences(keycode_ids)
            hl = self._pad_sequences(hl)
            il = self._pad_sequences(il)
            pl = self._pad_sequences(pl)
            rl = self._pad_sequences(rl)
        
        features = torch.FloatTensor(
                np.array([
                    keycode_ids,
                    hl,
                    il,
                    pl,
                    rl]
                ))
        return rearrange(features, 's h -> h s')
    
    
    def __getitem__(self, curr_idx):
        anchor_sample = self._getitem(curr_idx)
        anchor_label = self.labels[curr_idx]
        
        positive_idxs = self.index[self.index!=curr_idx][self.labels[self.index!=curr_idx]==anchor_label]
        positive_idx = random.choice(positive_idxs)
        positive_sample = self._getitem(positive_idx)
        
        negative_idxs = self.index[self.index!=curr_idx][self.labels[self.index!=curr_idx]!=anchor_label]
        negative_idx = random.choice(positive_idxs)
        negative_sample = self._getitem(negative_idx)
        
        return dict(
            anchor_label = anchor_label,
            anchor_features = anchor_sample,
            positive_features = positive_sample,
            negative_features = negative_sample,
        )

    
class TripletLossDatasetFromPandasDataFrame(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 max_length: int=128,
                 label_col_name: str='participant_id',
                 features_col_names: List[str]=['keycode_ids', 'hl', 'il', 'pl', 'rl'],
                 random_state: int=42):
        assert len(features_col_names) == 5, f"currently working with 5 features, got {len(features_col_names)}"
        
        self.max_length = max_length
        self.label_col_name = label_col_name
        self.features_col_names = features_col_names
        self.random_state = random_state            
        self.df = df
        #for Triplet Loss 
        self.index = df.index.values
        self.labels = df[label_col_name].values
        
    def __len__(self):
        return len(self.df)
        
    def _getitem(self, idx):
        label = torch.tensor(self.df.iloc[idx][self.label_col_name])
        features = torch.tensor(np.stack(self.df.iloc[idx][self.features_col_names], axis=0), dtype=torch.float32)
        if features.shape[1] > self.max_length:
            features = features[:, :self.max_length]
        
        features_len = features.shape[1]
        return features, features_len
    
    def __getitem__(self, curr_idx):
        anchor_features, anchor_len = self._getitem(curr_idx)
        
        
        positive_idxs = list(range(curr_idx//15 * 15, (curr_idx//15 + 1) * 15))
        positive_idxs.remove(curr_idx)
        positive_idx = random.choice(positive_idxs)
        positive_features, positive_len = self._getitem(positive_idx)
        
        
        positive_idxs_set = set(positive_idxs)
        positive_idxs_set.add(curr_idx)
        sample_for_negatives_indexes = (set(self.df.sample(4000, random_state=self.random_state).index.values) - positive_idxs_set)
        negative_idx = random.choice(tuple(sample_for_negatives_indexes))
        negative_features, negative_len = self._getitem(negative_idx)
        
        return (anchor_features,
                positive_features,
                negative_features,
                anchor_len,
                positive_len,
                negative_len)
        
        
        
def bucket_collate_fn(data, n_features=5):
    def _reshape_features(data, pos, features, max_len):
        for i in range(len(data)):
            j, k = data[i][pos].size(0), data[i][pos].size(1)
            features[i] = torch.cat([data[i][pos], torch.zeros((j, max_len - k))], dim=-1)

        return features
    '''
    data = List[dict]
    '''
    (_,
     _,
     _,
     anchor_len,
     positive_len,
     negative_len) = zip(*data)
    
    anchor_max_len = max(anchor_len)
    positive_max_len = max(positive_len)
    negative_max_len = max(negative_len)
    
    n_features = n_features
    
    anchor_features = torch.zeros((len(data), n_features, anchor_max_len))
    positive_features = torch.zeros((len(data), n_features, positive_max_len))
    negative_features = torch.zeros((len(data),  n_features, negative_max_len))
    
    anchor_pos, positive_pos, negative_pos = 0, 1, 2
    #for i in range(len(data)):
    #    j, k = data[i][anchor_pos].size(0), data[i][anchor_pos].size(1)
    #    anchor_features[i] = torch.cat([data[i][achor_pos], torch.zeros((anchor_max_len - j, k))])
     
    anchor_features = _reshape_features(data, anchor_pos, anchor_features, anchor_max_len)
    positive_features = _reshape_features(data, positive_pos, positive_features, positive_max_len)
    negative_features = _reshape_features(data, negative_pos, negative_features, negative_max_len)
    
    
    anchor_len = torch.tensor(anchor_len)
    positive_len = torch.tensor(positive_len)
    negative_len = torch.tensor(negative_len)
    
    return (
        anchor_features.float(),
        positive_features.float(),
        negative_features.float(),
        anchor_len.long(),
        positive_len.long(),
        negative_len.long(),
    )


    
    
    
    