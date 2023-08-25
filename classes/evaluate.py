import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from einops import rearrange
from pandarallel import pandarallel

from classes.datasets import AuthentificationDatasetFromPandasDataFrame, bucket_collate_fn_for_authentification

class Inference:
    def __init__(self,
                 model,
                 dataloader,
                 config,
                 device,
                 ):
        model.eval()
        self.model = model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device
        
    def predict_single_example(self,
                features: torch.Tensor):
        assert features.dim == 2, f'expected 2-D tensor, got {features.dim}-D'
        assert features.shape[0] == 5, f'expected input with 5 features, got {features.shape[0]} features'
        
        features = feautres.unsqueeze(0)
        with torch.no_grad():
            with torch.inference_mode():
                features = rearrange(features, 'b h s -> b s h').to(self.device)
                output = self.model()[0][-1]
                return output
            
    def predict_on_batch(self,
                         features: torch.Tensor):
        assert features.dim == 3, f'expected 3-D tensor, got {features.dim}-D'
        assert features.shape[1] ==5 , f'expected input with 5 features, got {features.shape[1]} features'
        
        with torch.no_grad():
            with torch.inference_mode():
                features = rearrange(features, 'b h s -> b s h').to(self.device)
                output = self.model()[0][-1]
                return output
    
    
    
class AuthentificationEvaluator:
    def __init__(self,
                # model: Inference,
                 dataset: AuthentificationDatasetFromPandasDataFrame,
                 config):
#        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.samples_df = pd.DataFrame()
        self._collect_samples()
        
        
    def _sample_funct(self, label):
        gallery, genuine, impostor = self.dataset._sample_authentification_ids(
                label,
                num_gallery_samples=self.config['authentification']['num_gallery_samples'],
                num_impostor_samples=self.config['authentification']['num_impostor_samples']
            )
        return dict(
                gallery_ids = gallery,
                genuine_ids = genuine,
                impostor_ids = impostor
            )
        
    
    
    def _collect_samples(self):
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True)
        self.samples_df['participant_ids'] = np.unique(self.dataset.labels) 
        self.samples_df['sampled_ids'] = self.samples_df['participant_ids'].parallel_apply(self._sample_funct)
        
    def get_dataset_per_participant(self, participant_id):
        curr_sample = self.samples_df[self.samples_df['participant_ids'] == participant_id]
        idxs = np.concatenate([curr_sample['samples_ids']['gallery_ids'],
                               curr_sample['samples_ids']['genuine_ids'],
                               curr_sample['samples_ids']['impostor_ids']], axis=0)
        gal = self.config['authentification']['num_gallery_samples']
        gen = self.config['authentification']['num_genuine_samples']
        imp = self.config['authentification']['num_impostor_samples']
        labels = np.zeros(idxs.shape)
        labels[: gal] = 0.0                 # gallery labels
        labels[gal: gal + gen] = 1.0        # genuine labels
        labels[gal + gen: ] = -1.0          # impostor labels
        
        df = self.dataset.df.iloc[idxs]
        df['labels'] = labels
        
        ds = AuthentificationDatasetFromPandasDataFrame(df)
        return ds
        
    
        
        
        
        
        
        
    
        
    
        
    
        
        
        
        
        
    
    
                
                    
                