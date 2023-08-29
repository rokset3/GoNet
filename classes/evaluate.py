import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from einops import rearrange
from pandarallel import pandarallel
from classes.datasets import AuthentificationDatasetFromPandasDataFrame, bucket_collate_fn_for_authentification, GoNetDatasetFromPandasDataFrame, get_dataloader_general

class Predictor:
    def __init__(self,
                 model,
                 config,
                 device,
                 ):
        model.eval()
        self.model = model.to(device)
        self.config = config
        self.device = device
        
    def predict_single_example(self,
                features: torch.Tensor):
        assert features.dim() == 2, f'expected 2-D tensor, got {features.dim()}-D'
        assert features.shape[0] == 5, f'expected input with 5 features, got {features.shape[0]} features'
        
        features = feautres.unsqueeze(0)
        with torch.no_grad():
            with torch.inference_mode():
                features = rearrange(features, 'b h s -> b s h').to(self.device)
                output = self.model(features)[0][-1]
                return output
            
    def predict_on_batch(self,
                         features: torch.Tensor):
        assert features.dim() == 3, f'expected 3-D tensor, got {features.dim}-D'
        assert features.shape[1] ==5 , f'expected input with 5 features, got {features.shape[1]} features'
        
        with torch.no_grad():
            with torch.inference_mode():
                features = rearrange(features, 'b h s -> b s h').to(self.device)
                output = self.model(features)[0][-1]
                return output
    
    
    
class AuthentificationEvaluator:
    def __init__(self,
                 model: torch.nn.Module,
                 dataset: AuthentificationDatasetFromPandasDataFrame,
                 config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = config['authentification']['device']
        self.samples_df = pd.DataFrame()
        self._collect_samples()
        self.predictor = Predictor(model, config, config['authentification']['device'])
        
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
        self.samples_df['participant_ids'] = np.unique(self.dataset.labels) 
        if self.config['authentification']['parallel_sampling']:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True) 
            self.samples_df['sampled_ids'] = self.samples_df['participant_ids'].parallel_apply(self._sample_funct)
        else:
            from tqdm import tqdm
            tqdm.pandas()
            self.samples_df['sampled_ids'] = self.samples_df['participant_ids'].progress_apply(self._sample_funct)
            
    def get_dataset_per_participant(self, participant_id):
        curr_sample = self.samples_df[self.samples_df['participant_ids'] == participant_id].iloc[0]
        idxs = np.concatenate([curr_sample['sampled_ids']['gallery_ids'],
                               curr_sample['sampled_ids']['genuine_ids'],
                               curr_sample['sampled_ids']['impostor_ids']], axis=0)
        gal = self.config['authentification']['num_gallery_samples']
        gen = self.config['authentification']['num_genuine_samples']
        imp = self.config['authentification']['num_impostor_samples']
        labels = np.zeros(idxs.shape)
        labels[: gal] = 100.0                 # gallery labels
        labels[gal: gal + gen] = 1.0        # genuine labels
        labels[gal + gen: ] = 0.0          # impostor labels
        
        df = self.dataset.df.iloc[idxs].copy()
        df['label'] = labels
        
        ds = GoNetDatasetFromPandasDataFrame(df)
        return ds
    
    def _calculate_euclidean_distance_between_samples(self, x1, x2):
        return (x1[:, None, :] - x2[None, :, :]).pow(2).sum(dim=2)

    def _get_dataloader(self, ds, collate_fn=bucket_collate_fn_for_authentification):
        return get_dataloader_general(ds, self.config, collate_fn=collate_fn)
    
    def _inference_on_one_participant_id(self, participant_id):
        ds = self.get_dataset_per_participant(participant_id)
        dl = self._get_dataloader(ds)
        for idx, (authentification_label, labels, features, seq_len) in enumerate(dl):
            embeddings = self.predictor.predict_on_batch(features).detach().cpu()
            if idx == 0:
                embeddings_total = embeddings.clone().detach().cpu()
                a_lbl = authentification_label.clone()
                lbl = labels.clone()
            else:
                embeddings_total = torch.cat([embeddings_total, embeddings], dim=0)
                a_lbl = torch.cat([a_lbl, authentification_label], dim=0)
                lbl = torch.cat([lbl, labels], dim=0)
        
        return dict(
            embeddings = embeddings_total,
            authentification_classes = a_lbl,
            participant_labels = lbl
        )
    
    def _get_scores_on_one_participant_id(self, col):
        participant_id = col['participant_ids']
        ds = self.get_dataset_per_participant(participant_id)
        dl = self._get_dataloader(ds)
        for idx, (authentification_label, labels, features, seq_len) in enumerate(dl):
            embeddings = self.predictor.predict_on_batch(features).detach().cpu()
            if idx == 0:
                embeddings_total = embeddings.clone().detach().cpu()
                a_lbl = authentification_label.clone()
                lbl = labels.clone()
            else:
                embeddings_total = torch.cat([embeddings_total, embeddings], dim=0)
                a_lbl = torch.cat([a_lbl, authentification_label], dim=0)
                lbl = torch.cat([lbl, labels], dim=0)
        
        gallery_samples = embeddings_total[a_lbl == torch.tensor(100.0)].clone()
        other_samples = embeddings_total[a_lbl!=torch.tensor(100.0)].clone()
        labels = a_lbl[a_lbl != torch.tensor(100.0)].clone()
        
        distances = self._calculate_euclidean_distance_between_samples(gallery_samples, other_samples).mean(dim=0)
        return (distances,
                labels)
        
    def get_scores(self,):
        from tqdm import tqdm
        tqdm.pandas()
        self.samples_df['distances'], self.samples_df['labels'] = zip(*self.samples_df.progress_apply(self._get_scores_on_one_participant_id, axis=1))
    
    def get_scores_labels(self):
        scores = np.stack(self.samples_df['distances']).flatten()
        labels = np.stack(self.samples_df['labels']).flatten()
        return scores, labels
    
    def get_results(self):
        return self.samples_df
        
    
    
            
        
        
        
        
    
        
        
        
        
        
        
    
        
    
        
    
        
        
        
        
        
    
    
                
                    
                