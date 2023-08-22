import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self,
                 margin: float=1.0):
        super().__init__()
        self.margin = margin
        
    def _calculate_euclidean_distance(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
        
    
    def forward(self,
               anchor: torch.Tensor,
               positive: torch.Tensor,
               negative: torch.Tensor) -> torch.Tensor:
        
        positive_distance = self._calculate_euclidean_distance(anchor, positive)
        negative_distance = self._calculate_euclidean_distance(anchor, negative)        
        return F.relu(positive_distance - negative_distance + self.margin).mean()