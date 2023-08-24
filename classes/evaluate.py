import torch
from einops import rearrange


class Inference:
    def __init__(self,
                 model,
                 dataloader,
                 config,
                 device,
                 ):
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
                 model: Inference,
                 test_ds,
                 evaluation_config):
        
        
    
    
                
                    
                