import torch
from typing import Dict, List


class JudicialAdjudicationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list: List):
        self.data = data_list
        
    def __getitem__(self, index) -> Dict:
        return self.data[index]
    
    def __len__(self):
        return len(self.data)