import os
import torch
from torch.utils.data import DataLoader

class JudicialAdjudicationEvaluator(object):
    
    def __init__(self, args, model, evaluate_loader: DataLoader):
        if args.device_id is not None:
            self.device_id = args.device_id
        else:
            self.device_id = "cpu"
        self.evalute_dir = args.evaluate_dir
        self.model = model.eval()
        self.evaluate_loader = evaluate_loader
        self.pred_crime, self.gold_crime = [], []
        self.pred_law, self.gold_law = [], []
        self.pred_period, self.gold_period = [], []
        self.evaluate()
        
    def evaluate(self):
        for i, data in enumerate(self.evaluate_loader):
            data = data.to(device=self.device_id)
            with torch.no_grad():
                pred = self.model(data[0])
                gold = data[1]
                print(pred, gold)
