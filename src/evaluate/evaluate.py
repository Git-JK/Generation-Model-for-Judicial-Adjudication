import sys

sys.path.append(".")
import os
import json
import torch
from loguru import logger
from torch.utils.data import DataLoader
import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.dataset.judicial_adjudication_dataset import JudicialAdjudicationDataset
from src.evaluate.evaluator import JudicialAdjudicationEvaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_id", type=str, )
    parser.add_argument("--work_dir", type=str, default="analysis")
    parser.add_argument("--test_set_path", type=str, default="data/test.json")
    parser.add_argument("--model_name", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--checkpoint_path", type=str, default="output/checkpoint-956")
    
    return parser.parse_args()


def build_dataset(args):
    data_list = []
    with open(args.test_set_path, 'r', encoding='utf-8') as f:
        json_list = f.readlines()
        for row in json_list:
            data_dict = json.loads(row)
            data_list.append([data_dict['content'], data_dict['summary']])
    test_dataset = JudicialAdjudicationDataset(data_list)
    
    return test_dataset
    

def build_model(args):
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(args.model_name, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(args.checkpoint_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    
    if args.device_id is not None:
        device_id = args.device_id
        model.to(device_id)
    else:
        device_id = None
        model.to('cpu')
    
    return model

def main(args):
    
    args.evaluate_dir = os.path.join(args.work_dir, "evaluate")
    args.log_dir = os.path.join(args.work_dir, "log")
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if not os.path.exists(args.evaluate_dir):
        os.makedirs(args.evaluate_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    log_path = os.path.join(args.log_dir, "analyze_log.log")
    logger.add(sink=log_path, backtrace=True, diagnose=True)
    
    model = build_model(args)
    
    test_dataset = build_dataset(args)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    model.eval()
    
    test_evaluator = JudicialAdjudicationEvaluator(args, model, test_loader)
    logger.info("Testing Metrics...")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
