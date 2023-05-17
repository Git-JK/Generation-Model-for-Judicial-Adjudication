import os
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output/adgen-chatglm-6b-pt-128-0.02/generated_predictions.json")
    parser.add_argument("--save_path", type=str, default="output/adgen-chatglm-6b-pt-128-0.02/predictions.json")
    
    return parser.parse_args()


def load_json_file(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list
            

def main(args):
    data_path = args.data_path
    save_path = args.save_path
    data_list = load_json_file(data_path)
    cleaned_data_list = []
    for row in data_list:
        cleaned_row = {}
        labels = row['labels']
        labels = labels.replace("<image_-100>", "").strip()
        predict = row['predict']
        cleaned_row['labels'] = labels
        cleaned_row['predict'] = predict
        cleaned_data_list.append(json.dumps(cleaned_row, ensure_ascii=False) + "\n")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_data_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)