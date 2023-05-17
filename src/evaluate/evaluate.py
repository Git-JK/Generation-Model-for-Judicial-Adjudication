import sys

sys.path.append(".")
import os
import json
from math import log
from tqdm import tqdm
from loguru import logger
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="analysis")
    parser.add_argument("--test_set_path", type=str, default="output/adgen-chatglm-6b-pt-128-0.02/predictions.json")
    
    return parser.parse_args()

def load_json_file(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list

def extract_info(label: str):
    if label.find("罪名") != -1 and label.find("刑期") != -1:
        crime_list = label.split("罪名:")[1].split(",刑期")[0].split(",")
    elif label.find("罪名") != -1 and label.find("法条") != -1:
        crime_list = label.split("刑期:")[1].split(",法条")[0].split(",")
    elif label.find("罪名") != -1:
        crime_list = label.split("罪名:")[1].split(",")
    if label.find("刑期") != -1 and label.find("法条") != -1:
        tmp_period = label.split("刑期:")[1].split(",法条")[0]
    elif label.find("刑期") != -1:
        tmp_period = label.split("刑期:")[1]
    else:
        tmp_period = None
    period = -3
    if tmp_period is not None:
        if tmp_period.find("有期徒刑") != -1:
            period = int(tmp_period.split("年有期徒刑")[0])
        elif tmp_period.find("死刑") != -1:
                period = -2
        elif tmp_period.find("无期徒刑") != -1:
            period = -1
            
    if label.find("法条") != -1:
        law_tmp_list = label.split("法条:")[1].split(" ")
        law_list = []
        for tmp_str in law_tmp_list:
            if len(tmp_str) == 0:
                continue
            if tmp_str[0] == "第" and tmp_str[-1] == "条":
                law_list.append(tmp_str)
    else:
        law_list = []
    return {
        "crime_list": crime_list,
        "period": period,
        "law_list": law_list
    }
    

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
    
    data_list = load_json_file(args.test_set_path)
    
    crime_score = []
    period_score = []
    law_score = []
    crime_name_list = []
    law_name_list = []
    
    for i, data in tqdm(enumerate(data_list)):
        gold = data['labels']
        pred = data['predict']
        gold_dict = extract_info(gold)
        pred_dict = extract_info(pred)
        crime_score.append(set(gold_dict['crime_list']) == set(pred_dict['crime_list']))
        crime_name_list.extend(gold_dict['crime_list'])
        law_name_list.extend(gold_dict['law_list'])
        gold_period = gold_dict['period']
        pred_period = pred_dict['period']
        if gold_period == -2:
            if pred_period == -2:
                period_score.append(1.0)
            else:
                period_score.append(0.0)
        elif gold_period == -1:
            if pred_period == -1:
                period_score.append(1.0)
            else:
                period_score.append(0.0)
        else:
            if pred_period <= -1:
                period_score.append(0.0)
            else:
                v = abs(log(gold_period + 1) - log(pred_period + 1))
                if v <= 0.2:
                    period_score.append(1.0)
                elif v <= 0.4:
                    period_score.append(0.8)
                elif v <= 0.6:
                    period_score.append(0.6)
                elif v <= 0.8:
                    period_score.append(0.4)
                elif v <= 1.0:
                    period_score.append(0.2)
                else:
                    period_score.append(0.0)
        law_score.append(set(gold_dict['law_list']) == set(pred_dict['law_list']))
    
    crime_name_list = set(crime_name_list)
    law_name_list = set(law_name_list)
    
    crime_info_dict = {}
    law_info_dict = {}
    
    for crime_name in crime_name_list:
        crime_info_dict[crime_name] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }
    
    for law_name in law_name_list:
        law_info_dict[law_name] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }
        
    for i, data in tqdm(enumerate(data_list)):
        gold = data['labels']
        pred = data['predict']
        gold_dict = extract_info(gold)
        pred_dict = extract_info(pred)
        for crime_name in crime_name_list:
            in_gold = crime_name in gold_dict["crime_list"]
            in_pred = crime_name in pred_dict["crime_list"]
            if in_gold:
                if in_pred:
                    crime_info_dict[crime_name]['tp'] += 1
                else:
                    crime_info_dict[crime_name]['fp'] += 1
            else:
                if in_pred:
                    crime_info_dict[crime_name]['fn'] += 1
                else:
                    crime_info_dict[crime_name]['tn'] += 1
        for law_name in law_name_list:
            in_gold = law_name in gold_dict["law_list"]
            in_pred = law_name in pred_dict["law_list"]
            if in_gold:
                if in_pred:
                    law_info_dict[law_name]['tp'] += 1
                else:
                    law_info_dict[law_name]['fp'] += 1
            else:
                if in_pred:
                    law_info_dict[law_name]['fn'] += 1
                else:
                    law_info_dict[law_name]['tn'] += 1
                    
    for crime_name, value_dict in crime_info_dict.items():
        if value_dict['tp'] != 0:
            precision = value_dict['tp'] / (value_dict['tp'] + value_dict['fp'])
        else:
            precision = 0.0
        if value_dict['tp'] != 0:
            recall = value_dict['tp'] / (value_dict['tp'] + value_dict['fn'])
        else:
            recall = 0.0
        if precision + recall > 1e-6:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        crime_info_dict[crime_name]['f1_score'] = f1_score
    
    for law_name, value_dict in law_info_dict.items():
        if value_dict['tp'] != 0:
            precision = value_dict['tp'] / (value_dict['tp'] + value_dict['fp'])
        else:
            precision = 0.0
        if value_dict['tp'] != 0:
            recall = value_dict['tp'] / (value_dict['tp'] + value_dict['fn'])
        else:
            recall = 0.0
        if precision + recall > 1e-6:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        law_info_dict[law_name]['f1_score'] = f1_score
        
    crime_f1_score = [value_dict['f1_score'] for k, value_dict in crime_info_dict.items()]
    law_f1_score = [value_dict['f1_score'] for k, value_dict in law_info_dict.items()]
    
    crime_accuracy = sum(crime_score) / len(crime_score)
    crime_macro_f1 = sum(crime_f1_score) / len(crime_f1_score)
    period_accuracy = sum(period_score) / len(period_score)
    law_accuracy = sum(law_score) / len(law_score)
    law_macro_f1 = sum(law_f1_score) / len(law_f1_score)
    logger.info(f"crime Accuracy: {crime_accuracy}")
    logger.info(f"crime Macro F1 Score: {crime_macro_f1}")
    logger.info(f"period Accuracy: {period_accuracy}")
    logger.info(f"law Macro F1 Score: {law_macro_f1}")
    logger.info(f"law Accuracy: {law_accuracy}")
    
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
