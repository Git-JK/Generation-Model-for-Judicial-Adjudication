import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_truncated/valid.json")
    parser.add_argument("--target_path", type=str, default="new_data/valid.json")
    
    return parser.parse_args()

def main(args):
    data_path = args.data_path
    target_path = args.target_path
    with open(data_path, 'r', encoding="utf-8") as f:
        data_list = json.load(f)
    new_data_list = []
    for data_dict in data_list:
        new_data_list.append(json.dumps(data_dict, ensure_ascii=False) + "\n")
    with open(target_path, 'w', encoding="utf-8") as f:
        f.writelines(new_data_list)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
    