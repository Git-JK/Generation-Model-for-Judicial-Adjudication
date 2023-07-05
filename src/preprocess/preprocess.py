import os
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def set_seed(seed_num):
    """
    set random seed to get same result of training
    Args:
        seed_num (int): radom seed num
    """
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_num", type=int, default=10)
    
    parser.add_argument("--data_path", type=str, default="CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json")
    parser.add_argument("--model_name", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--law_path", type=str, default="CAIL2018_ALL_DATA/law_tmp.txt")
    parser.add_argument("--target_path", type=str, default="new_data/train.json")
    
    return parser.parse_args()


def load_json_file(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list
            
def load_law(file_path):
    law_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace("【", "[").replace("】", "] ").replace("\n", "")
            law_list.append(line)
    return law_list    

def get_row_data(prompt_template, answer_template, origin_data, law_list, tokenizer):
    crime =','.join(origin_data['meta']['accusation'])
    prisonment = origin_data['meta']['term_of_imprisonment']
    laws_num = list(set(origin_data['meta']['relevant_articles']))
    laws_artical = ";".join(str(law_list[e]) for e in laws_num)
    death = prisonment['death_penalty']
    number = prisonment['imprisonment']
    life = prisonment['life_imprisonment']
    period = ''
    if death == True:
        period = '死刑'
    elif life == True:
        period = '无期徒刑'
    else:
        period = str(number) + '年有期徒刑'
    content = prompt_template.replace('XX', origin_data['fact'])
    summary = answer_template.replace('AA',crime).replace('BB',period).replace('CC',laws_artical)
    if len(tokenizer.encode(text=content, add_special_tokens=False)) + len(tokenizer.encode(text=summary, add_special_tokens=False)) > 2047:
        return None
    else:
        return {
            "content": content,
            "summary": summary
        }
        
    

def main(args):
    data_path = args.data_path
    law_path = args.law_path
    model_name = args.model_name
    target_path = args.target_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_list = load_json_file(data_path)
    law_list = load_law(law_path)
    prompt = "请扮演一个法官.下面有1个刑事案件'XX',请根据中华人民共和国刑法给出案件的罪名和应当判处的刑期,请以{罪名:罪名内容, 刑期:刑期内容, 法条:相关条目}的格式返回"
    answer = "罪名:AA,刑期:BB,法条:CC"
    save_data_list = []
    for i, origin_data in tqdm(enumerate(data_list), total=len(data_list)):
        row_data = get_row_data(prompt, answer, origin_data, law_list, tokenizer)
        if row_data is not None:
            save_data_list.append(json.dumps(row_data, ensure_ascii=False) + '\n')
    # save_data_list = random.sample(save_data_list, int(len(save_data_list) / 5))
    with open(target_path, 'w', encoding='utf-8') as f:
        f.writelines(save_data_list)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed_num)
    main(args)