import json
f = open("exercise_contest/data_train.json",'r',encoding='utf-8')
json_file_path = './train.json'
json_file = open(json_file_path, mode='w',encoding='utf-8')
data = []
for line in f:
    data.append(json.loads(line))
l = open("law_tmp.txt",'r',encoding='utf-8')
law_list = []
for line in l.readlines():
    law_list.append(line)
prompt = "请扮演一个法官.下面有1个刑事案件'XX',请根据中华人民共和国刑法给出案件的罪名和应当判处的刑期,请以{罪名:罪名内容, 刑期:刑期内容, 法条:相关条目}的格式返回"
answer = "罪名:AA,刑期:BB,法条:CC"
save_json = []
N = 2048
for item in data:
    crime = ','.join(item['meta']['accusation'])
    prisonment = item['meta']['term_of_imprisonment']
    laws_num = list(set(item['meta']['relevant_articles']))
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
    content = prompt.replace('XX',item['fact'])
    summary = answer.replace('AA',crime).replace('BB',period).replace('CC',laws_artical)
    if len(content)+len(summary) > N:
        content = content[:N-len(summary)]
        print(len(content),len(summary))
    tmp = {
        'content':content,
        'summary':summary
    }
    save_json.append(tmp)
# print(save_json)
json.dump(save_json, json_file, ensure_ascii=False) # 保存中文
