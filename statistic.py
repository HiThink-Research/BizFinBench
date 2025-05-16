import json
import os
import pandas as pd
import traceback
import requests
import yaml
from tqdm import tqdm
from io import BufferedReader
import re
from loguru import logger
import base64

def statistic(path,config):
    is_submit = config.get("is_submit", "1")
    submit_status = False if is_submit == "0" else True
    logs = os.listdir(path)
    results = {}
    for l in logs:
        if ".log" not in l:
            continue
        with open(os.path.join(path,l),'r',encoding='utf-8') as f:
            result = json.load(f)
            task = l.split('.')[0]
            results[task] = result[-1]['Average']
        
        # 结果统一为List传入提交函数
        if isinstance(results[task], dict):
            scores = results[task]
        elif isinstance(results[task], float):
            scores = {"score":results[task]}
    with open(os.path.join(path,"statistic.jsonl"),'w',encoding='utf-8') as f:
        f.write(json.dumps(results,ensure_ascii=False)+'\n')

if __name__ == "__main__":
    pass
    # with open("/mnt/data/wsj/llm-general/law_config.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    # print(config)
    # statistic("/mnt/data/damien/llm-eval/eval_result/v0120/HithinkGPT-8B-Instruct",config)
