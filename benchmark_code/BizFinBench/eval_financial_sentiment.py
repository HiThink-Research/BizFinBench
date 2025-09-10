import json
import re
import numpy as np
from utils import JsonPaser
import traceback

def evaluate_single_data(d):
    """
    对单条数据进行评分计算。
    Args:
        d (dict): 单条数据记录。
    Returns:
        dict: 更新后的数据记录，包含 eval_result 字段。
    """
    j_paser = JsonPaser()
    try:
        # matches = re.findall(r'"result":\s*"?(\[.*?\])', d["predict_result"])
        result = predict_data = j_paser.extract_json_from_text(d["predict_result"]).get('result',[])
        answer = json.loads(d["choices"][0]["message"]["content"][0]["text"])
        d["eval_result"] = {"result": "True"} if set(result) == set(answer) else {"result": "False"}
    except Exception as e:
        d["eval_result"] =  {"result": "False"}
        print(f"Error processing data: {e}")
        traceback.print_exc()
    return d

def evaluation(input_path, **kwargs):
    """
    对模型预测结果进行评分计算并更新文件内容。
    Args:
        input_path (str): 输入文件的路径，文件包含模型预测结果的数据。
        **kwargs: 可选参数，用于未来扩展功能。
    Returns:
        float: 所有有效评分的平均分。如果没有有效评分，则返回 0。
    """
    try:
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]


        out = []
        count = 0
        for i, d in enumerate(data):
            updated_data = evaluate_single_data(d)
            if updated_data["eval_result"]["result"] == "True":
                count += 1
            out.append(updated_data)

        with open(input_path, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')

        count = count / len(data) if data else 0
        return {'acc':count}
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {'acc':0}