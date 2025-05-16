import json
import re
import os
import numpy as np
import copy

def data_preprocess(input_path, input_file, save_dir='step1'):
    prompt = '''你是一个金融内容评测专家，正在进行金融数据描述准确性的评估。

    你的打分需要考虑两个方面：

    1. 数据错用：<answer>中的指标数字应该和<question>中的对应上，不应该出现指标错用、时间错用等情况，例如：从55.32增长到59.14描述成从55.24增长到58.32。

    2. 数据描述: 只需判断<answer>中是否存在描述与具体数据相背的情况，如果有则得0分。例如：一连串数值越来越大，描述却是递减、两两比较错误，或最大、最小值判断错误、涨跌幅大于零说成下跌、主力资金小于零说成资金流入。当<question>中未取到数或取到的数据为空时，<answer>中回答不能说该数据为0，如果有则得0分。

    | 分数    | 描述                                                         |
    | ------- | ------------------------------------------------------------ | 
    | **100** | 完全正确。趋势描述和数据描述的均完全正确，且语言流程，无幻觉。 |
    | **60** | 部分错误。数据趋势描述正确，但数据值描述错误，例如从55.32增长到59.14描述成从55.24增长到58.32。 |
    | **0** | 错误较多。数据趋势描述错误即不得分，例如数据趋势是越来越大，描述是递减。       |

    ### 以下是你需要评分的案例：
    <question>
    {question}
    </question>

    <answer>
    {predict_result}
    </answer>

    ### 要求：
    返回结果以json格式展示，参考：
    {"评分分数":"xx","描述": "xxxx"}

    ### 回答如下：
'''.strip()
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)

    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    with open(os.path.join(input_path, input_file), encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    template={"query":"","messages":[{"role":"user","content":[{"text":"","type":"text"}]}],"choices":[{"index":0,"message":{"role":"assistant","content":[{"text":"","type":"text"}]}}],"model_prompt_tmpl":"","model_prompt_placeholder":[]}
    template_copy = copy.deepcopy(template)
    for i, line_info in enumerate(data):
        predict_result = line_info["predict_result"]
        question = line_info["query"]
        inst = prompt.replace("{question}",question).replace("{predict_result}",predict_result)
        template_copy = copy.deepcopy(template)
        template_copy["query"] = inst
        template_copy["messages"][0]["content"][0]["text"] = inst
        template_copy['question'] = question
        template_copy['answer'] = predict_result
        with open(out_file, 'a', encoding='utf8') as fw:
            json.dump(template_copy, fw, ensure_ascii=False)
            fw.write('\n')

def evaluation(input_path, **kwargs):
    sum_score = 0
    data = []
    num = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            l = json.loads(line)
            predict_result = l["predict_result"]
            pattern = r'```json\s*({.*?})\s*```'
            matches = re.findall(pattern, predict_result, re.DOTALL)

            if matches:
                eval_result_str = matches[0]
            else:
                eval_result_str = predict_result.split("\n\n")[0].strip()

            try:
                eval_result = json.loads(eval_result_str)
                l["eval_result"] = eval_result
                num += 1
                sum_score += int(eval_result["评分分数"]) 
            except json.JSONDecodeError as e:
                l["eval_result"] = eval_result_str
                num += 1
                sum_score += 0
                print(f"Invalid JSON: {e}")
            except KeyError as e:
                l["eval_result"] = eval_result_str
                num += 1
                sum_score += 0
                print(f"KeyError: Missing '评分分数' in JSON data")
            except Exception as e:
                l["eval_result"] = eval_result_str
                num += 1
                sum_score += 0
                print(f"Unexpected error: {e}")
            data.append(l)

    with open(input_path, "w", encoding="utf-8") as f:
        for o in data:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    sum_score = (sum_score / num) / 100 if num != 0 else 0
    return {"acc": sum_score}