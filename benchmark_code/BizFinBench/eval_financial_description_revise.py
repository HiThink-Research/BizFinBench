import json
import re
import os
import numpy as np
import copy
import asyncio
import concurrent.futures
from tools.ExternalApi import GPT4OClient
prompt_1='''请充当强大的修订者，根据以下评估标准修订 AI 助手对 instruction 生成的输出。如果模型生成的输出已足够好，则直接返回原始输出，无需任何修改或额外说明。
**instruction：**
<instruction>
{question}
</instruction>

**评估标准:  **
<评估标准>
数据错用：<模型生成的输出>中的指标数字应该和<instruction>中的对应上，不应该出现指标错用、时间错用等情况，例如：从55.32增长到59.14描述成从55.24增长到58.32。
</评估标准>

**模型生成的输出：**
<模型生成的输出>
{answer}
</模型生成的输出>

不要为您的响应提供任何解释。
仅输出完整的修订答案，而无需说任何其他内容。'''
prompt_2='''请充当强大的修订者，根据以下评估标准修订 AI 助手对 instruction 生成的输出。如果模型生成的输出已足够好，则直接返回原始输出，无需任何修改或额外说明。
**instruction：**
<instruction>
{question}
</instruction>

**评估标准:  **
<评估标准>
数据描述: 只需判断<模型生成的输出>中是否存在描述与具体数据相背的情况，如果有则得0分。例如：一连串数值越来越大，描述却是递减、两两比较错误，或最大、最小值判断错误、涨跌幅大于零说成下跌、主力资金小于零说成资金流入。当<instruction>中未取到数或取到的数据为空时，<模型生成的输出>中回答不能说该数据为0，如果有则得0分。
</评估标准>

**模型生成的输出：**
<模型生成的输出>
{answer}
</模型生成的输出>

不要为您的响应提供任何解释。
仅输出完整的修订答案，而无需说任何其他内容。'''
prompt_judge='''你是一个金融内容评测专家，正在进行金融数据描述准确性的评估。

    请你评估时将<answer>与<reference answer>进行比较，同时考虑以下两个方面进行打分：

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
    {answer}
    </answer>

    ### 以下是一个参考输出：
    <reference answer>
    {ref_answer}
    </reference answer>

    ### 要求：
    返回结果以json格式展示，参考：
    {"评分分数":"xx","描述": "xxxx"}

    ### 回答如下：
'''.strip()
# 获取模板填充后的提示
def get_prompt(prompt_template: str, params: dict):
    for key, value in params.items():
        placeholder = f"{{{key}}}"
        prompt_template = prompt_template.replace(placeholder, str(value))
    return prompt_template

# 格式化上传数据
def upload_pass_format(query: str, answer: str, model_prompt_tmpl=None, is_return_dict=True) -> str | dict:
    template = {
        "query": query,
        "messages": [{"role": "user", "content": [{"text": query, "type": "text"}]}],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": [{"text": answer, "type": "text"}]}}],
        "model_prompt_tmpl": model_prompt_tmpl or "",
        "model_prompt_placeholder": []
    }
    return copy.deepcopy(template) if is_return_dict else json.dumps(template, ensure_ascii=False)

async def run_gpt4o_tasks(revise_list):
    gpt4o = GPT4OClient()
    result = await gpt4o.texts2texts(revise_list, 0)
    return result
def run_async_in_thread(revise_list):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_gpt4o_tasks(revise_list))
    finally:
        loop.close() 
    return result


def data_preprocess(input_path, input_file, save_dir='step1'):
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)
    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    with open(os.path.join(input_path, input_file), encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    template={"query":"","messages":[{"role":"user","content":[{"text":"","type":"text"}]}],"choices":[{"index":0,"message":{"role":"assistant","content":[{"text":"","type":"text"}]}}],"model_prompt_tmpl":"","model_prompt_placeholder":[]}
    first_revise = []
    # 第一次修正，使用prompt1
    print('开始第一轮修正')
    for i, line_info in enumerate(data):
        predict_result = line_info["predict_result"]
        question = line_info["query"]
        inst = prompt_1.replace("{question}",question).replace("{answer}",predict_result)
        template_copy = copy.deepcopy(template)
        template_copy["query"] = inst
        template_copy["messages"][0]["content"][0]["text"] = inst
        template_copy['question'] = question
        template_copy['answer'] = predict_result
        first_revise.append(template_copy.copy())
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(run_async_in_thread, first_revise)
        first_revise_result = future.result() 
    
    temp = []
    for result,origin in zip(first_revise_result,first_revise):
        origin['predict_result'] = result
        temp.append(origin)
    first_revise_result = temp
    second_revise = []
    print('开始第二轮修正')
    for i, line_info in enumerate(first_revise_result):
        predict_result = line_info["predict_result"] # 第一轮修正结果
        question = line_info["question"]
        inst = prompt_2.replace("{question}",question).replace("{answer}",predict_result) # 开始第二轮修正
        template_copy = copy.deepcopy(template)
        template_copy["query"] = inst
        template_copy["messages"][0]["content"][0]["text"] = inst
        template_copy['question'] = question
        template_copy['answer'] = line_info['answer'] # 原始需要被评估的内容
        template_copy['first_revise_result'] = predict_result
        second_revise.append(template_copy.copy())
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(run_async_in_thread, second_revise)
        second_revise_result = future.result() 
    temp = []
    for result,origin in zip(second_revise_result,second_revise):
        origin['predict_result'] = result
        temp.append(origin)
    second_revise_result = temp
    judge_instructions = []
    for i,line_info in enumerate(second_revise_result):
        reference_result = line_info["predict_result"] # 第二轮修正结果，即最终的参考内容
        question = line_info["question"]
        answer = line_info["answer"]
        inst = prompt_judge.replace("{question}",question).replace("{answer}",answer).replace("{ref_answer}",reference_result) # 得到最终评估提示词
        template_copy = copy.deepcopy(template)
        template_copy["query"] = inst
        template_copy["messages"][0]["content"][0]["text"] = inst
        template_copy['question'] = question
        template_copy['answer'] = line_info['answer'] # 原始需要被评估的内容
        template_copy['first_revise_result'] = line_info['first_revise_result']
        template_copy['second_revise_result'] = reference_result
        judge_instructions.append(template_copy.copy())
    with open(out_file, 'w', encoding='utf8') as f:
        for ins in judge_instructions:
            f.write(json.dumps(ins, ensure_ascii=False) + '\n')
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
    sum_score = sum_score / num if num != 0 else 0
    return {"acc": sum_score}