import json
import re
import os
import numpy as np

def data_preprocess(input_path, input_file, save_dir='step1'):
    '''judge预处理函数，将被评估模型的结果嵌入到评测提示词中，生成一个新的 JSON 文件用于后续的评分任务'''
    prompt = '''你是一个金融内容评测专家，可以通过以下几个维度对模型生成得内容进行合理、正确得评分；

### **1. 准确性 (Accuracy)**

评估答案是否完全符合金融领域的标准知识和常识，重点关注概念、公式、计算、结论是否正确。

| 等级    | 描述                                                         | 分数范围 |
| ------- | ------------------------------------------------------------ | -------- |
| **5分** | 完全正确。所有概念、公式、计算、结论无误，完全符合金融理论和实际应用。 | 90-100   |
| **4分** | 基本正确。核心概念和计算正确，少量细节或边缘部分略有偏差。   | 75-89    |
| **3分** | 部分错误。关键概念正确，但存在明显错误或不一致，部分计算或结论偏差较大。 | 60-74    |
| **2分** | 错误较多。多个关键部分出错，影响了整体理解和计算过程。       | 40-59    |
| **1分** | 严重错误。答案完全错误，核心理论或计算无法支持结论。         | 0-39     |

### **2. 完整性 (Completeness)**

评估答案是否涵盖所有问题要求，是否提供了充分的背景信息、推理过程和支持性细节。

| 等级    | 描述                                                         | 分数范围 |
| ------- | ------------------------------------------------------------ | -------- |
| **5分** | 完全完整。涵盖了所有核心内容，充分解释了相关背景、方法、步骤，提供了清晰的计算过程。 | 90-100   |
| **4分** | 大部分完整。覆盖了问题的核心内容，少数非核心部分略有遗漏。   | 75-89    |
| **3分** | 部分遗漏。遗漏了几个关键细节或步骤，解答不够全面，部分内容较简略。 | 60-74    |
| **2分** | 缺失较多。多个关键信息或步骤遗漏，未能提供完整的分析或解答。 | 40-59    |
| **1分** | 极为不完整。缺少关键内容，无法回答问题的核心或大部分问题。   | 0-39     |

### **3. 分析深度 (Analytical Depth)**

评估模型在解答问题时的推理深度和多角度思考，是否能够给出深入的分析，考虑到各个可能的影响因素。

| 等级    | 描述                                                         | 分数范围 |
| ------- | ------------------------------------------------------------ | -------- |
| **5分** | 深入分析。提供多维度的分析，清晰展示了推理过程，考虑到多个影响因素，展现了深度思考。 | 90-100   |
| **4分** | 较为深入。分析覆盖了多个重要因素，但缺少对某些复杂因素的深入探讨。 | 75-89    |
| **3分** | 表面分析。分析较为简单，未能充分考虑问题的各个维度，缺少深入推理。 | 60-74    |
| **2分** | 浅显分析。分析层次较浅，未能探讨问题的复杂性，缺乏有效推理。 | 40-59    |
| **1分** | 无分析。没有提供有效的分析过程，仅做表面描述。               | 0-39     |

### **4. 清晰度 (Clarity)**

评估答案的表达是否简洁明了，逻辑是否清晰，语言是否流畅，确保读者易于理解。

| 等级    | 描述                                                         | 分数范围 |
| ------- | ------------------------------------------------------------ | -------- |
| **5分** | 极其清晰。结构清晰，表达简洁流畅，逻辑严密，语言易于理解，术语准确。 | 90-100   |
| **4分** | 清晰。表达清楚，结构合理，但部分内容略显复杂或冗长。         | 75-89    |
| **3分** | 一般清晰。表达略显复杂，部分内容需要多次阅读才能理解，结构略显松散。 | 60-74    |
| **2分** | 不够清晰。表达不简洁，部分内容冗长或不明确，理解有一定困难。 | 40-59    |
| **1分** | 非常混乱。语言不清晰，逻辑松散，内容难以理解，术语使用不当。 | 0-39     |

#### 总评分：

- 每个维度的最高分为 **25分**，总分 **100分**。
- 可以根据实际需求调整维度的权重。常见的权重分配如下：
  - **准确性**：40%
  - **完整性**：30%
  - **分析深度**：20%
  - **清晰度**：10%

#### **总评分示例**：

- 准确性： 23分（40%）
- 完整性： 21分（30%）
- 分析深度： 17分（20%）
- 清晰度： 8分（10%）

**总分** = 23×0.40+21×0.30+17×0.20+8×0.10=9.2+6.3+3.4+0.8=19.723 

### 以下是你需要评分的案例：

question：{question}
<answer>
{predict_result}
</answer>

### 要求：

返回结果以json格式展示，参考：

{"评分结果":{"准确性":{"得分":23,"描述":"该答案准确地解释了货币政策和财政政策的作用及其常见措施，如降息、量化宽松、政府支出、减税等，均符合金融领域的标准知识和实际应用。概念和措施没有明显错误，符合经济学和金融学的基本原理。"},"完整性":{"得分":22,"描述":"答案完整地涵盖了货币政策和财政政策的关键措施，并指出了这两者结合使用的可能性。提到了实施政策时的时机、潜在副作用以及国际协调的需要，涉及了多个重要方面。不过，某些细节，如具体的政策效果评估和可能的长期影响，未做更深入探讨。"},"分析深度":{"得分":18,"描述":"从多个维度分析了货币政策和财政政策的作用，并提到了一些重要的影响因素，如时机、副作用和国际合作等。尽管涉及到了一些复杂因素，但在某些方面（如副作用的深度分析、政策效果的具体评估）可进一步加强讨论。"},"清晰度":{"得分":24,"描述":"表达非常清晰，逻辑严密，语言流畅。各部分内容结构合理，易于理解，专业术语准确无误。整体条理清晰，读者能轻松跟随作者的思路。"}},"总评分":{"准确性":9.2,"完整性":6.6,"分析深度":3.6,"清晰度":2.4,"最终评分":21.8,"总分":87.2}}

### 各维度得分情况如下：
'''.strip()
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)

    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    data = [json.loads(i) for i in open(os.path.join(input_path, input_file), encoding='utf8')]
    for id, line_info in enumerate(data):
        predict_result = line_info["predict_result"]
        question = line_info["messages"][0]["content"][0]["text"] if isinstance(line_info["messages"][0]["content"],list) else line_info["messages"][0]["content"]
        inst = prompt.replace("{question}",question).replace("{predict_result}",predict_result)
        with open(out_file, 'a', encoding='utf8') as fw:
            json.dump({'instruction': inst,'question':question,'model_predict':predict_result}, fw, ensure_ascii=False)
            fw.write('\n')

def evaluation(input_path, **kwargs):
    """
    对模型预测结果进行评分计算并更新文件内容。
    
    Args:
        input_path (str): 输入文件的路径，文件包含模型预测结果的数据。
        **kwargs: 可选参数，用于未来扩展功能。
    
    Returns:
        float: 所有有效评分的平均分。如果没有有效评分，则返回 0。
    """
    def parse_eval_result(predict_result):
        """
        解析 predict_result 字段，提取 JSON 数据或总分信息。
        
        Args:
            predict_result (str): 模型预测结果字符串。
        
        Returns:
            dict or str: 解析后的 JSON 数据或原始字符串。
        """
        pattern = r'```json\s*({.*?})\s*```'
        matches = re.findall(pattern, predict_result, re.DOTALL)
        if matches:
            eval_result_str = matches[0].strip().strip("```").lstrip("json").strip()
        else:
            eval_result_str = predict_result.strip().strip("```").lstrip("json").strip()
        try:
            return json.loads(eval_result_str)
        except Exception:
            return eval_result_str

    def extract_score(eval_result):
        """
        从解析后的 eval_result 中提取总分。
        
        Args:
            eval_result (dict or str): 解析后的预测结果。
        
        Returns:
            float: 提取的总分值。如果无法提取，则返回 0。
        """
        if isinstance(eval_result, dict) and "总评分" in eval_result and "总分" in eval_result["总评分"]:
            return float(eval_result["总评分"]["总分"])
        elif isinstance(eval_result, str):
            pattern = r'"总分"\s*:\s*(\d+(\.\d+)?)'
            match = re.search(pattern, eval_result)
            if match:
                return float(match.group(1))
        return 0

    
    sum_score = 0
    num = 0
    out = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i,line in enumerate(lines):
        try:
            l = json.loads(line)
            eval_result = parse_eval_result(l["predict_result"])
            l["eval_result"] = eval_result
            score = extract_score(eval_result)
            if score > 0:
                sum_score += score
                num += 1
        except Exception as e:
            print(f"第{i+1}条出现报错,Error processing line: {e}")
            l["eval_result"] = l.get("predict_result", "")
        finally:
            out.append(l)

    dir_path, file_name = os.path.split(input_path)
    name, ext = os.path.splitext(file_name)  
    new_file_name = f"{name}_eval_result{ext}"
    new_path = os.path.join(dir_path, new_file_name)
    with open(new_path, "w", encoding="utf-8") as f:
        for o in out:
            try:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error writing to file: {e}")

    
    sum_score = (sum_score / num) / 100 if num != 0 else 0
    return {'acc':sum_score}