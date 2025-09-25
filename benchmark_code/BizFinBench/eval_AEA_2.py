import json
import re
from utils import JsonPaser

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    
    out = []
    for d in data:
        try:
            choices = d.get("choices", [])
            correct_answer = ""
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    correct_answer = content[0].get("text", "")
                else:
                    correct_answer = content    
            
            predict_result_str = d.get("predict_result", "")

            j_paser = JsonPaser()
            
            predict_data = j_paser.extract_json_from_text(predict_result_str)
            # import pdb;pdb.set_trace()

            if predict_data:
                if predict_data.get("相关新闻序号", ""):
                    predicted_answers = predict_data["相关新闻序号"]
                elif any('序号' in key for key in predict_data):
                    predicted_answers = predict_data["相关内容序号"] if predict_data.get("相关内容序号", "") else predict_data.get("无关内容序号", "")
                else:
                    predicted_answers = predict_data["Relevant Content Numbers"] if predict_data.get("Relevant Content Numbers", "") else predict_data.get("Irrelevant Content Numbers", "")
            else:
                predicted_answers = []


            if not isinstance(predicted_answers, list):
                predicted_answers = []

            correct_answers = d['choices'][0]['message']['content'][0]['text']
            correct_answers = [int(x) for x in correct_answers.split(',')]
            if not isinstance(correct_answers, list):
                correct_answers = []

            predicted_answers = [int(x) for x in predicted_answers]

            # fix: 修改评分规则，当全符合的时候，得分为1；当部分符合的时候，分两种情况：
            # 1.predicted_answers是correct_answers的子集，score为predicted_answers数量/correct_answers的数量 2.否则score为0
            
            if sorted(predicted_answers) == sorted(correct_answers):
                score = 1.0
                d['eval_result'] = {"result": "True"}
            elif predicted_answers != [] and set(predicted_answers).issubset(set(correct_answers)):
                score = round(len(predicted_answers)/len(correct_answers), 2)
                d['eval_result'] = {"result": "Partially Correct"}
            else:
                score = 0.0
                d['eval_result'] = {"result": "False"}
            
            
            d['score'] = score
            
            corrects.append(score)
            total_scores.append(1)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            d['eval_result'] = {"result": f"error: {str(e)}"}
            d['score'] = 0
            total_scores.append(1)

        out.append(d)

    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0

    return {"acc": overall_score}  
