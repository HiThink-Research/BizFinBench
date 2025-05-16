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
                predicted_answer = predict_data.get("涨跌情况", "")
            else:
                predicted_answer = ""

            score = 1.0 if str(correct_answer).strip() == str(predicted_answer).strip() else 0

            d['eval_result'] = {
                "result": "True" if score == 1 else "False",
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer
            }
            d['score'] = score
            corrects.append(score)
            total_scores.append(1)
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            d['eval_result'] = {"result": "False", "error": str(e)}
            d['score'] = 0
            print(f"Error processing data: {e}")
        
        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0
    return {"acc": overall_score}