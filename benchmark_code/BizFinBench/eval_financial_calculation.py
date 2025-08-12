import json
import re

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    
    out = []
    for d in data:
        try:
            choices = d.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    correct_answer = content[0].get("text", "")
                else:
                    correct_answer = content if isinstance(content, str) else ""
            else:
                correct_answer = d.get("messages", [])[-1].get("content", "")
                if isinstance(correct_answer, list):
                    correct_answer = correct_answer[0].get("text", "") if correct_answer else ""

            predict_result = d.get("predict_result", "")
            predicted_answer = None
            matches = re.findall(r'"answer":\s*"?([\-0-9.年月日]+)', predict_result)
            predicted_answer = matches[-1] if matches else None
            try:
                if predicted_answer is not None:
                    predicted_answer = float(predicted_answer)

                correct_float = float(correct_answer) if correct_answer else None

                if predicted_answer is not None and correct_float is not None:
                    score = 1.0 if abs(predicted_answer - correct_float) < 1e-6 else 0
                else:
                    score = 0
            except (ValueError, TypeError):
                try:
                    p = re.match(r'([0-9]+)年([0-9]+)月([0-9]+)日', str(predicted_answer).strip())
                    t = re.match(r'([0-9]+)年([0-9]+)月([0-9]+)日', str(correct_answer).strip())
                    score = 1.0 if all(p.group(i + 1).lstrip('0') == t.group(i + 1).lstrip('0') for i in range(3)) else 0
                except (AttributeError, ValueError):
                    score = 1.0 if str(predicted_answer).strip() == str(correct_answer).strip() else 0

            d['eval_result'] = {
                "result": "True" if score == 1 else "False",
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer
            }
            d['score'] = score
            corrects.append(score)
            total_scores.append(1)
            
        except Exception as e:
            d['eval_result'] = {"result": "False", "error": str(e)}
            d['score'] = 0
            total_scores.append(1)
            print(f"Error processing data: {e}")
        
        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0
    return {"acc": overall_score}          
