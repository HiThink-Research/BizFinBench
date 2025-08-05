import json
import re
from datetime import datetime


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

            predict_result_raw = d.get("predict_result", "")
            predicted_answer = ""
            matches = re.findall(r'"answer":\s*"?([0-9.\/\-年月日是否]+)', predict_result_raw)
            predicted_answer = matches[-1] if matches else ''

            def parse_date(date_str):
                if not isinstance(date_str, str):
                    return None

                date_str = date_str.strip()
                if date_str.startswith('"') and date_str.endswith('"'):
                    date_str = date_str[1:-1]
                
                date_formats = ["%Y年%m月%d日", "%Y-%m-%d", "%Y/%m/%d"]
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                return None

            correct_date = parse_date(correct_answer)
            predicted_date = parse_date(predicted_answer)

            if correct_date and predicted_date:
                score = 1.0 if correct_date == predicted_date else 0
            else:
                try:
                    score = 1.0 if float(correct_answer) == float(predicted_answer) else 0
                except (TypeError, ValueError):
                    score = 1.0 if str(correct_answer).strip() == str(predicted_answer).strip() else 0

            d['eval_result'] = {"result": "True" if score == 1 else "False"}
            d['score'] = score

            corrects.append(score)
            total_scores.append(1)

        except Exception as e:
            d['eval_result'] = {"result": "False", "error": str(e)}
            d['score'] = 0
            total_scores.append(1)
            print(f"处理数据时出错: {e}")

        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0
    return {"acc": overall_score}