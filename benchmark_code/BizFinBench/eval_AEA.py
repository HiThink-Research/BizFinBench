import json
import chardet
import re

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []

    with open(input_path, 'rb') as f:
        raw_data = f.read(1000)
        encoding = chardet.detect(raw_data)['encoding']
    with open(input_path, 'r', encoding=encoding) as f:
        data = [json.loads(l) for l in f]

    out = []

    for d in data:
        try:
            predict_result = d["predict_result"]
            
            try:
                predict_data = json.loads(predict_result)
                predicted_answers = predict_data.get("相关新闻序号", [])
            except json.JSONDecodeError:
                json_pattern = r'(\{.*?"相关新闻序号"\s*:\s*\[.*?\].*?\})'
                match = re.search(json_pattern, predict_result, re.DOTALL)
                
                if match:
                    try:
                        json_str = match.group(1)
                        predict_data = json.loads(json_str)
                        predicted_answers = predict_data.get("相关新闻序号", [])
                    except json.JSONDecodeError:
                        array_pattern = r'"相关新闻序号"\s*:\s*\[(.*?)\]'
                        array_match = re.search(array_pattern, predict_result, re.DOTALL)
                        if array_match:
                            array_str = array_match.group(1).strip()
                            predicted_answers = [int(num.strip()) for num in array_str.split(',') if num.strip().isdigit()]
                        else:
                            predicted_answers = []
                else:
                    predicted_answers = []

            if not isinstance(predicted_answers, list):
                predicted_answers = []

            correct_answers = d['choices'][0]['message']['content'][0]['text']
            correct_answers = [int(x) for x in correct_answers.split(',')]
            if not isinstance(correct_answers, list):
                correct_answers = []

            predicted_answers = [int(x) for x in predicted_answers]

            if sorted(predicted_answers) == sorted(correct_answers):
                score = 1.0
            else:
                score = 0.0

            d['eval_result'] = {"result": "True"} if score == 1.0 else {"result": "False"}
            d['score'] = score
            
            corrects.append(score)
            total_scores.append(1)
        except Exception as e:
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