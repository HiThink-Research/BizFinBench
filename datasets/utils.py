import json

with open("TESTSET__金融股票涨跌评估数据集__0-0-1.jsonl",'r') as f:
    data = [json.loads(l) for l in f]

out = []
for d in data:
    filtered = {
        "messages": d.get("messages"),
        "choices": d.get("choices"),
    }
    out.append(filtered)

with open("Stock_Price_Prediction.jsonl",'w',encoding='utf-8') as f:
    for t in out:
        f.write(json.dumps(t,ensure_ascii=False)+'\n')