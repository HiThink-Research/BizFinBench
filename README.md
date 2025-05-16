<p align="center">
  <h1 align="center">BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs</h1>
    <p align="center">
    <strong>Guilong Lu</strong>
    ·
    <strong>Xuntao Guo</strong>
    ·
    <strong>Rongjunchen Zhang</strong>
    ·
    <strong>Wenqiao Zhu</strong>
    ·
    <strong>Ji Liu</strong>
  </p>
  📖<a href="https://arxiv.org/abs/25xx.xxxxx">Paper</a> |🏠<a href="https://hithink-research.github.io/BizFinBench/">Homepage</a></h3>|🤗<a href="https://huggingface.co/datasets/HiThink-Research/BizFinBench">Huggingface</a></h3>
<div align="center"></div>
<p align="center">
  <p>
In recent years, multimodal benchmarks for general domains have guided the rapid development of multimodal models on general tasks. However, the financial field has its peculiarities. It features unique graphical images (e.g., candlestick charts, technical indicator charts) and possesses a wealth of specialized financial knowledge (e.g., futures, turnover rate).

Large language models excel across general tasks, yet judging their reliability in logic‑heavy, precision‑critical domains such as finance, law and healthcare is still difficult. To address this challenge, we propose **BizFinBench**, the first benchmark grounded in real-world financial applications. BizFinBench consists of 6,781 well-annotated queries in Chinese, covering five dimensions: numerical calculation, reasoning, information extraction, prediction recognition and knowledge‐based question answering, which are mapped to nine fine-grained categories.

This dataset contains multiple subtasks, each focusing on a different financial understanding and reasoning ability, as follows:

| Dataset                                | Description                                                  | Evaluation Dimensions                                        | Volume |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------ |
| **Anomalous Event Attribution**        | A financial anomaly attribution evaluation dataset assessing models' ability to trace stock fluctuations based on given information (e.g., timestamps, news articles, financial reports, and stock movements). | Causal consistency, information relevance, noise resistance  | 1,064  |
| **Financial Numerical Computation**    | A financial numerical computation dataset evaluating models' ability to perform accurate numerical calculations in financial scenarios, including interest rate calculations, gain/loss computations, etc. | Calculation accuracy, unit consistency                       | 581    |
| **Financial Time Reasoning**           | A financial temporal reasoning evaluation dataset assessing models' ability to comprehend and reason about time-based financial events, such as "the previous trading day" or "the first trading day of the year." | Temporal reasoning correctness                               | 514    |
| **Financial Data Description**         | A financial data description evaluation dataset measuring models' ability to analyze and describe structured/unstructured financial data, e.g., "the stock price first rose to XX before falling to XX." | Trend accuracy, data consistency                             | 1,461  |
| **Stock Price Prediction**             | A stock price movement prediction dataset evaluating models' ability to forecast future stock price trends based on historical data, financial indicators, and market news. | Trend judgment, causal rationality                           | 497    |
| **Financial Named Entity Recognition** | A financial named entity recognition dataset assessing models' ability to identify entities (Person, Organization, Market, Location, Financial Products, Date/Time) in short/long financial news. | Recognition accuracy, entity category correctness            | 433    |
| **Emotion_Recognition**                | A financial sentiment recognition dataset evaluating models' ability to discern nuanced user emotions in complex financial market environments. Inputs include multi-dimensional data such as market conditions, news, research reports, user holdings, and queries, covering six emotion categories: optimism, anxiety, pessimism, excitement, calmness, and regret. | Emotion classification accuracy, implicit information extraction and reasoning correctness | 600    |
| **Financial Tool Usage**               | A financial tool usage dataset evaluating models' ability to understand user queries and appropriately utilize various financial tools (investment analysis, market research, information retrieval, etc.) to solve real-world problems. Tools include calculators, financial encyclopedia queries, search engines, data queries, news queries, economic calendars, and company lookups. Models must accurately interpret user intent, select appropriate tools, input correct parameters, and coordinate multiple tools when necessary. | Tool selection rationality, parameter input accuracy, multi-tool coordination capability | 641    |
| **Financial Knowledge QA**             | A financial encyclopedia QA dataset assessing models' understanding and response accuracy regarding core financial knowledge, covering key domains: financial fundamentals, markets, investment theories, macroeconomics, etc. | Query comprehension accuracy, knowledge coverage breadth, answer accuracy and professionalism | 990    |

## 📢 News 
- 🚀 [16/05/2025] We released <strong>BizFinBench</strong> benchmark, the first benchmark grounded in real-world financial applications.

## 💡 Highlights
- 🔥  **Benchmark:** We propose **BizFinBench**, the first evaluation benchmark in the financial domain that integrates business-oriented tasks, covering 5 dimensions and 9 categories. It is designed to assess the capacity of LLMs in real-world financial scenarios.
- 🔥  **Judge model:** We design a novel evaluation method, i.e., **Iterajudge**, which enhances the capability of LLMs as a judge by refining their decision boundaries in specific financial evaluation tasks.
- 🔥  **key insights:** We conduct a comprehensive evaluation with **25 LLMs** based on BizFinBench, uncovering key insights into their strengths and limitations in financial applications.


## 🛠️ Usage

### Contents
```
llm-eval
├── README.md
├── __init__.py
├── benchmark_code
├── config #所有的自定义样例config可以在此文件夹下找到
├── docs #自动生成的API文档，使用sphinx实现
├── config.yaml #这是一个评估开源测试集+业务测试集+金融能力测试集的配置文件，仅供参考，自己需要维护对应的配置文件
├── eval.py
├── inference #所有的推理引擎相关的代码都在此文件夹下
├── post_eval.py #推理完成后的评估启动代码
├── reqirements.txt
├── run.py #整个运行流程的启动入口
├── run.sh #评估启动的执行文件，仅供参考，需要自己维护自己的run.sh文件
├── run_judge.py
├── scripts #一些参考的run.sh脚本
├── tools #一些常用的方法进行封装，如http requests
├── src
├── statistic.py #统计最终评估结果和上传的脚本
├── testsets #所有的非业务测试集都在此文件夹下
└── utils #所有的打分函数都在此文件夹下
```

### Quick Start 评估本地模型（使用HuggingFace model.generate()函数）
<p>评估新模型的时候无法使用vllm推理时，可以设置backend参数为hf使用model.generate()进行评估</p>

```sh
export MODEL_PATH=/mnt/data/llm/models/chat/Qwen2.5-0.5B #待评测模型需要将路径放在环境变量中
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template # Hithink llama3 llama2 none qwen chat_template 推荐使用chat_template

#先将模型启动为服务
python inference/predict_multi_gpu.py --model ${MODEL_PATH} --server_port ${REMOTE_MODEL_PORT} --prompt ${PROMPT_TYPE} --preprocess preprocess --run_forever --max_new_tokens 4096 --tensor_parallel ${TENSOR_PARALLEL} --backend hf & 

#传入config文件路径进行评测
python run.py --config config.yaml --model_name ${MODEL_NAME}
```

### Quick Start 评估本地模型，使用大模型对评估结果打分

```sh
export MODEL_PATH=/mnt/data/llm/models/chat/Qwen2.5-0.5B #待评测模型需要将路径放在环境变量中
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template # Hithink llama3 llama2 none qwen chat_template 推荐使用chat_template

#先将模型启动为服务
python inference/predict_multi_gpu.py --model ${MODEL_PATH} --server_port ${REMOTE_MODEL_PORT} --prompt ${PROMPT_TYPE} --preprocess preprocess --run_forever --max_new_tokens 4096 --tensor_parallel ${TENSOR_PARALLEL} --low_vram & 

# 启动裁判员模型
export JUDGE_MODEL_PATH=/mnt/data/llm/models/base/Qwen2.5-7B
export JUDGE_TENSOR_PARALLEL=1
export JUDGE_MODEL_PORT=16667
python inference/predict_multi_gpu.py --model ${JUDGE_MODEL_PATH} --server_port ${JUDGE_MODEL_PORT} --prompt chat_template --preprocess preprocess  --run_forever --manual_start --max_new_tokens 4096 --tensor_parallel ${JUDGE_TENSOR_PARALLEL} --low_vram &

# 传入config文件路径进行评测
python run.py --config "config_all_yewu.yaml" --model_name ${MODEL_NAME}
```
注意在启动裁判员模型时增加了`--manual_start`入参，因为裁判员模型需要等待模型推理完成后再启动（由`run.py`中的`maybe_start_judge_model`方法自动触发）。

## ✒️Citation

comming soon

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* We would like to thank [Weijie Zhang](https://github.com/zhangwj618) for his contribution to the development of the inference engine.
* This work leverages [vLLM](https://github.com/vllm-project/vllm) as the backend model server for evaluation purposes.
