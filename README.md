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
  📖<a href="https://arxiv.org/abs/25xx.xxxxx">Paper (coming soom)</a> |🏠<a href="https://hithink-research.github.io/BizFinBench/">Homepage (coming soom)</a></h3>|🤗<a href="https://huggingface.co/datasets/HiThink-Research/BizFinBench">Huggingface</a></h3>
<div align="center"></div>
<p align="center">
  <p>
In recent years, multimodal benchmarks for general domains have guided the rapid development of multimodal models on general tasks. However, the financial field has its peculiarities. It features unique graphical images (e.g., candlestick charts, technical indicator charts) and possesses a wealth of specialized financial knowledge (e.g., futures, turnover rate).

Large language models excel across general tasks, yet judging their reliability in logic‑heavy, precision‑critical domains such as finance, law and healthcare is still difficult. To address this challenge, we propose **BizFinBench**, the first benchmark grounded in real-world financial applications. BizFinBench consists of **6,781** well-annotated queries in Chinese, covering five dimensions: numerical calculation, reasoning, information extraction, prediction recognition and knowledge‐based question answering, which are mapped to nine fine-grained categories.

## 📢 News 
- 🚀 [16/05/2025] We released <strong>BizFinBench</strong> benchmark (V1), the first benchmark grounded in real-world financial applications.

## 💡 Highlights
- 🔥  **Benchmark:** We propose **BizFinBench**, the first evaluation benchmark in the financial domain that integrates business-oriented tasks, covering 5 dimensions and 9 categories. It is designed to assess the capacity of LLMs in real-world financial scenarios.
- 🔥  **Judge model:** We design a novel evaluation method, i.e., **Iterajudge**, which enhances the capability of LLMs as a judge by refining their decision boundaries in specific financial evaluation tasks.
- 🔥  **key insights:** We conduct a comprehensive evaluation with **25 LLMs** based on BizFinBench, uncovering key insights into their strengths and limitations in financial applications.

## 📕 Data Distrubution
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

## 🛠️ Usage

### Contents

```
llm-eval
├── README.md
├── benchmark_code
├── config # All custom sample configs can be found in this folder
├── eval.py
├── inference # All inference-engine-related code is in this folder
├── post_eval.py # Evaluation launcher after inference is finished
├── reqirements.txt
├── run.py # Entry point for the entire evaluation workflow
├── run.sh # Sample execution script for launching an evaluation; maintain your own run.sh as needed
├── scripts # Reference run.sh scripts
├── statistic.py # Aggregates final evaluation statistics
└── utils
```

### Quick Start – Evaluate a Local Model

```sh
export MODEL_PATH=/mnt/data/llm/models/chat/Qwen2.5-0.5B   # Path to the model to be evaluated
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template   # Hithink llama3 llama2 none qwen chat_template; chat_template is recommended

# First start the model as a service
python inference/predict_multi_gpu.py \
    --model ${MODEL_PATH} \
    --server_port ${REMOTE_MODEL_PORT} \
    --prompt ${PROMPT_TYPE} \
    --preprocess preprocess \
    --run_forever \
    --max_new_tokens 4096 \
    --tensor_parallel ${TENSOR_PARALLEL} & 

# Pass in the config file path to start evaluation
python run.py --config config.yaml --model_name ${MODEL_NAME}
```

### Quick Start – Evaluate a Local Model and Score with a Judge Model

```sh
export MODEL_PATH=/mnt/data/llm/models/chat/Qwen2.5-0.5B   # Path to the model to be evaluated
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template   # llama3 llama2 none qwen chat_template; chat_template is recommended

# First start the model as a service
python inference/predict_multi_gpu.py \
    --model ${MODEL_PATH} \
    --server_port ${REMOTE_MODEL_PORT} \
    --prompt ${PROMPT_TYPE} \
    --preprocess preprocess \
    --run_forever \
    --max_new_tokens 4096 \
    --tensor_parallel ${TENSOR_PARALLEL} \
    --low_vram & 

# Start the judge model
export JUDGE_MODEL_PATH=/mnt/data/llm/models/base/Qwen2.5-7B
export JUDGE_TENSOR_PARALLEL=1
export JUDGE_MODEL_PORT=16667
python inference/predict_multi_gpu.py \
    --model ${JUDGE_MODEL_PATH} \
    --server_port ${JUDGE_MODEL_PORT} \
    --prompt chat_template \
    --preprocess preprocess \
    --run_forever \
    --manual_start \
    --max_new_tokens 4096 \
    --tensor_parallel ${JUDGE_TENSOR_PARALLEL} \
    --low_vram &

# Pass in the config file path to start evaluation
python run.py --config "config_all_yewu.yaml" --model_name ${MODEL_NAME}
```

> **Note**: Add the `--manual_start` argument when launching the judge model, because the judge must wait until the main model finishes inference before starting (this is handled automatically by the `maybe_start_judge_model` function in `run.py`).


## ✒️Citation

comming soon

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* We would like to thank [Weijie Zhang](https://github.com/zhangwj618) for his contribution to the development of the inference engine.
* This work leverages [vLLM](https://github.com/vllm-project/vllm) as the backend model server for evaluation purposes.
