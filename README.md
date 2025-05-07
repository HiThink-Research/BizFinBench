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
  📖<a href="https://arxiv.org/abs/2411.03314">Paper</a> |🏠<a href="https://hithink-research.github.io/MME-Finance/">Homepage</a></h3>|🤗<a href="https://huggingface.co/datasets/hithink-ai/MME-Finance">Huggingface</a></h3>
<div align="center"></div>
<p align="center">
  <p>
In recent years, multimodal benchmarks for general domains have guided the rapid development of multimodal models on general tasks. However, the financial field has its peculiarities. It features unique graphical images (e.g., candlestick charts, technical indicator charts) and possesses a wealth of specialized financial knowledge (e.g., futures, turnover rate).
    
Benchmarks from general fields often fail to measure the performance of multimodal models in the financial domain, and thus cannot effectively guide the rapid development of large financial models. To promote the development of large financial multimodal models, we propose <strong>MME-Finance</strong>, an bilingual open-ended and practical usage-oriented Visual Question Answering (VQA) benchmark. 

## 📢 News 
- 🚀 [07/05/2025] We released <strong>BizFinBench</strong> benchmark, xxxx.

## 💡 Highlights
- 🔥 <strong>xxxx:</strong> xxxx.


## 🛠️ Usage

<!-- ### Judgement -->
We have integrated MMfin into the VLMEvalKit framework. For the environment configuration and the use of API, please refer to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
Regarding the data, first of all, you should download the `MMfin.tsv` and `MMfin_CN.tsv` files, as well as the relevant financial images. The folder structure is shown as follows:
```
├─ datasets
    ├─ images
        ├─ MMfin
            ...
        ├─ MMfin_CN
            ...
    │ MMfin.tsv
    │ MMfin_CN.tsv
```
The following is the process of inference and evaluation (Qwen2-VL-2B-Instruct as an example):
```
export LMUData="The path of the datasets"
python run.py --data MMfin --model Qwen2-VL-2B-Instruct --verbose
python run.py --data MMfin_CN --model Qwen2-VL-2B-Instruct --verbose
```

## ✒️Citation
```
@article{gan2024woodpecker,
  title={MME-Finance: A Multimodal Finance Benchmark for Expert-level Understanding and Reasoning},
  author={Gan, Ziliang and Lu, Yu and Zang, Dong and Li, Haohan and Liu, Che and Liu, Jian and Liu, Ji and Wu, Haipang and Fu, Chaoyou and Xu, Zenglin and Zhang, Rongjunchen and Dai, Yong},
  journal={arXiv preprint arXiv:2411.03314},
  year={2024}
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* <a href="https://github.com/open-compass/VLMEvalKit"><u>VLMEvalKit</u></a>: the codebase we built upon. 
