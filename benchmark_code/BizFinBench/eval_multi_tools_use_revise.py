import json
import re
import os
import numpy as np
from tqdm import tqdm
from tools.ExternalApi import GPT4OClient
import copy
import asyncio
import concurrent.futures
GENERATE_APIS_DATA_TEMPLATE ='''请你模拟api接口，下面是你需要模拟的api接口说明，用户会向你发送api调用请求，请你合理生成返回的内容：
[
  {
    "name": "Calculator",
    "description": "用于执行复杂的数学计算，如多位数乘除。简单的计算如加减法或比较大小不需要使用此工具。",
    "required_parameters": [
      {
        "name": "expression",
        "type": "string",
        "description": "需要计算的数学表达式（例如：'123 * 456 / 789'）。"
      }
    ],
    "optional_parameters": [],
    "return_info": [
      {
        "name": "result",
        "type": "number",
        "description": "计算结果。"
      },
      {
        "name": "error",
        "type": "string",
        "description": "如果表达式无效，返回错误信息。"
      }
    ]
  },
  {
    "name": "Clarify",
    "description": "当问题不明确或违背常识时使用此工具。可以向用户询问更多信息或提供二次确认。",
    "required_parameters": [
      {
        "name": "message",
        "type": "string",
        "description": "向用户显示的澄清或确认信息。"
      }
    ],
    "optional_parameters": [],
    "return_info": [
      {
        "name": "response",
        "type": "string",
        "description": "用户对澄清或确认的回复。"
      }
    ]
  },
  {
    "name": "FinWiki",
    "description": "金融百科工具，用于查询金融知识、投资方法、金融术语、人物、机构、行业概念等百科词条相关数据。",
    "required_parameters": [
      {
        "name": "query",
        "type": "string",
        "description": "需要查询的金融术语或概念（例如：'ROE', 'Black-Scholes模型'）。"
      }
    ],
    "optional_parameters": [],
    "return_info": [
      {
        "name": "definition",
        "type": "string",
        "description": "术语的定义或解释。"
      },
      {
        "name": "related_terms",
        "type": "array",
        "description": "相关的金融术语或概念列表。"
      }
    ]
  },
  {
    "name": "Search",
    "description": "搜索工具，用于基于自然语言短语或关键词（最多5个关键词）检索非结构化数据。",
    "required_parameters": [
      {
        "name": "keywords",
        "type": "array",
        "description": "需要搜索的关键词或短语列表（例如：['2023年股市趋势']）。"
      }
    ],
    "optional_parameters": [],
    "return_info": [
      {
        "name": "results",
        "type": "array",
        "description": "搜索结果列表，包含标题、URL和摘要。"
      }
    ]
  },
  {
  "name": "FinQuery",
  "description": "金融数据查询工具，用于获取标的相关的金融数据，如宏观数据、财务数据、行情数据、交易数据、个人账户数据、自选股等。支持股票、美股、港股、基金、指数、宏观、可转债、期货等。",
  "required_parameters": [
    {
      "name": "indicators",
      "type": "array",
      "description": "需要查询的金融指标列表（例如：['AAPL营收', '标普500市盈率']）。注意：指标中不应包含时间信息，时间范围请通过`time_range`参数指定。"
    }
  ],
  "optional_parameters": [
    {
      "name": "company_identifier",
      "type": "string",
      "description": "公司名称或股票代码（例如：'Apple Inc.' 或 'AAPL'）。在查询指标时，如果涉及公司名称或股票代码，则必须提供公司名称或股票代码。"
    },
    {
      "name": "time_range",
      "type": "string",
      "description": "查询的时间范围，支持多种格式：
- 单个年份：'2022'
- 年份范围：'2022-2023'
- 单个季度：'2022Q1'
- 季度范围：'2022Q1-2023Q4'
- 单个月份：'2022-01'
- 月份范围：'2022-01-2023-12'
- 单个日期：'2022-01-01'
- 日期范围：'2022-01-01-2023-12-31'
- 相对时间：'last 5 years', 'last 3 months', 'last 7 days'
- 默认时间范围：'last 1 year'（如果未提供该参数，则默认查询最近一年的数据）。对于高频指标（如行情数据、交易数据），建议使用 'last 7 days' 或 'last 1 month'。"
    }
  ],
  "return_info": [
    {
      "name": "data",
      "type": "object",
      "description": "请求的金融数据，以结构化格式返回。"
    }
  ]
  },
  {
    "name": "StockNews",
    "description": "个股新闻工具，用于获取股票、指数、概念、大宗商品等标的的最新资讯。",
    "required_parameters": [
      {
        "name": "symbol",
        "type": "string",
        "description": "需要查询的股票代码、指数或商品（例如：'AAPL', '标普500'）。"
      }
    ],
    "optional_parameters": [
      {
        "name": "limit",
        "type": "number",
        "description": "返回的新闻条目数量上限（默认：5）。"
      },
      {
        "name": "time_range",
        "type": "string",
        "description": "查询的时间范围，支持多种格式：
- 单个日期：'2023-10-01'
- 日期范围：'2023-10-01-2023-10-31'
- 相对时间：'last 7 days', 'last 1 month', 'last 1 year'
- 默认时间范围：'last 7 days'（如果未提供该参数，则默认查询最近7天的新闻）。"
      }
    ],
    "return_info": [
      {
        "name": "news",
        "type": "array",
        "description": "新闻条目列表，包含标题、URL和发布日期。"
      }
    ]
  },
  {
    "name": "CalendarQuery",
    "description": "财经日历查询工具，用于判断指定日期是否为交易日。",
    "required_parameters": [
      {
        "name": "date",
        "type": "string",
        "description": "需要查询的日期（例如：'2023-10-01'）。"
      }
    ],
    "optional_parameters": [],
    "return_info": [
      {
        "name": "is_trading_day",
        "type": "boolean",
        "description": "如果是交易日，返回`true`，否则返回`false`。"
      }
    ]
  },
  {
    "name": "EnterpriseQuery",
    "description": "企业快查工具，用于获取围绕企业（非上市公司）的客观的结构性数据，包括但不限于基本信息、股权信息、经营信息等。",
    "required_parameters": [
      {
        "name": "company_name",
        "type": "string",
        "description": "需要查询的企业名称。"
      }
    ],
    "optional_parameters": [
      {
        "name": "info_type",
        "type": "string",
        "description": "需要查询的信息类型（例如：'基本信息', '股权信息', '经营信息'）。"
      },
      {
        "name": "time_range",
        "type": "string",
        "description": "查询的时间范围，支持多种格式：
- 单个年份：'2022'
- 年份范围：'2022-2023'
- 单个季度：'2022Q1'
- 季度范围：'2022Q1-2023Q4'
- 单个月份：'2022-01'
- 月份范围：'2022-01-2023-12'
- 单个日期：'2022-01-01'
- 日期范围：'2022-01-01-2023-12-31'
- 相对时间：'last 5 years', 'last 3 months', 'last 7 days'
- 默认时间范围：'last 1 year'（如果未提供该参数，则默认查询最近一年的数据）。"
      }
    ],
    "return_info": [
      {
        "name": "company_data",
        "type": "object",
        "description": "请求的企业信息，以结构化格式返回。"
      }
    ]
  },
  {
    "name": "NoticeSearch",
    "description": "公告搜索工具，用于获取个股所有公告或某个具体公告。",
    "required_parameters": [
      {
        "name": "symbol",
        "type": "string",
        "description": "需要查询的股票代码（例如：'AAPL'）。"
      }
    ],
    "optional_parameters": [
      {
        "name": "notice_id",
        "type": "string",
        "description": "需要查询的具体公告ID。"
      },
      {
        "name": "time_range",
        "type": "string",
        "description": "查询的时间范围，支持多种格式：
- 单个日期：'2023-10-01'
- 日期范围：'2023-10-01-2023-10-31'
- 相对时间：'last 7 days', 'last 1 month', 'last 1 year'
- 默认时间范围：'last 1 month'（如果未提供该参数，则默认查询最近1个月的公告）。"
      }
    ],
    "return_info": [
      {
        "name": "notices",
        "type": "array",
        "description": "公告列表，包含标题、日期和内容。"
      }
    ]
  },
  {
    "name": "ReportQuery",
    "description": "研报查询工具，用于搜索个股或行业研报的观点总结信息。",
    "required_parameters": [
      {
        "name": "query",
        "type": "string",
        "description": "需要查询的股票代码或行业（例如：'AAPL', '半导体行业'）。"
      }
    ],
    "optional_parameters": [
      {
        "name": "limit",
        "type": "number",
        "description": "返回的研报数量上限（默认：5）。"
      },
      {
        "name": "time_range",
        "type": "string",
        "description": "查询的时间范围，支持多种格式：
- 单个年份：'2022'
- 年份范围：'2022-2023'
- 单个季度：'2022Q1'
- 季度范围：'2022Q1-2023Q4'
- 单个月份：'2022-01'
- 月份范围：'2022-01-2023-12'
- 单个日期：'2022-01-01'
- 日期范围：'2022-01-01-2023-12-31'
- 相对时间：'last 5 years', 'last 3 months', 'last 7 days'
- 默认时间范围：'last 1 year'（如果未提供该参数，则默认查询最近一年的研报）。"
      }
    ],
    "return_info": [
      {
        "name": "reports",
        "type": "array",
        "description": "研报列表，包含摘要、作者和发布日期。"
      }
    ]
  },
  {
    "name": "CommunityReviews",
    "description": "大众点评工具，用于查询常见财经站点中近期有一定质量的用户讨论，侧重于查找近期A股公司传闻、点评等内容。",
    "required_parameters": [
      {
        "name": "query",
        "type": "string",
        "description": "需要查询的股票简称、热点事件或特殊问法（例如：'AAPL', '热点事件', '小作文'）。"
      }
    ],
    "optional_parameters": [
      {
        "name": "limit",
        "type": "number",
        "description": "返回的讨论数量上限（默认：5）。"
      },
      {
        "name": "time_range",
        "type": "string",
        "description": "查询的时间范围，支持多种格式：
- 单个日期：'2023-10-01'
- 日期范围：'2023-10-01-2023-10-31'
- 相对时间：'last 7 days', 'last 1 month', 'last 1 year'
- 默认时间范围：'last 7 days'（如果未提供该参数，则默认查询最近7天的讨论）。"
      }
    ],
    "return_info": [
      {
        "name": "discussions",
        "type": "array",
        "description": "讨论列表，包含标题、来源和摘要。"
      }
    ]
  },
  {
  "name": "PublicCompanyInfo",
  "description": "上市公司信息查询工具，用于获取上市公司的基本信息、财务数据、股东信息、行业分类、公司规模、主营业务等。支持多语言查询和多种信息类型的筛选。",
  "required_parameters": [
    {
      "name": "stock_code",
      "type": "string",
      "description": "需要查询的股票代码（例如：'603194' 或 'AAPL'）。"
    }
  ],
  "optional_parameters": [
    {
      "name": "info_type",
      "type": "string",
      "description": "需要查询的信息类型（例如：'基本信息', '财务数据', '股东信息', '行业分类', '公司规模', '主营业务'）。默认为返回所有信息。"
    },
    {
      "name": "language",
      "type": "string",
      "description": "返回信息的语言（例如：'zh' 表示中文，'en' 表示英文）。默认为 'zh'。"
    },
    {
      "name": "time_range",
      "type": "string",
      "description": "查询的时间范围，支持多种格式：
- 单个年份：'2022'
- 年份范围：'2022-2023'
- 单个季度：'2022Q1'
- 季度范围：'2022Q1-2023Q4'
- 单个月份：'2022-01'
- 月份范围：'2022-01-2023-12'
- 单个日期：'2022-01-01'
- 日期范围：'2022-01-01-2023-12-31'
- 相对时间：'last 5 years', 'last 3 months', 'last 7 days'
- 默认时间范围：'last 1 year'（如果未提供该参数，则默认查询最近一年的数据）。"
    }
  ],
  "return_info": [
    {
      "name": "company_info",
      "type": "object",
      "description": "请求的上市公司信息，以结构化格式返回。包含以下字段：",
      "fields": [
        {
          "name": "basic_info",
          "type": "object",
          "description": "公司基本信息，包括公司名称、股票代码、上市交易所、成立日期、公司地址等。"
        },
        {
          "name": "financial_data",
          "type": "object",
          "description": "公司财务数据，包括收入、利润、负债、现金流等。支持按时间范围筛选。"
        },
        {
          "name": "shareholder_info",
          "type": "object",
          "description": "股东信息，包括主要股东名单、持股比例、股权结构等。"
        },
        {
          "name": "industry_classification",
          "type": "string",
          "description": "公司所属行业分类（例如：'科技', '金融', '制造业'）。"
        },
        {
          "name": "company_size",
          "type": "string",
          "description": "公司规模（例如：'大型', '中型', '小型'）。"
        },
        {
          "name": "main_business",
          "type": "string",
          "description": "公司主营业务描述。"
        },
        {
          "name": "historical_data",
          "type": "array",
          "description": "历史数据，包含公司过去几年的财务和运营数据。"
        }
      ]
    },
    {
      "name": "error",
      "type": "string",
      "description": "如果查询失败，返回错误信息。"
    }
  ]
}
]

注意当前日期：2025年1月19日，星期日。

API调用请求:
{api_request}
'''
GENERATE_JUDGE_APIS_PROMPT  = '''请你判断两种apis返回的结果，哪一种更有助于解决用户的问题。
用户的问题和历史对话信息：{question}
第一种：
{first_apis_result}

第二种：
{second_apis_result}

注意当前日期：2025年1月19日，星期日。

如果两种返回的结果对于解决用户的问题效果基本一致，则输出True。
如果第二种明显优于第一种，则输出True。
如果第一种明显优于第二种，则输出False。
你无需判断数据是否异常。
请按照以下格式提交你的评估结果：
{'cot':<填写你的判断依据>,'result':<填写你的判断结果>}'''
# 上传数据格式模板
UPLOAD_PASS_TEMPLATE = {
    "query": "",
    "messages": [{"role": "user", "content": [{"text": "", "type": "text"}]}],
    "choices": [{"index": 0, "message": {"role": "assistant", "content": [{"text": "", "type": "text"}]}}],
    "model_prompt_tmpl": "",
    "model_prompt_placeholder": []
}
def extract_actions(first_round_actions):
    """提取并格式化动作列表"""
    formatted_actions = []
    for action in first_round_actions:
        formatted_actions.append({action["Action"]: action.get("ActionInput", action.get("Action Input", ""))})
    return formatted_actions

def get_prompt(prompt_template: str, params: dict):
    """根据模板和参数生成提示"""
    if isinstance(params, dict):
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            prompt_template = prompt_template.replace(placeholder, str(value))
    return prompt_template

def upload_pass_format(query: str, answer: str, model_prompt_tmpl=None, is_return_dict=True) -> str | dict:
    """生成上传数据格式"""
    template = copy.deepcopy(UPLOAD_PASS_TEMPLATE)
    template["query"] = query
    template["messages"][0]["content"][0]["text"] = query
    if model_prompt_tmpl:
        template['model_prompt_tmpl'] = model_prompt_tmpl
    template["choices"][0]["message"]["content"][0]["text"] = answer
    return template if is_return_dict else json.dumps(template, ensure_ascii=False)

def extract_json_from_text(text):
    """
    从文本中提取包含在其他内容中的 JSON 内容。
    
    参数:
        text (str): 包含 JSON 内容的文本，可能在 markdown 代码块中或以其他形式存在。
        
    返回:
        dict 或 None: 如果找到 JSON 对象则返回解析后的对象，否则返回 None。
    """
    if '</think>' in text:
        text = text.split('</think>', 1)[-1].strip()  # 只保留 `</think>` 之后的内容
    patterns = [
        r'```json\s*([\s\S]+?)\s*```',      # 匹配 ```json ... ``` 格式
        r'```\s*([\s\S]+?)\s*```',           # 匹配 ``` ... ``` 格式
        r'json\s*(\{[\s\S]+\})',             # 匹配前缀 json 后跟 JSON 对象
        r'(\{[\s\S]+\})'                    # 匹配独立的 JSON 对象
    ]
    
    # 遍历每个模式，尝试匹配
    for pat in patterns:
        regex = re.compile(pat, re.DOTALL)  # 使用 DOTALL 标志，允许匹配换行符
        match = regex.search(text)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    return None
async def run_gpt4o_tasks(preds_apis, answers_apis):
    gpt4o = GPT4OClient()
    pred_apis_results = await gpt4o.texts2texts(preds_apis, 0)
    answers_apis_results = await gpt4o.texts2texts(answers_apis, 0)
    return pred_apis_results, answers_apis_results
def run_async_in_thread(preds_apis, answers_apis):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_gpt4o_tasks(preds_apis, answers_apis))
    finally:
        loop.close()  # 确保事件循环关闭
    return result
def data_preprocess(input_path, input_file, save_dir='step1'):
    """数据预处理：提取行为动作、模拟API返回结果并生成判断指令"""
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)
    current_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, current_file_name)

    
    data = [json.loads(line) for line in open(os.path.join(input_path, input_file), encoding='utf8')]

    out = []
    for d in data:
        result = d['predict_result']
        try:
            result_dict = extract_json_from_text(result)
            predict_result = extract_actions(result_dict['Actions'])
            d['predict_result'] = predict_result
        except Exception as e:
            d['predict_result'] = {}
        out.append(d)

    
    preds_apis = []
    answers_apis = []

    
    for o in out:
        pred_apis = get_prompt(GENERATE_APIS_DATA_TEMPLATE, {'api_request': json.dumps(o['predict_result'], ensure_ascii=False)})
        answer_apis = get_prompt(GENERATE_APIS_DATA_TEMPLATE, {'api_request': o["choices"][0]["message"]["content"][0]["text"]})
        preds_apis.append(upload_pass_format(pred_apis, ''))
        answers_apis.append(upload_pass_format(answer_apis, ''))
    print('开始模拟apis的响应结果')
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(run_async_in_thread, preds_apis, answers_apis)
        result = future.result() 
    preds_apis_results, answers_apis_results = result
    judge_instructions = []
    for i, (o, pred, ans) in enumerate(zip(out, preds_apis_results, answers_apis_results)):
        if o['predict_result'] == json.loads(o["choices"][0]["message"]["content"][0]["text"]):
            pred = ans
        ins = o['query']
        start = ins.find('下面是用户的输入和历史对话信息：') + len('下面是用户的输入和历史对话信息：')
        end = ins.find('请按照输出格式要求，输出你的Thought和Actions!')
        question = ins[start:end]
        judge_ins = get_prompt(GENERATE_JUDGE_APIS_PROMPT, {'question': question, 'first_apis_result': ans, 'second_apis_result': pred})
        judge_instruction = upload_pass_format(judge_ins,o["choices"][0]["message"]["content"][0]["text"])
        judge_instruction['ans_apis_result'] = ans
        judge_instruction['pred_answer'] = json.dumps(o['predict_result'], ensure_ascii=False)
        judge_instruction['pred_apis_result'] = pred
        judge_instructions.append(judge_instruction)

    
    with open(out_file, 'w', encoding='utf8') as f:
        for ins in judge_instructions:
            f.write(json.dumps(ins, ensure_ascii=False) + '\n')

def evaluation(input_path, **kwargs):
    """评估API返回结果的有效性"""
    data = [json.loads(line) for line in open(input_path, encoding='utf8')]
    count = 0
    out = []
    for d in data:
        if 'true' in d['predict_result'] or 'True' in d['predict_result']:
            count += 1
            d['eval_result'] = True
        else:
            d['eval_result'] = False
        out.append(d)

    
    with open(input_path, 'w', encoding='utf8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    count = count / len(data) if len(data) != 0 else 0
    return {'acc':count}