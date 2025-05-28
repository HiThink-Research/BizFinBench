import re
import json
from transformers import AutoProcessor, PreTrainedTokenizer


PROMPT = {
    'none': '{}',
    'Hithink': 'Human: {} \n\nAssistant:',
    "qwen":"<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    'llama3':'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n',
    'llama2':'User:\n\n{}Assistant:\n\n',
    'chat_template': None,
    'deepseek':'<｜begin▁of▁sentence｜>User: {s_in}\nAssistant: '
}


MESSAGESPROMPT = {
    'none': '{system}{s_in}',
    'Hithink': '<<SYS>>\n{system}\n<</SYS>>\n\n{s_in}',
    'qwen': '<|im_start|>system\n{system}<|im_end|>\n\n{s_in}',
    'llama3':'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>{s_in}',
    'llama2':'{system}\n\n{s_in}',
    'deepseek':'<｜begin▁of▁sentence｜>{system}\n\n{s_in}',
}
HumanPROMPT = {
    'none': '{human_in}',
    'Hithink': 'Human:{human_in}',
    'qwen': '<|im_start|>user\n{human_in}\n<|im_end|>',
    'llama3':'<|start_header_id|>user<|end_header_id|>\n\n{human_in}<|eot_id|>',
    'llama2':'User:\n\n{human_in}',
    'deepseek':'User: {human_in}',
}
AssistantPROMPT = {
    'none': '{ai_in}',
    'Hithink': 'Assistant:{ai_in}',
    'qwen': '<|im_start|>assistant\n{ai_in}\n<|im_end|>',
    'llama3':'<|start_header_id|>assistant<|end_header_id|>\n\n{ai_in}<|eot_id|>',
    'llama2':'Assistant:\n\n{ai_in}',
    'deepseek':'Assistant: {ai_in}<｜end▁of▁sentence｜>',
}
ContinuePROMPT = {
    'none': '',
    'Hithink': 'Assistant:',
    'qwen': '<|im_start|>assistant\n',
    'llama3': '<|start_header_id|>assistant<|end_header_id|>\n\n',
    'llama2':'Assistant:\n\n',
    'deepseek':'Assistant: ',
}


def get_prompt(tokenizer, q: str, prompt_type: str):
    if PROMPT[prompt_type] is not None:
        q = re.sub(r'^Human[:：]\s*', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s*Assistant[:：]\s*$', '', q, flags=re.IGNORECASE)
        # q = re.sub(r'\n*[答A][:：] *$', '', q)
        return PROMPT[prompt_type].format(q)
    else:
        s_in = [{'role': 'user', 'content': q}]
        prompt = tokenizer.apply_chat_template(s_in, tokenize=False, add_generation_prompt=True)
        return prompt


def process_messages(tokenizer, sample, prompt_type, is_multi_modal=False) -> str:
    """
    将messages格式的数据拼接成完整的输入
    """
    messages = sample['messages']
    text_messages = list()
    for turn in messages:
        if isinstance(turn['content'], list):
            text = '\n'.join([x['text'] for x in turn['content'] if x['type'] == 'text'])
        elif isinstance(turn['content'], str):
            text = turn['content']
        else:
            raise ValueError(f'"content" must be either list or str: {turn["content"]}')
        text_messages.append({'role': turn['role'], 'content': text})
        if t := turn.get('tool_calls'):
            text_messages[-1]['tool_calls'] = t

    if text_messages[-1]['role'] == 'assistant':
        # 若最后是答案则去掉
        text_messages = text_messages[:-1]
        messages = messages[:-1]

    # 替换最后一轮
    if 'model_prompt_tmpl' in sample and sample['model_prompt_tmpl'] and text_messages[-1]['role'] == 'user':
        model_prompt_tmpl = sample['model_prompt_tmpl']
        model_prompt_placeholder = sample['model_prompt_placeholder']
        for placeholder_item in model_prompt_placeholder:
            key = placeholder_item['key']
            value = placeholder_item['value']
            model_prompt_tmpl = model_prompt_tmpl.replace('{{' + '{}'.format(key) + '}}', value)
        text_messages = text_messages[:-1] + [{'role': 'user', 'content': model_prompt_tmpl}]
        # TODO: messages的最后一轮是否需要替换？messages在输入多模态数据时会用到，暂不替换

    if prompt_type != 'chat_template' and prompt_type in PROMPT.keys():
        assert not is_multi_modal, '输入包含多模态数据，只支持prompt_type="chat_template"！'
        if text_messages[0]['role'] == 'system':
            system_str = text_messages[0]['content']
            text_messages = text_messages[1:]
        else:
            system_str = ""

        if system_str:
            final_template = MESSAGESPROMPT[prompt_type].replace("{system}", system_str.strip())
        else:
            final_template = "{s_in}"

        s_in_list = []
        for input_ele in text_messages:
            if input_ele["role"].lower() == "user":
                s_in_list.append(HumanPROMPT[prompt_type].replace("{human_in}", input_ele["content"]))
            elif input_ele["role"].lower() == "assistant":
                s_in_list.append(AssistantPROMPT[prompt_type].replace("{ai_in}", input_ele["content"]))
        s_in = "\n".join(s_in_list) + "\n" + ContinuePROMPT[prompt_type]

        final_prompt = final_template.replace("{s_in}", s_in)
    else:
        tools = sample.get('tools')
        if isinstance(tools, str) and tools:
            tools = json.loads(tools)
        final_prompt = tokenizer.apply_chat_template(
            messages if is_multi_modal else text_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True
        )
        if isinstance(final_prompt, list):  # e.g. Qwen2.5-Omni-7B
            final_prompt = final_prompt[0]

    return final_prompt


def check_for_multi_modal(messages):
    """
    判断在messages中是否有multi-modal数据
    如果有的话进行处理并返回 True, List[Image path], List[Audio path]
    """
    # 处理所有 'user' 消息，去除其中的jpg和url 
    user_message = [msg for msg in messages['messages'] if msg['role'] == 'user']

    if not user_message:
        return False, None, None

    # 检查是否有 type 不为 'text' 的内容
    images = []
    audios = []
    for msg in user_message:
        # 检查 content 是否存在且为列表，否则的话，这是一条纯文本格式的消息
        if 'content' not in msg or not isinstance(msg['content'], list):
            continue
        for item in msg['content']:
            if item.get('type') == 'image_url':
                images.append(item['image_url']['url'])
            if item.get('type') == 'audio_url':
                audios.append(item['audio_url']['url'])
    is_multi_modal = bool(images or audios)
    return is_multi_modal, images, audios


def convert_sample_to_input_ids(s: dict, prompt_type: str, tokenizer: PreTrainedTokenizer):
    if 'token_ids' in s:
        return s['token_ids']

    if 'messages' in s:  # 多轮对话格式
        is_multi_modal, images, audios = check_for_multi_modal(s)
        s_in = process_messages(tokenizer, s, prompt_type, is_multi_modal)
        if is_multi_modal:
            inputs = {
                "prompt": s_in,
                "multi_modal_data": {}
            }
            if images:
                inputs["multi_modal_data"]["image"] = images
            if audios:
                inputs["multi_modal_data"]["audio"] = audios
            return inputs

    elif 'target' in s:  # 输入格式：input/source + target
        s_in = s['source'] if 'source' in s else s['input']
    else:  # 输入格式：instruction + input + output
        s_in = s.get('instruction', '')
        if not s_in:
            s_in = s.get('system', '')
        if s.get('input'):
            if s_in:
                s_in += '\n'
            s_in += s['input']

    if prompt_type in PROMPT.keys():
        if "messages" in s or "model_prompt_tmpl" in s:
            # messages 格式
            prompt = s_in
        else:
            if not isinstance(s_in, str):
                s_in = str(s_in)
            prompt = get_prompt(tokenizer, s_in, prompt_type)

        #处理当输入是多模态Processor的时候tokenizer异常
        if "Processor" in type(tokenizer).__name__:
            tokenizer = tokenizer.tokenizer
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    else: 
        raise ValueError("prompt type must be in: {}".format(PROMPT.keys()))

    return input_ids


if __name__ == '__main__':
    tokenizer = AutoProcessor.from_pretrained('/mnt/data/model/chat/multimodal-models/Qwen/Qwen2.5-VL-3B-Instruct/')
    s= {"query":"Please reason step by step, and put your final answer within \\boxed{}.\n Question:Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.","messages":[{"role":"user","content":[{"text":"Please reason step by step, and put your final answer within \\boxed{}.\n Question:Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.","type":"text"}]}],"choices":[{"index":0,"message":{"role":"assistant","content":[{"text":"73","type":"text"}]}}],"model_prompt_tmpl":"{{prompt}}","model_prompt_placeholder":[{"key":"prompt","value":"Please reason step by step, and put your final answer within \\boxed{}.\n Question:Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things."}]}
    s2 = {
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": "Q1",
                    "type": "text"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "response 1",
                    "type": "text"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "q2",
                    "type": "text"
                }
            ]
        }
    ],
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": "73",
                        "type": "text"
                    }
                ]
            }
        }
    ],
    "model_prompt_tmpl": "",
    "model_prompt_placeholder": []
}
    print('unit test 1')
    input_ids = convert_sample_to_input_ids(s, 'chat_template', tokenizer)
    print('-' * 50)
    print(input_ids)
    print('-' * 50)
    print(tokenizer.decode(input_ids))
    print('-' * 50)

    print('unit test 2')
    input_ids = convert_sample_to_input_ids(s2, 'chat_template', tokenizer)
    print('-' * 50)
    print(input_ids)
    print('-' * 50)
    print(tokenizer.decode(input_ids))
    print('-' * 50)
