import json
import traceback
import aiohttp
import asyncio
from tqdm import tqdm
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import yaml
from openai import OpenAI
from loguru import logger
import copy
import sys,os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

logger.remove()
logger.add(sink=sys.stderr, level="INFO")

class GPTClient:
    def __init__(self, api_name,api_key,model_name, 
                    base_url=None,
                    timeout=600, semaphore_limit=5):
        """
        初始化通用参数

        Args:
            api_name (str): API name e.g., chatgpt
            api_key (str): API 密钥
            model_name (str): 使用的模型名称
            base_url (str): 基础 URL
            timeout (int): 请求超时时间，默认 60 秒
            semaphore_limit (int): 并发限制，默认 5
        """
        self.api_name = api_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore_limit = semaphore_limit
        self.session = None

    async def initialize(self):
        """初始化异步会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        if self.api_name == 'chatgpt':
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.client = OpenAI()
        return self

    async def close(self):
        """关闭异步会话"""
        if self.session:
            await self.session.close()
            self.session = None

    @staticmethod
    def _encode_image(image_path):
        """
        将图片编码为 Base64 格式

        Args:
            image_path (str): 图片路径

        Returns:
            str: Base64 编码的图片
        """
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def get_post_response(self, messages, chat_url, temperature=0, try_num=3, pbar=None):
        """
        获取大模型响应

        Args:
            messages (list): 消息列表
            temperature (float): 温度参数
            try_num (int): 最大重试次数
            pbar (tqdm): 进度条对象（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if self.session is None:
            await self.initialize()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }

        for attempt in range(try_num):
            try:
                async with self.session.post(chat_url, json=payload, headers=headers, timeout=self.timeout) as response:
                    response_data = await response.json()

                    if not response_data.get("success", True):
                        logger.error(f"Retry time: {attempt}, Error: {response_data}")
                        continue
                    # import pdb;pdb.set_trace()
                    choices = response_data['choices']
                    if choices is None:
                        logger.error("Openai error: {}".format(response_data))
                        continue
                    result = choices[0]['message']['content']
                    if pbar:
                        pbar.update(1)
                    return result
            except Exception as e:
                error_message = traceback.format_exc()
                logger.error(f"Retry time: {attempt}, Error: {error_message}, Openai response: {response_data}")
                # await self.refresh_authority()

        return {"error": "Max retries exceeded"}

    async def get_chatgpt_response(self, messages, temperature=0, try_num=3, pbar=None):
        """
        获取大模型响应

        Args:
            messages (list): 消息列表
            temperature (float): 温度参数
            try_num (int): 最大重试次数
            pbar (tqdm): 进度条对象（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if self.session is None:
            await self.initialize()

        for attempt in range(try_num):
            try:
                response_data = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )

                choices = response_data.choices
                if choices is None:
                    logger.error("Openai error: {}".format(response_data))
                    continue
                result = choices[0].message.content
                if pbar:
                    pbar.update(1)
                return result
            except Exception as e:
                error_message = traceback.format_exc()
                logger.error(f"Retry time: {attempt}, Error: {error_message}, Openai response: {response_data}")

        return {"error": "Max retries exceeded"}

    async def text2text(self, instruction, temperature=0, pbar=None, output_file=None):
        """
        文本生成文本

        Args:
            instruction (str or list): 输入文本或消息列表
            temperature (float): 温度参数
            pbar (tqdm): 进度条对象（可选）
            output_file (str): 输出文件路径（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if isinstance(instruction, str):
            messages = [
                {"role": "user", "content": instruction}
            ]
            instruction = {'messages':messages}
        else:
            messages = instruction['messages']
        predict_result = await self.get_post_response(messages,temperature=temperature,chat_url=self.base_url, try_num=2, pbar=pbar)
        # predict_result = await self.get_chatgpt_response(messages,temperature=temperature, try_num=2, pbar=pbar)

        if output_file:
            ins = instruction
            predict_result = predict_result
            self.save_file(output_file,ins,predict_result)
        
        return {'instruction':instruction,'predict_result':predict_result}

    async def texts2texts(self, instructions, temperature=0, output_file=None):
        """
        批量文本生成文本

        Args:
            instructions (list): 输入指令列表
            temperature (float): 温度参数
            output_file (str): 输出文件路径（可选）

        Returns:
            list: 接口返回的结果列表
        """
        await self.initialize()
        pbar = tqdm(total=len(instructions), desc="Processing Text-to-Text ({self.api_name})")

        semaphore = asyncio.Semaphore(self.semaphore_limit)
        # import pdb;pdb.set_trace()
        async def process_with_semaphore(instruction):
            async with semaphore:
                return await self.text2text(instruction, temperature=temperature, pbar=pbar, output_file=output_file)

        tasks = [process_with_semaphore(instruction) for instruction in instructions]
        results = await asyncio.gather(*tasks)
        
        pbar.close()
        return results

    @staticmethod
    def save_file(path, instruction, predict_result):
        """
        保存结果到文件

        Args:
            path (str): 文件路径
            instruction (dict): 输入指令
            predict_result (dict): 预测结果
        """
        with open(path, 'a', encoding='utf-8') as f:
            tmp = instruction
            tmp['reasoning_content'] = predict_result.get('reasoning_content', "") if isinstance(predict_result, dict) else ""
            tmp['predict_result'] = predict_result.get('content', predict_result) if isinstance(predict_result, dict) else predict_result
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")


async def debug():
    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")
    test = GPTClient(api_name='chatgpt',api_key='your api key',model_name='gpt-4.1',base_url='hhttps://api.openai.com/v1/chat/completions',semaphore_limit=5)
    instructions = ['who are you','hello!','hello2','hello3','hello4','hello5','hello6','hello7']
    results = await test.texts2texts(instructions,output_file="reasoning.jsonl")
    print(results)


if __name__ == "__main__":
    asyncio.run(debug())
