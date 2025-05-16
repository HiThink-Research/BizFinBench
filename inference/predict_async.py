import os
import sys

os.environ['LOG_LEVEL'] = 'CRITICAL'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_NVLS_ENABLE'] = '0'  # 如不设置，tensor_parallel=8时，腾讯云报错：misc/socket.cc:484 NCCL WARN socketStartConnect: Connect to 10.170.1.33<37999> failed : Cannot assign requested address
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
for i in range(len(sys.argv)):
    if sys.argv[i] == '--device':
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        del sys.argv[i: i + 2]
        break

import argparse
import asyncio
import time
import traceback
import math
import ujson as json
import subprocess
import aiohttp
import torch
import transformers

from importlib.metadata import version
from collections import OrderedDict
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput
from loguru import logger

import vllm.inputs

vllm_prompt_type = 0
if hasattr(vllm.inputs, 'PromptInputs') or hasattr(vllm.inputs, 'PromptType'):  # vllm 0.5.0之后，LLMEngine.generate 入参改为 PromptInputs，0.6.3之后改为 PromptType
    vllm_prompt_type = 1

is_v1_engine = version('vllm') >= '0.8.0'

from multimodal import load_multimodal_data


OUTPUT_TYPES = ['text', 'reward', 'loss', 'prompt_tokens', 'next_token_prob']
waiting = set()


async def run_predict_until_complete(model: AsyncLLMEngine, args: argparse.Namespace):
    """`predict_async.py`的主函数，异步请求data server获取待预测样本，提交vllm推理，最后把结果推送给data server，直到所有数据推理完成"""

    engine = getattr(model, 'engine_core', None) or model.engine
    model.log_requests = False
    engine.log_stats = False

    idx2task = OrderedDict()
    r_latest = {}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as session:  # 取消超时限制（默认5分钟）

        async def add_sample_until_full():
            """从data server获取待预测样本，直到填满推理队列"""
            n_added = 0
            while True:
                if (
                    n_added >= 2  # 一次最多添加 2 条样本
                    or len(waiting) >= 8  # 超过 8 条样本在排队，暂停添加新样本
                    or (idx2task and time.time() - float(next(iter(idx2task.values())).get_name()) > 120)  # 有样本超过 120 秒未完成，暂停添加新样本
                ):
                    break
                try:
                    async with session.get(f'http://{args.server_addr}:{args.server_port}/data') as r:
                        r = await r.text()
                        r = json.loads(r)
                        r_latest.update(r)
                        if r['status_code'] > 0:  # 1: all done, 2: terminated
                            return
                        # print('add:', r['idx'])
                        otype = r.get('output_type', args.output_type)
                        params = r.get('generation_params', {})
                        if otype == 'loss':
                            sampling_params = SamplingParams(
                                temperature=0,
                                prompt_logprobs=1,
                                max_tokens=1
                            )
                        elif otype == 'prompt_tokens':
                            sampling_params = SamplingParams(
                                temperature=0,
                                prompt_logprobs=20,  # 默认top-20，后期需要自定义可改为入参传入
                                max_tokens=1
                            )
                        elif otype == 'next_token_prob':
                            sampling_params = SamplingParams(
                                temperature=0,
                                max_tokens=1,
                                logprob_token_ids=r.get('next_token_ids')
                            )
                        else:
                            sampling_params = SamplingParams(
                                presence_penalty=args.presence_penalty,
                                temperature=args.temperature,
                                stop=args.stop,
                                max_tokens=params.get('max_new_tokens', args.max_new_tokens),
                            )

                        results_generator = await generate(model, r['inputs'], r['idx'], sampling_params)
                        idx2task[r['idx']] = asyncio.create_task(
                            process_request(r, results_generator),
                            name=str(time.time())
                        )
                        waiting.add(r['idx'])
                        n_added += 1
                except (
                    aiohttp.client_exceptions.ClientConnectionError,
                    asyncio.exceptions.TimeoutError
                ):
                    # print('get err')
                    await asyncio.sleep(3)

        async def process_request(inputs, results_generator):
            """将预测完成的样本推送给data server"""
            idx = inputs['idx']
            final_output = await get_generator_result(results_generator, idx)
            # import pdb; pdb.set_trace()

            # print('done:', idx)
            otype = inputs.get('output_type', args.output_type)
            output = convert_final_output(inputs, final_output, otype=inputs.get('output_type', args.output_type))
            data = {'idx': idx, 'output_type': otype, 'output': output}

            while True:
                try:
                    async with session.post(f'http://{args.server_addr}:{args.server_port}/result', data=json.dumps(data)) as r:
                        r = await r.text()
                        r = json.loads(r)
                        break
                except OSError:  # 有时会出现 ConnectionResetError: [Errno 104] Connection reset by peer，超过系统连接数限制？尝试再次连接
                    # traceback.print_exc()
                    await asyncio.sleep(1.)
            await add_sample_until_full()  # 尝试添加新样本的推理队列
            task_done = idx2task.pop(idx)

        await add_sample_until_full()
        while True:
            if idx2task:
                await asyncio.gather(*idx2task.values())
            elif r_latest.get('status_code', 0) == 2:  # terminated
                break
            else:
                await add_sample_until_full()
                await asyncio.sleep(1)


def convert_final_output(inputs: dict, final_output: RequestOutput | list[RequestOutput], otype: str):
    """
    将vllm的输出转换为返回data server的数据格式

    Args:
        inputs (`dict`):
            通过调用data server获取的模型输入
        final_output (`RequestOutput | list[RequestOutput]`):
            vllm的输出
        otype (`str`):
            输出结果的类型，支持的类型见`OUTPUT_TYPES`
    """
    if isinstance(final_output, list):
        return [convert_final_output(inputs, o, otype) for o in final_output]

    if otype == 'reward':
        return final_output.outputs[0].rewards
    elif otype == 'text':
        output_ids = [
            o.token_ids if isinstance(o.token_ids, (tuple, list)) else tuple(o.token_ids)  # vllm 0.5.5: array -> tuple
            for o in final_output.outputs
        ]
        return output_ids[0] if len(output_ids) == 1 else output_ids
    elif otype == 'beam_search':
        return final_output
    elif otype == 'loss':
        logprobs = [
            logprobs[token_id] if isinstance(logprobs[token_id], float) else logprobs[token_id].logprob  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
            for token_id, logprobs in zip(inputs['inputs'], final_output.prompt_logprobs)
            if logprobs
        ]
        return - sum(logprobs) / len(logprobs)
    elif otype == 'prompt_tokens':
        return [
            {
                token_id: round(logprobs_j if isinstance(logprobs_j, float) else logprobs_j.logprob, 3)  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
                for token_id, logprobs_j in logprobs_i.items()
            }
            for logprobs_i in final_output.prompt_logprobs
            if logprobs_i
        ]
    elif otype == 'next_token_prob':
        return {
            token_id: math.exp(logprob if isinstance(logprob, float) else logprob.logprob)  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
            for token_id, logprob in final_output.outputs[0].logprobs[0].items()
            if token_id in inputs['next_token_ids']
        }
    else:
        raise NotImplementedError(f'Unsupported output_type: "{otype}"')


async def generate(model: AsyncLLMEngine, inputs: list[int] | dict, request_id: str, sampling_params: SamplingParams):
    """
    单条样本推理

    Args:
        model (`AsyncLLMEngine`):
            通过vllm加载的模型
        inputs (`list[int] | dict`):
            模型输入，文本输入的token ids（`list[int]`），或多模态的输入（`dict`）
        request_id (`str`):
            请求的唯一id
        sampling_params (`SamplingParams`):
            采样参数，参考vllm.sampling_params
    """
    if isinstance(inputs, dict):  # multimodal
        # inputs['multi_modal_data'] = load_multimodal_data(processor, **inputs['multi_modal_data'], return_dict=True)
        inputs['multi_modal_data'] = await asyncio.get_running_loop().run_in_executor(  # 使用多线程加载多模态数据
            None,
            load_multimodal_data,
            processor,
            inputs['multi_modal_data'].get('image'),
            inputs['multi_modal_data'].get('audio'),
            True,  # return_dict=True
        )

    else:  # list[int]
        inputs = dict(prompt=None, prompt_token_ids=inputs)

    if vllm_prompt_type == 1:  # vllm >= 0.5.3
        return model.generate(
            inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        )
    else:
        return model.generate(
            **inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        )



async def get_generator_result(result_generator, request_id) -> RequestOutput:
    if isinstance(result_generator, list):
        final_output = await asyncio.gather(*[get_generator_result(g, request_id) for g in result_generator])
        return final_output

    final_output = None
    async for request_output in result_generator:
        waiting.discard(request_id)
        final_output = request_output
    return final_output


def get_gpu_count():
    """获取当前进程可用的GPU数量"""
    device = os.environ.get('CUDA_VISIBLE_DEVICES')
    if device is None:
        ngpus = int(subprocess.run(  # all available gpus
            'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l', shell=True, stdout=subprocess.PIPE
        ).stdout)
    else:
        assert device, 'No available gpus! (CUDA_VISIBLE_DEVICES=)'
        ngpus = len(device.split(','))
    return ngpus


def load_model(args: argparse.Namespace):
    """加载模型，以及processor（tokenizer）"""
    args.trust_remote_code = True
    config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    processor = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.output_type == 'reward':
        args.max_new_tokens = 1
        assert {
            'LlamaForRewardModelWithGating',
            'LlamaForSequenceClassification',
            'Qwen2ForRewardModel',
            'RewardModel',
        }.intersection(getattr(config, "architectures", [])), '不是Reward模型！'

    if hasattr(config, 'text_config'):  # e.g. Qwen2-Audio
        config = config.text_config

    max_model_length = int(
        getattr(config, 'max_position_embeddings', 0) * (getattr(config, 'rope_scaling', {}) or {}).get('factor', 1.)
        or getattr(config, 'seq_length', 0)
    )

    import vllm.config
    get_config = vllm.config.get_config

    def _get_config(model, *_args, **kwargs):
        config = get_config(model, *_args, **kwargs)
        if 'RewardModel' in config.architectures:  # OpenRLHF训练的RewardModel
            if config.model_type == 'llama':
                config.architectures.append('LlamaForRewardModel')
            elif config.model_type == 'qwen2':
                config.architectures.append('Qwen2ForRewardModel')
            else:
                raise ValueError('不支持当前Reward模型！')
        if any(config.__class__.__name__.startswith(custom_model) for custom_model in ['Dongwu', 'Hithinkgpt']):
            if getattr(config, 'qkv_bias', False):
                config = transformers.Qwen2Config.from_pretrained(model)
                config.architectures.append('Qwen2ForCausalLM')
            else:
                config.architectures.append('MistralForCausalLM')
        text_config = getattr(config, 'text_config', config)
        if args.max_length and args.max_length != max_model_length and not is_v1_engine:
            max_pe = int(args.max_length / (getattr(text_config, 'rope_scaling', {}) or {}).get('factor', 1.))
            print(f'Overriding max_position_embeddings: {text_config.max_position_embeddings} -> {max_pe}')
            text_config.max_position_embeddings = max_pe
        return config

    vllm.config.get_config = _get_config

    if is_v1_engine:
        if args.max_length is not None:
            args.max_model_len = args.max_length

    if args.low_vram:
        max_length = args.max_length or max_model_length
        if args.block_size is None:
            args.block_size = 32  # 默认值
        num_cache_blocks = int((max_length or 8192) * 1.1 / args.block_size)
        for key in ['num_gpu_blocks', 'num_cpu_blocks', 'forced_num_gpu_blocks', 'num_gpu_blocks_override']:
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, num_cache_blocks)
        args.gpu_memory_utilization = 1  # 防止出现类似报错：The model's max seq len (8192) is larger than the maximum number of tokens that can be stored in KV cache (7984). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
    # else:
    #     args.gpu_memory_utilization = 0.8  # 默认0.9，遇到过70B模型cache分配异常的情况
    if hasattr(args, 'disable_custom_all_reduce'):
        args.disable_custom_all_reduce = True
    if hasattr(args, 'enforce_eager') and getattr(config, 'dual_chunk_attention_config', None):
        args.enforce_eager = True

    args.tensor_parallel_size = get_gpu_count()
    if args.tensor_parallel_size > 1 and hasattr(args, 'enforce_eager') and int(torch.version.cuda.split('.')[0]) < 12:  # https://github.com/vllm-project/vllm/issues/7548
        args.enforce_eager = True

    if "Processor" in type(processor).__name__:
        args.limit_mm_per_prompt = {"image":10}
    engine_args = AsyncEngineArgs.from_cli_args(args)
    logger.info(engine_args)
    model = AsyncLLMEngine.from_engine_args(engine_args)
    return model, processor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0., help="0.0 means greedy decoding")
    parser.add_argument("--presence_penalty", type=float, default=0., help="0.0 means no penalty")
    parser.add_argument("--stop", type=str, default=None, help="token/word at which generation will be stopped")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    args.stop = args.stop.replace('\\n', '\n').split(',') if args.stop else None
    model, processor = load_model(args)

    asyncio.run(run_predict_until_complete(model, args))
