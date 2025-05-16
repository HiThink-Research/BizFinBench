import os
import argparse
import random
import ujson as json
import asyncio
import time
import importlib
import traceback
import datasets

from collections import deque
from dataclasses import dataclass, field
from transformers import AutoProcessor, PreTrainedTokenizer
from sanic import Sanic, response
from sanic.request import Request
from typing import Callable, List, Dict, Deque, Optional, Union


OUTPUT_TYPES = ['text', 'reward', 'loss', 'prompt_tokens', 'next_token_prob']

app = Sanic("DataServer")
app.config.RESPONSE_TIMEOUT = 60 * 60 * 24  # 24 hours

args: argparse.Namespace = None
postprocess: Callable = None
tokenizer: PreTrainedTokenizer = None
stop: str = None
preprocess: str = None
preprocessors: Dict[str, Callable] = {}  # preprocess -> convert_sample_to_inputs

info = {  # 全局信息
    'i_pred': 0,   # 当前预测的data序号
    'n_queue': 0,  # 已提交到推理队列的样本数量
    'n_pred': 0,   # 已返回预测结果的样本数量
    'n_read': 0,   # 已提交到推理队列的文件数量
    'n_skip': 0,   # （因为超过最大长度）跳过的样本数量
    'n_done': 0,   # 完成写入的文件数量
    'n_token': 0,   # 已生成的token数量
    't_start': 0.,  # 开始预测的时间戳
    't_curr': 0.,   # 当前的时间戳
    'error': 0,
    'terminated': False,  # 下游发送终止信号
    'post_start': False  # 下游发送启动信号
}


@dataclass
class FileData:
    input_path: str  # './input/a.json'
    output_path: str  # './output/a.json'
    file_type: str = 'jsonl'
    preprocess: Optional[str] = None
    output_key: Optional[str] = None
    samples: Optional[List[dict]] = None
    results: Optional[List[dict]] = None
    results_new: Optional[List[dict]] = None  # 仅用于no_order为True时，保存预测完成待写入的样本
    subsample_ids: Optional[List[int]] = None  # 仅用于args.subsample不为None时，采样一部分数据进行预测
    j_pred: int = 0   # 当前预测的样本序号
    n_pred: int = 0   # 已有预测结果的数量
    n_write: int = -1   # 已写入文件的样本数量
    read: Optional[asyncio.Task] = None
    write: tuple[asyncio.Semaphore, dict[int, asyncio.Task]] = field(
        default_factory=lambda : (asyncio.Semaphore(), {})
    )
    is_done: bool = False  # 已完成预测并写入结果
    generation_params: Optional[dict] = None


@dataclass
class Sample:
    inputs: list[int] | dict
    output_type: Optional[str] = None
    output: Optional[Union[str, float, dict]] = None
    next_token_ids: Optional[List[int]] = None  # only used when output_type == 'next_token_prob'
    uid: str = field(default_factory=lambda : str(time.time()))
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


file_list: List[FileData] = []   # 待预测的文件
file_path2idx: Dict[str, int] = {}

input_queue: Deque[Sample] = deque()  # 通过HTTP请求添加的待预测样本
uid2sample: Dict[str, Sample] = {}


def is_file_all_read(d: FileData):
    """判断数据（文件）是否已全部读取并提交推理"""
    return d.read is not None and d.read.done() and d.j_pred == len(
        d.samples if args.subsample is None else d.subsample_ids
    )


def is_file_all_done(d: FileData, written=False):
    """判断数据（文件）是否已全部完成推理"""
    n_curr = d.n_write if written else d.n_pred
    return d.read is not None and d.read.done() and n_curr == len(
        d.samples if args.subsample is None else d.subsample_ids
    )


def get_num_done():
    """返回已完成推理的数据（文件）数量"""
    n = 0
    for d in file_list:
        if not d.is_done:
            d.is_done = is_file_all_done(d, written=True)
        if d.is_done:
            n += 1
    return n


@app.get("/info")
async def get_info(request: Request):
    """供主进程（`predict_multi_gpu.py`）调用，返回全局信息，包括已完成的样本/文件数量等"""
    info['n_file'] = len(file_list)
    info['n_done'] = get_num_done()
    info['n_queue'] = sum(d.j_pred for d in file_list[:info['i_pred'] + 1])
    info['n_read'] = sum(1 for d in file_list[:info['i_pred'] + 1] if is_file_all_read(d))
    info['t_curr'] = time.time()
    return response.json(info)


@app.get("/data")
async def get_data(request: Request):
    """供worker（predict_async.py）调用，返回待预测的inputs"""
    try:
        if info['terminated']:
            return response.json({'status_code': 2, 'status_msg': 'terminated'})

        if not info['t_start']:
            info['t_start'] = time.time()

        while input_queue:
            s = input_queue.popleft()
            if args.max_length and isinstance(s.inputs, list) and len(s.inputs) >= args.max_length:
                s.output = ''
                s.done_event.set()
                info['n_skip'] += 1
                continue
            r = {
                'status_code': 0,
                'inputs': s.inputs,
                'output_type': s.output_type,
                'idx': s.uid
            }
            if s.next_token_ids is not None:
                r['next_token_ids'] = s.next_token_ids
            return response.json(r)

        while True:
            i = info['i_pred']
            try:
                d = file_list[i]
            except IndexError:
                d = None
            if d is None or is_file_all_read(d):  # 当前数据（文件）已全部提交推理
                if i < len(file_list) - 1:  # 还有其他数据文件
                    i += 1
                    info['i_pred'] = i
                    d = file_list[i]
                else:
                    # print('get: all done!')
                    return response.json({'status_code': 1})

            if d.read is None:  # 开始读取数据（异步）
                d.read = asyncio.get_running_loop().run_in_executor(
                    None, load_file, i
                )

            idx = d.j_pred
            try:
                s = d.samples[idx if args.subsample is None else d.subsample_ids[idx]]
            except (TypeError, IndexError):
                await asyncio.sleep(0.1)  # 等待读取
                continue
            d.j_pred = idx + 1

            output_key = d.output_key or args.output_key
            if args.reuse and (s.get(output_key) and not s[output_key].startswith('ERROR')):
                set_result(i, idx, s)
                continue

            # print('get:', d.input_path, idx)

            convert_sample_to_inputs = get_preprocess_func(d.preprocess)
            inputs = convert_sample_to_inputs(s, args.prompt, tokenizer)
            if args.max_length and isinstance(inputs, list) and inputs and isinstance(inputs[0], int) and len(inputs) >= args.max_length:
                r = s.copy()
                r[output_key] = ''
                set_result(i, idx, r)
                info['n_skip'] += 1
                continue

            r = {
                'status_code': 0,
                'inputs': inputs,
                'output_type': args.output_type,
                'idx': '_'.join([str(i), str(idx)]),
                'generation_params': d.generation_params or {},
            }
            if 'max_new_tokens' in s:  # 支持针对单条样本设置生成长度
                r['generation_params']['max_new_tokens'] = s['max_new_tokens']
            return response.json(r)
    except:
        info['error'] += 1
        raise


@app.post("/result")
async def post_result(request: Request):
    """供worker（predict_async.py）调用，传入预测结果"""
    try:
        otype = request.json['output_type']
        output = request.json['output']
        if otype in ['text', 'beam_search']:
            output = convert_output_ids_to_text(output)
        elif otype == 'reward':
            if isinstance(output, dict):
                output['rewards'] = [round(s, 4) for s in output['rewards']]
                output['gating_output'] = [round(s, 4) for s in output['gating_output']]

        idx = request.json['idx']
        # print('post:', idx)
        if idx in uid2sample:
            s = uid2sample.pop(idx)
            s.output = output
            s.done_event.set()
        else:
            i_data, i_sample = map(int, idx.split('_'))
            d = file_list[i_data]
            r = d.samples[i_sample if args.subsample is None else d.subsample_ids[i_sample]].copy()
            output_key = d.output_key or args.output_key
            r[output_key] = output
            set_result(i_data, i_sample, r)
            d.write[1][i_sample] = asyncio.create_task(write_result_async(i_data, i_sample))
        info['n_pred'] += 1
        if info['terminated']:
            return response.json({'status_code': 2, 'status_msg': 'terminated'})
        else:
            return response.json({'status_code': 0})
    except:
        info['error'] += 1
        raise


@app.post("/model")
async def post_inputs(request: Request):
    """供下游任务调用，传入待预测的样本"""
    try:
        prompt = request.json.get('prompt')
        if prompt:
            convert_sample_to_inputs = get_preprocess_func()
            inputs = convert_sample_to_inputs(prompt, args.prompt, tokenizer)
        else:
            inputs = request.json.get('inputs')
        if not inputs:
            return response.json({
                'status_code': -1,
                'status_msg': f'"inputs" is missing!'
            })
        if isinstance(inputs[0], int):
            is_batched = False
            inputs = [inputs]
        else:
            is_batched = True

        otype = request.json.get('output_type')
        if otype is None:
            otype = args.output_type
        elif otype not in OUTPUT_TYPES:
            return response.json({
                'status_code': -1,
                'status_msg': f'"output_type" must be one of {OUTPUT_TYPES}!'
            })

        samples = []
        for inputs_i in inputs:
            s = Sample(
                inputs=inputs_i,
                output_type=otype,
                next_token_ids=request.json.get('next_token_ids') if otype == 'next_token_prob' else None
            )
            samples.append(s)
            input_queue.append(s)
            uid2sample[s.uid] = s

        await asyncio.gather(*[s.done_event.wait() for s in samples])
        output = [s.output for s in samples] if is_batched else samples[0].output
        return response.json({'status_code': 0, 'output': output, 'output_type': otype})

    except:
        info['error'] += 1
        raise


@app.post("/file")
async def post_file(request: Request):
    """供下游任务调用，传入待预测的文件"""
    try:
        input_path = request.form.get('input_path')
        if not input_path:
            return response.json({
                'status_code': -1,
                'status_msg': f'"input_path" is missing!'
            })
        output_dir = request.form.get('output_dir')
        output_path = request.form.get('output_path')
        generation_params = request.form.get('generation_params')
        if not (output_dir or output_path):
            return response.json({
                'status_code': -1,
                'status_msg': f'"output_dir" or "output_path" is missing!'
            })
        added, existed = add_file_for_prediction(
            input_path,
            output_dir=output_dir,
            output_path=output_path,
            generation_params=generation_params,
        )

        p = request.form.get('preprocess')
        if p:
            get_preprocess_func(p)
            for d in added:
                d.preprocess = p

        v = request.form.get('output_key')
        if v:
            for d in added:
                d.output_key = v

        return response.json({
            'status_code': 0,
            'n_added': len(added),
            'output_path': [d.output_path for d in (added + existed)]
        })
    except:
        return response.json({
            'status_code': -1,
            'status_msg': traceback.format_exc()
        })


@app.get("/file")
async def get_file(request: Request):
    """供下游任务调用，返回文件信息（是否已完成、已完成的样本数等）"""
    try:
        input_path = request.args.get('input_path')
        d = get_file_info(
            input_path,
            request.args.get('output_path')
        )
        if d is None:
            return response.json({
                'status_code': -1,
                'status_msg': f'"{input_path}" is *NOT* in queue!'
            })
        return response.json({
            'status_code': 0,
            'is_done': is_file_all_done(d, written=True),
            'n_samples': len(d.samples) if d.samples else 0,
            'n_pred': d.n_pred,
        })
    except:
        return response.json({
            'status_code': -1,
            'status_msg': traceback.format_exc()
        })


def get_file_info(input_path, output_path):
    """根据文件的输入输出路径，获取推理状态信息"""
    if output_path is None:
        for f in reversed(file_list):
            if f.input_path == input_path:
                return f
    else:
        i = file_path2idx.get((input_path, output_path))
        if i is not None:
            return file_list[i]


@app.post("/start")
async def post_start(request: Request):
    """供下游任务调用，启动推理进程（当manual_start为True时）"""
    info['post_start'] = True
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


@app.post("/terminate")
async def post_terminate(request: Request):
    """供下游任务调用，退出推理，结束所有相关进程"""
    info['terminated'] = True
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


def get_preprocess_func(preprocess_file: str = None):
    if preprocess_file is None:
        preprocess_file = preprocess
    if preprocess_file not in preprocessors:
        module = importlib.import_module(preprocess_file)
        assert args.prompt in module.PROMPT, f'Can not find prompt type "{args.prompt}" in module "{preprocess_file}"!'
        f = getattr(module, 'convert_sample_to_input_ids')
        preprocessors[preprocess_file] = f
    return preprocessors[preprocess_file]


def convert_output_ids_to_text(output_ids):
    """将推理输出的token ids转换为文本"""
    if not output_ids:  # 推理无结果（输入超过模型最大长度？）
        return
    if not isinstance(output_ids[0], list):  # output_ids: list[int]
        info['n_token'] += len(output_ids) + 1
        return remove_stop_str(tokenizer.decode(output_ids, skip_special_tokens=True), stop).strip()
    else:  # output_ids: list[list[int]]
        return [convert_output_ids_to_text(o) for o in output_ids]


def remove_stop_str(t, stop):
    """去除文本中包含的停止符"""
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        while True:
            _t = t
            for s in stop:
                if _t.endswith(s):
                    _t = _t[:-len(s)]
            if _t == t:
                break
            else:
                t = _t
    return t


def set_result(i_data, i_sample, r):
    if postprocess is not None:
        r = postprocess(r)
    d = file_list[i_data]
    if d.results is None:
        d.results = []
        d.results_new = deque()
    rs = d.results
    while len(rs) < i_sample + 1:
        rs.append(None)
    rs[i_sample] = r
    if args.no_order:
        d.results_new.append(r)
    d.n_pred += 1


async def write_result_async(i_data, i_sample):
    """调用线程池异步写入结果"""
    d = file_list[i_data]
    sem, i2task = d.write
    async with sem:
        await asyncio.get_running_loop().run_in_executor(
            None, write_result, i_data
        )
    del i2task[i_sample]


def write_result(i_data):
    """将推理结果写入文件"""
    try:
        d = file_list[i_data]
        # print('save:', d.output_path)

        if d.file_type == 'json':
            if is_file_all_done(d):  # 已全部预测完
                os.makedirs(os.path.split(d.output_path)[0], exist_ok=True)
                with open(d.output_path, 'w') as f:
                    json.dump(d.results, f, ensure_ascii=False, indent=2)
        else:  # jsonl
            os.makedirs(os.path.split(d.output_path)[0], exist_ok=True)
            if d.n_write == -1:
                d.n_write = 0
                if args.overwrite:  # 覆盖旧文件
                    f = open(d.output_path, 'w')
                    f.close()
            with open(d.output_path, 'a') as f:
                if args.no_order:
                    while d.results_new:
                        r = d.results_new.popleft()
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                        d.n_write += 1
                else:
                    while d.n_write < len(d.results):
                        i = d.n_write
                        r = d.results[i]
                        if r is None:
                            break
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                        d.results[i] = None  # 释放内存
                        d.n_write += 1

    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_file(i_data):
    """加载输入数据文件"""
    try:
        d = file_list[i_data]
        if d.file_type.startswith('json'):
            load_json_file(i_data)
        elif d.file_type == 'dataset':
            load_dataset(i_data)
        if args.subsample is not None:
            if args.subsample >= 1.:
                n_sample = int(args.subsample)
            else:
                n_sample = max(int(len(d.samples) * args.subsample), 1)
            if args.seed is not None:
                random.seed(args.seed)
            if n_sample < len(d.samples):
                d.subsample_ids = random.sample(range(len(d.samples)), n_sample)
            else:
                d.subsample_ids = list(range(len(d.samples)))
    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_json_file(i_data):
    """加载json/jsonl格式的数据文件"""
    try:
        d = file_list[i_data]
        # print('read:', d.input_path)
        if args.reuse and os.path.isfile(d.output_path):
            file_path = d.output_path
        else:
            file_path = d.input_path

        with open(file_path) as f:
            try:
                l = next(f)  # 读取一行，用来判断文件是json还是jsonl格式
            except StopIteration:  # 空文件
                d.samples = []
                d.is_done = True
                return
            else:
                f.seek(0)
                try:
                    _ = json.loads(l)
                except ValueError:
                    d.file_type = 'json'
                    d.samples = json.load(f)  # 整个文件是一个json对象
                else:
                    d.file_type = 'jsonl'
                    d.samples = []
                    # 已有jsonl格式的结果，跳过已预测的样本
                    if not args.overwrite and os.path.isfile(d.output_path):
                        n_pred = int(os.popen('wc -l ' + d.output_path).read().split()[0])
                        d.j_pred = n_pred
                        d.n_pred = n_pred
                        d.n_write = n_pred
                    for l in f:  # 每一行是一个json对象
                        d.samples.append(json.loads(l))
                    if d.n_write == len(d.samples):  # 已有全部样本的预测结果
                        d.is_done = True

    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_dataset(i_data):
    """加载datasets格式的数据文件"""
    try:
        d = file_list[i_data]
        d.samples = datasets.load_from_disk(d.input_path)
        # 已有jsonl格式的结果，跳过已预测的样本
        if not args.overwrite and os.path.isfile(d.output_path):
            n_pred = int(os.popen('wc -l ' + d.output_path).read().split()[0])
            d.j_pred = n_pred
            d.n_pred = n_pred
            d.n_write = n_pred
        if d.n_write == len(d.samples):  # 已有全部样本的预测结果
            d.is_done = True
    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_tokenizer(tokenizer_path):
    """加载processor/tokenizer"""
    global tokenizer
    tokenizer = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)


def add_file_for_prediction(data_path, output_dir=None, output_path=None, data_type='json', generation_params=None) -> List[FileData]:
    """添加文件到推理队列"""
    if data_type == 'json':
        fs = add_json_file_for_prediction(data_path, output_dir, output_path)
    else:
        fs = add_dataset_for_prediction(data_path, output_dir)

    if isinstance(generation_params, str):
        generation_params = json.loads(generation_params)

    added = []  # 新提交的文件
    existed = []  # 之前已提交过，避免重复推理
    for f in fs:
        k = (f.input_path, f.output_path)  # 根据“输入-输出”去重，因为有时即使输入路径相同，也会使用不同的preprocess，输出到不同路径
        if k in file_path2idx:
            existed.append(f)
        else:
            file_path2idx[k] = len(file_list)
            f.generation_params = generation_params
            file_list.append(f)
            added.append(f)
    return added, existed


def add_json_file_for_prediction(data_path, output_dir=None, output_path=None) -> List[FileData]:
    """
    添加json/jsonl文件到推理队列

    Args:
        data_path (`str`):
            输入的json/jsonl文件路径，支持多个输入文件，以英文逗号分隔，或者输入目录，自动读取目录下的所有json/jsonl文件
        output_dir (`str | None`):
            输出的目录，会自动创建与源文件同名的结果文件，仅当未指定`output_path`时生效
        output_path (`str | None`):
            输出的文件路径，如果指定了`output_path`，则`output_dir`不生效
    """
    if output_path:  # 如果指定了输出文件路径，输入路径必须是单个文件
        assert os.path.isfile(data_path), f"输入文件不存在：{data_path}！"
    fs = []
    if not os.path.isdir(data_path):  # 支持多个输入文件，以英文逗号分隔
        for f in data_path.split(','):
            fs.append(FileData(
                input_path=f,
                output_path=output_path or os.path.join(output_dir, os.path.basename(f))
            ))
    else:  # 读取目录下的所有json文件
        for p, q, v in os.walk(data_path):
            for f in v:
                if f.endswith('.json') or f.endswith('.jsonl'):
                    apath = os.path.join(p, f)
                    fs.append(FileData(
                        input_path=apath,
                        output_path=os.path.join(output_dir, apath[len(data_path):].lstrip(os.path.sep)),
                    ))
    return fs


def add_dataset_for_prediction(data_path, output_dir=None) -> List[FileData]:
    """添加datasets格式数据到推理队列"""
    fs = []
    for c in os.walk(data_path):
        if 'dataset_info.json' in c[2]:
            p = c[0]
            fs.append(FileData(
                input_path=p,
                output_path=os.path.join(output_dir, os.path.basename(p) if p == data_path else p[len(data_path):].lstrip(os.path.sep)),
                file_type='dataset'
            ))
    return fs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Data file to predict")
    parser.add_argument("--data_type", type=str, default='json', choices=['json', 'dataset'], help="Iutput/Output data type")
    parser.add_argument("--preprocess", type=str, default='preprocess', help="Module that provides preprocessing function")
    parser.add_argument("--postprocess", type=str, default=None, help="Module that provides postprocessing function")
    parser.add_argument("--prompt", type=str, default='Hithink', help="Prompt type")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--output_key", type=str, default='output', help="Output key")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer path")
    parser.add_argument("--port", type=int, default=7888, help="Server port")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_input_tokens", type=int, default=None,
                        help="Max number of input tokens, longer inputs will be truncated")
    parser.add_argument("--stop", type=str, default='', help="token/word at which generation will be stopped")
    parser.add_argument("--reuse", action='store_true',
                        help="If prediction file exists, reuse previous outputs and only predict samples with empty output")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite if output file exists. If not set, will skip finished samples")
    parser.add_argument("--no_order", action='store_true', help="Output will be written in different order than input")
    parser.add_argument("--subsample", type=float, help="proportion (0.0 - 1.0) or number (>= 1) of samples used")
    parser.add_argument("--seed", type=int, help="seed used for subsampling")
    args = parser.parse_args()

    preprocess = args.preprocess
    get_preprocess_func()
    if args.postprocess:
        postprocess = getattr(importlib.import_module(args.postprocess), 'postprocess')

    if args.data:
        add_file_for_prediction(args.data, args.output_dir, args.output_path, args.data_type)
    load_tokenizer(args.tokenizer)
    if args.max_length:
        tokenizer.model_max_length = args.max_length  # preprocess 代码可能会从 tokenizer 读取最大长度
    if args.stop:
        stop = args.stop.replace('\\n', '\n').split(',')

    app.run(host='0.0.0.0', port=args.port, single_process=True, access_log=False)
