import yaml
import subprocess
import traceback
import select
import os
import argparse
import importlib
import asyncio
import requests
import time
import json
from tqdm import tqdm
from loguru import logger
from statistic import statistic

cur_path = os.path.dirname(os.path.abspath(__file__))

remote_model_port = os.getenv('REMOTE_MODEL_PORT')
judge_model_port = os.getenv('JUDGE_MODEL_PORT')
session = requests.Session()
task_status: dict[str, dict] = {}
model_status: dict[str, dict] = {}


def check_inference_engine_running(port=None):
    if port is None:
        port = remote_model_port
    ps_res = os.popen(f'ps auxww | grep predict_multi_gpu.py | grep "\-\-server_port {port}" | grep -v grep').read()
    return bool(ps_res)


async def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
    )
    while True:
        # 检查进程是否结束
        if process.poll() is not None:
            break
        await asyncio.sleep(1.)

    # 获取命令的返回码
    rc = process.poll()


async def send_request(data, method, port, path='file', retry_interval=10, max_retries=10, ignore_errors=False):
    """用于请求推理引擎，包括：提交文件、请求推理进度、发送终止信号"""
    assert method in ['get', 'post']
    retry_times = 0
    while True:
        if not check_inference_engine_running(port):
            raise RuntimeError(f'推理引擎已退出！({port=})')
        try:
            if method == 'post':
                print('http://localhost:{}/{},{}'.format(port, path,data))
                r = session.post('http://localhost:{}/{}'.format(port, path), data=data)
            else:
                r = session.get('http://localhost:{}/{}'.format(port, path), params=data)
            r = r.json()
            assert r['status_code'] == 0, str(r)
            return r
        except OSError:  # 有时会出现 ConnectionResetError: [Errno 104] Connection reset by peer，超过系统连接数限制？尝试再次连接
            if not ignore_errors:
                traceback.print_exc()
            if retry_times > max_retries:
                if ignore_errors:
                    break
                else:
                    raise RuntimeError(f'请求失败：重试超过{max_retries}次！')
            await asyncio.sleep(retry_interval)
            retry_times += 1
            continue


async def run_judge_inference(config, task):
    """
    运行裁判员模型的推理流程，用于评估特定任务的结果。
    
    该函数负责准备裁判员模型的输入文件，提交推理请求，并等待推理完成。
    根据配置可以使用本地大模型或外部API作为裁判。
    
    参数:
        config (dict): 包含任务配置信息的字典，必须包含以下键:
            - save_dir: 保存输出结果的目录
            - tasks: 包含特定任务配置的字典
        task (str): 要评估的任务名称，此名称用于定位配置中的相关设置
            
    返回:
        list: 包含所有输出文件路径的列表
        
    流程:
        1. 准备裁判员模型的输入文件:
           - 如果没有预处理步骤，直接使用任务输出文件
           - 如果有预处理步骤，调用指定的预处理函数生成输入文件
        2. 提交裁判员推理:
           - 如果配置指定了外部API方法，调用相应的外部接口
           - 否则使用本地大模型作为裁判
        3. 等待推理完成并更新任务状态
        
    注意:
        - 预处理函数需要在配置中正确指定，格式为"module.submodule.function"
        - 外部API方法需要在ExternalApi枚举中定义
        - 函数执行是异步的，调用时需要使用await
    """
    output_path = []
    judge_data = config['tasks'][task]['judge']
    # 准备裁判员模型的输入文件
    if 'preprocess' not in judge_data:
        input_file = os.path.join(config['save_dir'], task + '.jsonl')
    else:
        preprocess_module, preprocess_func = judge_data['preprocess'].rsplit('.', 1)  # e.g. "utils.judge.data_preprocess"
        preprocess_module = importlib.import_module(preprocess_module)
        preprocess_func = getattr(preprocess_module, preprocess_func)
        judge_input_dir = 'judge/input'
        judge_prompt_path = config.get('tasks', {}).get(task, {}).get('judge_prompt')

        if judge_prompt_path:
            preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir, prompt_path=judge_prompt_path)
        else:
            preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir)
        #preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir)
        input_file = os.path.join(config['save_dir'], judge_input_dir, task + '.jsonl')
    # 提交裁判员推理
    d = {
        'input_path': input_file,
        'output_path': os.path.join(config['save_dir'], 'judge/output', task + '.jsonl')
    }
    
    if 'method' in judge_data:
        for k, v in judge_data.get('inference', {}).items():
            if k == 'output_dir':
                d['output_path'] = os.path.join(config['save_dir'], v, task + '.jsonl')
            else:
                d[k] = v
        from envs.constants import ExternalApi
        external_api = ExternalApi[config['tasks'][task]['judge']['method']]
        os.makedirs(os.path.join(config['save_dir'], 'judge/output'),exist_ok=True)
        with open(input_file,'r',encoding='utf-8') as f:
            data = [json.loads(l) for l in f]
        out = []
        if output_path:
            output_path.extend(d['output_path'])
        else:
            output_path.append(d['output_path'])
        input_list = [sample['instruction'] if 'instruction' in sample else sample for sample in data]
        logger.info("开始调用外部接口：{} 评估".format(config['tasks'][task]['judge']['method']))
        await external_api.texts2texts(input_list,output_file=d['output_path'])
    else:
        logger.info("开始使用本地大模型作为裁判进行评估")
        r = await send_request(d, 'post', judge_model_port)
        output_path.extend(r['output_path'])

        # 等待裁判员推理完成
        await wait_for_inference(task + ' judge', judge_model_port, input_file, output_path[-1])

    task_status[task]['judge'] = True  # 裁判员模型推理完成
    return output_path


async def run_task_inference(config, task):
    """
    为指定任务运行推理，使用外部API或本地模型。
    
    此函数通过以下步骤处理推理过程：
    1. 从配置中确定输入数据路径
    2. 检查是否应使用外部API
    3. 根据配置执行适当的推理方法
    
    参数:
        config (dict): 包含任务设置和路径的配置字典
        task (str): 要运行推理的任务名称
    
    返回:
        None: 此函数执行操作但不返回值
    
    可能引发:
        外部API调用或本地模型推理可能产生的异常
    """
    input_path = config['tasks'][task]['data_path']
    #获取envs检测是否有external api
    external_api = os.environ.get('API_NAME')
    
    if external_api:
        logger.info("Using external api: {} to predict".format(external_api))
        from envs.constants import ExternalApi
        external_api = ExternalApi[external_api]
        await wait_for_external_api(config,task, external_api,input_path)
    else:
        await send_request({
            'input_path': input_path,
            'output_path': os.path.join(config['save_dir'], task + '.jsonl'),
            'generation_params': json.dumps({
                k: v for k, v in config['tasks'][task].items() if k in ['max_new_tokens']
            })
        }, 'post', remote_model_port)
        await wait_for_inference(task, remote_model_port, input_path)


async def wait_for_external_api(config,task, external_api,input_path):
    """
    调用外部API处理输入数据并保存结果。
    
    根据输入数据类型（图像或文本）自动选择相应的API调用方法，并将结果保存到指定目录下。
    
    Args:
        config (dict): 配置信息，必须包含'save_dir'键，指定结果保存目录
        task (str): 任务名称，用于生成输出文件名
        external_api (object): 外部API接口对象，需要实现images2texts和texts2texts方法
        input_path (str): 输入数据文件路径，应为jsonl格式
        
    Returns:
        list: API调用的结果列表
        
    Notes:
        - 输入文件必须是jsonl格式，每行为一个可解析为JSON的字符串
        - 当消息中包含"image_url"时会调用图像处理API，否则调用文本处理API
        - 输出文件将保存为{save_dir}/{task}.jsonl
    """
    data = [json.loads(i) for i in open(input_path, encoding='utf8')]

    output_file = os.path.join(config['save_dir'], task + '.jsonl')
    os.makedirs(config['save_dir'], exist_ok=True)

    # NOTE 目前只做了图像和文本的调用支持
    if "image_url" in str(data[0]['messages']):
        results = await external_api.images2texts(data,output_file=output_file)
    else:
        results = await external_api.texts2texts(data,output_file=output_file)
        

async def wait_for_inference(task, port, input_path, output_path=None):
    r_old = None
    while True:
        await asyncio.sleep(5)
        r = await send_request({
            'input_path': input_path,
            'output_path': output_path,
        }, 'get', port)

        if r != r_old and r['n_pred'] > 0:  # 只有状态发生变化时才打印
            logger.info("current task:{} status {}".format(task, r))  # {"status_code":0,"is_done":false,"n_samples":244128,"n_pred":74}
            r_old = r

        if r['is_done']:
            break


async def maybe_start_judge_model():
    """
    等待全部测试集推理完成后，根据需要启动裁判员模型。
    
    此协程监控推理过程并管理到裁判员模型的转换：
    1. 持续检查所有推理任务是否完成
    2. 如果任何任务需要裁判员模型，则终止推理引擎
    3. 在允许GPU资源释放的时间后启动裁判员模型进程
    
    返回：
        无
        
    异常：
        RuntimeError: 如果在监控过程中推理引擎意外退出
    
    注意：
        - 如果未设置judge_model_port，函数将立即退出
        - 当所有任务完成推理且裁判员模型（如需要）启动后，函数退出
        - 在终止推理后等待60秒再启动裁判员模型
    """
    while True:
        # 添加判断如果不需要启动裁判员模型直接break
        if not judge_model_port:
            logger.info("不使用本地模型作为裁判，监听进程终止")
            break
        await asyncio.sleep(5)
        if not check_inference_engine_running():
            raise RuntimeError(f'推理引擎已退出！')
        if all(t.get('inference') for t in task_status.values()):  # 全部任务都已推理完成
            if any(t.get('judge') is False for t in task_status.values()):  # 任意任务需要裁判员模型
                logger.info('测试集推理完成，正在停止推理引擎，释放显卡，以启动裁判员模型')
                await terminate_inference_engine(remote_model_port)  # 终止模型推理引擎，腾空显卡
                await asyncio.sleep(60)
                logger.info('正在启动裁判员模型')
                await send_request(None, 'post', judge_model_port, 'start')  # 启动裁判员模型推理进程
                await asyncio.sleep(5)
            break


async def run_loss_eval(config, task, model_path):
    cmd = "python -u {} --model_name_or_path {} --template {} --task {} --split test --lang default --n_shot 0 --batch_size 8 --save_dir {} --task_path {} --eval_type {} && ".format(os.path.join(cur_path, "eval.py"), model_path,config['prompt_type'],task,config['save_dir'],config['tasks'][task]['data_path'],config['tasks'][task]['type'])
    cmd += "python {} --eval_func {} --input_path '{}' --output_path '{}'\n".format(os.path.join(cur_path, "post_eval.py"),os.path.join(cur_path, "utils/eval_loss.py"), os.path.join(config['save_dir'],task+'.jsonl'),os.path.join(config['save_dir'],task+'.log'))
    await run_command(cmd)
    task_status[task]['inference'] = True  # 模型推理完成


async def run_next_word_prob_eval(config, task, model_path):
    cmd = "python -u {} --model_name_or_path {} --template {} --task {} --split test --lang default --n_shot 0 --batch_size 8 --save_dir {} --task_path {} --eval_type {} && ".format(os.path.join(cur_path, "eval.py"), model_path,config['prompt_type'],task,config['save_dir'],config['tasks'][task]['data_path'],config['tasks'][task]['type'])
    cmd += "python {} --eval_func {} --input_path '{}' --output_path '{}'\n".format(os.path.join(cur_path, "post_eval.py"),os.path.join(cur_path, "utils/eval_next_word_probability.py"), os.path.join(config['save_dir'],task+'.jsonl'),os.path.join(config['save_dir'],task+'.log'))
    await run_command(cmd)
    task_status[task]['inference'] = True  # 模型推理完成


async def run_mt_bench_eval(config, task):
    async def get_response(msg):
        result = await send_request(json.dumps({"prompt":{"instruction":msg}}), 'post', remote_model_port, path='model')
        return result

    with open(config['tasks'][task]['data_path'],'r') as f:
        data = [json.loads(l) for l in f]

    out = []
    for d in tqdm(data,desc="predict mtbench"):
        # import pdb;pdb.set_trace()
        prompt_1 = d['turns'][0]
        result_1 = await get_response(prompt_1)
        
        d['response'] = [result_1.json()['output']] if result_1 else []

        prompt_2 = "Question:{}\nAnswer:{}\nQuestion:{}".format(d['turns'][0],d['response'][0],d['turns'][1])
        result_2 = await get_response(prompt_2)
        if result_1:
            d['response'].append(result_2.json()['output'])
        out.append(d)
    with open("{}/mtbench.jsonl".format(config['save_dir']),'w') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    task_status[task]['inference'] = True  # 模型推理完成
    
    await run_post_eval(config, task, os.path.join(config['save_dir'], task+'.jsonl'))


async def run_text_eval(config, task):
    await run_task_inference(config, task)
    task_status[task]['inference'] = True  # 模型推理完成

    if 'judge' in config['tasks'][task]:
        task_status[task]['judge'] = False  # 等待裁判员模型推理
        output_path = await run_judge_inference(config, task)
        output_path = ','.join(output_path)
    else:
        output_path = os.path.join(config['save_dir'], task + '.jsonl')

    await run_post_eval(config, task, output_path)


async def run_post_eval(config, task, output_path):
    func_path = config['tasks'][task]['compare_func']['path']
    params = config['tasks'][task]['compare_func'].get('params')
    params = " --kwargs '{}'".format(json.dumps(params)) if params else ""
    cmd_eval = "python {} --eval_func {} --input_path '{}' --output_path '{}'{}\n".format(os.path.join(cur_path, "post_eval.py"),
        func_path, output_path, os.path.join(config['save_dir'], task + '.log'), params
    )
    print(cmd_eval)
    await run_command(cmd_eval)


async def llm_eval(file_path, model_name=""):
    """模型评估

    Args:
        file_path (str): config地址
        model_name (int): 可选参数，存储模型的sub path

    Returns:
        None

    Examples:
        >>> llm_eval(path, model_name)
        None
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config["save_dir"] = config["save_dir"] + f"/{model_name}"
    print(config)
    model_path = os.getenv('MODEL_PATH')
    os.makedirs(config["save_dir"], exist_ok=True)

    tasks_async = {}
    
    for task in config['tasks']:
        logger.info(f'开始评测任务：{task}')
        task_status[task] = {}

        if config['tasks'][task]['type'] in ['loss']:
            coro = run_loss_eval(config, task, model_path)
        elif config['tasks'][task]['type'] in ['next_word_probability']:
            coro = run_next_word_prob_eval(config, task, model_path)
        elif config['tasks'][task]['type'] in ['text']:
            if task == 'mtbench':
                coro = run_mt_bench_eval(config, task)
            else:
                coro = run_text_eval(config, task)

        tasks_async[task] = asyncio.create_task(coro)
        await asyncio.sleep(1)

    await maybe_start_judge_model()
    await asyncio.gather(*tasks_async.values())
    

    statistic(config["save_dir"],config)


async def terminate_inference_engine(port=None, ignore_errors=False):
    if port is not None:
        ports = [port]
    else:
        ports = [remote_model_port, judge_model_port]
    for port in ports:
        if port and not model_status.get(port, {}).get('terminated'):
            await send_request(None, 'post', port, 'terminate', ignore_errors=ignore_errors)
            model_status.setdefault(port, {})['terminated'] = True


async def main(args):
    try:
        await llm_eval(args.config, args.model_name)
        await terminate_inference_engine()
    except:
        await terminate_inference_engine(ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="File path of config yaml")
    parser.add_argument("--model_name", type=str, required=False, default="", help="model name that is evaluated")
    args = parser.parse_args()
    ## 添加判断，知道推理服务启动再开始评估
    t0 = time.time()

    while True:
        logger.info("infer engine initializing {:.2f} s".format(time.time()-t0))
        if time.time()-t0 > 20:
            asyncio.run(main(args))
            break
        time.sleep(5)
