import os
import argparse
import time
import datetime
import re
import json
import requests
import signal
import subprocess

from importlib.metadata import version
from subprocess import Popen, PIPE, TimeoutExpired

cur_path = os.path.dirname(os.path.abspath(__file__))

def terminate_subprocess(proc: Popen):
    for s in [signal.SIGINT, signal.SIGTERM]:
        for _ in range(2):
            try:
                proc.send_signal(s) # .terminate()
                proc.wait(timeout=3)
                return
            except TimeoutExpired:
                continue
    proc.kill()


def print_subprocess_out_err(proc):
    outs, errs = proc.communicate()
    if outs:
        print(outs.decode())
    if errs:
        print(errs.decode())


def is_gpu_idle():
    """有任意一块GPU使用率为0，则返回True"""
    gpu_info = os.popen('nvidia-smi | grep MiB').read().split('\n')
    for s in gpu_info:
        m = re.search(r'[0-9]+%', s)
        if m and m.group(0) == '0%':
            return True


def find_checkpoint(model_path, load_type):
    log_file = os.path.join(model_path, 'logging.jsonl')
    if not os.path.isfile(log_file):
        checkpoint = find_latest_checkpoint(model_path)

    else:
        with open(log_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        model_path_info = json.loads(lines[-1])
        last_model_checkpoint = model_path_info["last_model_checkpoint"]
        best_model_checkpoint = model_path_info["best_model_checkpoint"]
        if not best_model_checkpoint:
            best_model_checkpoint = last_model_checkpoint

        if load_type == "last":
            checkpoint = last_model_checkpoint
        if load_type == "best":
            checkpoint = best_model_checkpoint

    if os.path.isdir(f'{checkpoint}-merged'):  # LoRA合并后的目录
        checkpoint = f'{checkpoint}-merged'

    return checkpoint


def find_latest_checkpoint(model_path):
    cs = os.listdir(model_path)
    if not any(re.match(r'pytorch_model.*\.bin', f) or re.match(r'model.*\.safetensors', f) for f in cs):
        c2s = {}
        for c in cs:
            m = re.match(r'checkpoint-([0-9]+)$', c)
            if m:
                c2s[c] = int(m.group(1))
        if c2s:
            model_path = os.path.join(model_path, max(c2s, key=c2s.get))
            print(f'自动使用模型最新checkpoint：{model_path}')
    return model_path


def run_predict_multi_gpu(args):
    """
    用于多卡推理，跑文本生成任务。
    基于vllm，启动多个推理进程（predict_async.py），并通过统一的data_server实现数据的异步读取和写入，以及推理进程之间的负载均衡。
    """
    # ----------------------- 启动 data server（负责数据读写）---------------------------
    cmd_data_server = [
        'python', '{}'.format(os.path.join(cur_path, "data_server.py")),
        '--tokenizer', args.model,
        '--preprocess', args.preprocess,
        '--prompt', args.prompt,
        '--output_key', args.output_key,
        '--output_type', args.output_type,
        '--port', str(args.server_port),
    ]
    if args.data:
        cmd_data_server.extend(['--data', args.data, '--data_type', args.data_type])
    for k in ['output_dir', 'output_path', 'postprocess', 'max_length', 'stop', 'subsample', 'seed']:
        if v := getattr(args, k):
            cmd_data_server.extend([f'--{k}', str(v)])
    for k in ['reuse', 'overwrite', 'no_order']:
        if v := getattr(args, k):
            cmd_data_server.append(f'--{k}')
    print('Starting:', ' '.join(cmd_data_server))
    proc_data_server = Popen(cmd_data_server)  # , stdout=PIPE, stderr=PIPE)

    # -------------------------- 启动推理进程（多个）------------------------------------
    if args.device:
        gpus = args.device.split(',')
    else:
        ngpus = int(subprocess.run(  # all available gpus
            'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l', shell=True, stdout=subprocess.PIPE
        ).stdout)
        gpus = list(map(str, range(ngpus)))
    assert len(gpus) >= args.tensor_parallel, 'GPU数量不足，请确认GPU数量大于 tensor_parallel 数值！'

    proc_list = [proc_data_server]
    cmd_predict_list = []
    for i in range(len(gpus) // args.tensor_parallel):
        device = gpus[i * args.tensor_parallel : (i + 1) * args.tensor_parallel]
        if args.backend == 'vllm':
            predict_script = 'predict_async.py'
        elif args.backend == 'hf':
            predict_script = 'predict_async_hf.py'
        cmd_predict = [
            'python', '{}'.format(os.path.join(cur_path, predict_script)),
            '--server_addr', '127.0.0.1',
            '--server_port', str(args.server_port),
            '--model', args.model,
            '--output_type', args.output_type,
            '--max_new_tokens', str(args.max_new_tokens),
            '--temperature', str(args.temperature),
            '--presence_penalty', str(args.presence_penalty),
        ]
        if len(device) < len(gpus):
            cmd_predict.extend(['--device', ','.join(device)])
        if args.max_length:
            cmd_predict.extend(['--max_length', str(args.max_length)])
        if args.stop:
            cmd_predict.extend(['--stop', args.stop])
        if args.low_vram:
            cmd_predict.append('--low_vram')
        if args.manual_start:
            print('Manual start:', ' '.join(cmd_predict))
            cmd_predict_list.append(cmd_predict)
        else:
            print('Starting:', ' '.join(cmd_predict))
            proc_list.append(Popen(cmd_predict))  # , stdout=PIPE, stderr=PIPE))

    # -------------------------- 推理过程中，打印信息 -----------------------------------

    n_pred_0 = 0
    t0 = time.time()
    err = ''
    session = requests.Session()
    try:
        while True:
            time.sleep(args.log_interval)
            try:
                r = session.get(f'http://127.0.0.1:{args.server_port}/info')
            except requests.exceptions.ConnectionError:
                pass
            else:
                try:
                    r = r.json()
                except requests.exceptions.JSONDecodeError:
                    terminate_subprocess(proc_data_server)
                    print_subprocess_out_err(proc_data_server)
                    err = 'Data server 异常'
                    break
                samples_per_sec = r['n_pred'] / (r['t_curr'] - r['t_start'])
                tokens_per_sec = r['n_token'] / (r['t_curr'] - r['t_start'])
                s_time = f'[{datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")}] '
                if not args.run_forever and not r['n_pred']:
                    print(s_time + 'loading...')
                    if time.time() - t0 > 1800:  # 30分钟没有加载完成，OOM？
                        for proc in proc_list[1:]:
                            terminate_subprocess(proc)
                            print_subprocess_out_err(proc)
                        err = '30分钟没有加载完成，超时退出'
                        break
                if args.manual_start and cmd_predict_list and not r['n_pred']:
                    pass  # 推理进程可能还没启动，不打印日志
                else:
                    s_skipped = f', skipped {r["n_skip"]} samples >= max len' if r["n_skip"] else ''
                    print(
                        s_time + f'Finished {r["n_pred"]} samples '
                        f'({samples_per_sec:.1f} samples/s, {tokens_per_sec:.1f} tokens/s{s_skipped}), '
                        f'{r["n_done"]} files ({r["n_file"]} files in total)'
                    )
                    if not args.run_forever and n_pred_0 and r['n_pred'] == n_pred_0:
                        if time.time() - t0 > 600 and is_gpu_idle():  # 10分钟没有预测完成的样本，推理进程卡住不动？-> 检查GPU使用率
                            for proc in proc_list[1:]:
                                terminate_subprocess(proc)
                                print_subprocess_out_err(proc)
                            err = '超过10分钟没有预测完成的样本，且GPU使用率为0，超时退出'
                            break
                    else:
                        n_pred_0 = r['n_pred']
                        t0 = time.time()
                if r['post_start']:  # 下游发送启动信号（当manual_start为True时）
                    for cmd_predict in cmd_predict_list:
                        print('Starting:', ' '.join(cmd_predict))
                        proc_list.append(Popen(cmd_predict))
                    cmd_predict_list.clear()
                if r['terminated']:  # 下游任务已发送结束信号
                    time.sleep(5)  # 等待子进程自行退出
                    break
                if not args.run_forever and r['n_done'] == r['n_file']:  # 全部预测完成
                    print(s_time + '全部预测完成!')
                    try:
                        session.post(f'http://127.0.0.1:{args.server_port}/terminate')
                    except requests.exceptions.ConnectionError:
                        pass
                    time.sleep(5)  # 等待子进程自行退出
                    break
                if r.get('error'):
                    terminate_subprocess(proc_data_server)
                    print_subprocess_out_err(proc_data_server)
                    err = 'Data server 异常'
                    break
                for proc in proc_list[1:]:  # 推理进程异常退出
                    if proc.poll() is not None:  # poll() -> returncode
                        print_subprocess_out_err(proc)
                        err = '推理进程异常退出'
                if err:
                    break

            if proc_data_server.poll() is not None:  # terminated
                print_subprocess_out_err(proc_data_server)
                err = 'Data server 异常退出'
                break

    except KeyboardInterrupt:  # 提前退出
        pass

    # ---------------------------- 预测完成，结束子进程 ---------------------------------
    for proc in proc_list:
        terminate_subprocess(proc)

    if err:
        raise RuntimeError(err)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, help="Data file to predict")
    parser.add_argument("--data_type", type=str, default='json', choices=['json', 'dataset'], help="Iutput/Output data type")
    parser.add_argument("--preprocess", type=str, default='preprocess', help="Module that provides preprocessing function")
    parser.add_argument("--postprocess", type=str, default=None, help="Module that provides postprocessing function")
    parser.add_argument("--prompt", type=str, default='Hithink', help="Prompt type")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--output_key", type=str, default='predict_result', help="Output key")
    parser.add_argument("--output_type", type=str, default='text', choices=['text', 'reward', 'prompt_tokens'], help="Output type")
    parser.add_argument("--max_new_tokens", default=2048, help="Max number of tokens to generate")
    parser.add_argument("--max_input_tokens", type=int, default=None,
                        help="Max number of input tokens, longer inputs will be truncated")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--temperature", type=float, default=0., help="0.0 means greedy decoding")
    parser.add_argument("--presence_penalty", type=float, default=0., help="0.0 means no penalty")
    parser.add_argument("--stop", type=str, default=None, help="token/word at which generation will be stopped")
    parser.add_argument("--reuse", action='store_true',
                        help="If prediction file exists, reuse previous outputs and only predict samples with empty output")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite if output file exists. If not set, will skip finished samples")
    parser.add_argument("--no_order", action='store_true', help="Output will be written in different order than input")
    parser.add_argument("--data_server", type=str, default="data_server.py", help="Modify to use custom data server")
    parser.add_argument("--server_port", type=int, default=17888, help="Data server port")
    parser.add_argument("--backend", type=str, default='vllm', choices=['vllm', 'hf'], help="Inference backend")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Number of gpus per model/engine")
    parser.add_argument("--device", type=str, default=None, help="e.g. 0,1,2... If not specified, use all gpus by default")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    parser.add_argument("--log_interval", type=float, default=5., help="seconds between printed logs")
    parser.add_argument("--run_forever", action='store_true', help="If not set, program will quit when there are no data to predict")
    parser.add_argument("--manual_start", action='store_true', help="If set, workers (predict_async.py) will not start automatically")
    parser.add_argument("--load_type", type=str, default='last', choices=['last', 'best'], help="Load the latest model or the best performing model on the validation set")
    parser.add_argument("--subsample", type=float, help="proportion (0.0 - 1.0) or number (>= 1) of samples used")
    parser.add_argument("--seed", type=int, help="seed used for subsampling")
    args = parser.parse_args()

    args.model = find_checkpoint(args.model, args.load_type)
    print(f"model path is {args.model}...")
    if args.data:
        assert args.output_dir or args.output_path, '未指定输出路径！（--output_dir 或 --output_path）'
        if os.path.isdir(args.data):
            assert args.output_dir, '输入（--data）是目录，需要指定输出路径！（--output_dir ）'
        if args.output_dir:
            assert args.output_dir != args.data, '输入路径（--data）与输出路径（--output_dir）不能相同！'

    if args.prompt == 'llama3':
        args.stop = ','.join([args.stop, '<|eot_id|>']) if args.stop else '<|eot_id|>'

    run_predict_multi_gpu(args)
