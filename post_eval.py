from importlib.util import spec_from_file_location, module_from_spec
import sys
import argparse
import json
from loguru import logger
import os
cur_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_func", type=str, help="Path of eval function")
    parser.add_argument("--input_path", type=str, help="input_path for eval file")
    parser.add_argument("--output_path", type=str, help="path of output eval result")
    parser.add_argument("--statistic_func",default=None, type=str, help="path of output eval result")
    parser.add_argument("--kwargs",default=None, type=str, help="Addtional eval args")

    args = parser.parse_args()
    # 导入评估文件
    spec = spec_from_file_location("eval", os.path.join(cur_path,args.eval_func))
    module = module_from_spec(spec)
    sys.modules["eval"] = module
    spec.loader.exec_module(module)

    logger.info(args)
    
    if args.kwargs:
        kwargs = json.loads(args.kwargs)
        logger.info("loading kwargs {}".format(kwargs))
        result = module.evaluation(args.input_path,**kwargs)
        
    else:
        result = module.evaluation(args.input_path)
    # import pdb; pdb.set_trace()
    if args.statistic_func:
        spec = spec_from_file_location("statistic", args.statistic_func)
        statistic = module_from_spec(spec)
        sys.modules["statistic"] = statistic
        spec.loader.exec_module(statistic)
        result = statistic.statistic(result)
    to_save = [{"Average":result}]
    with open(args.output_path,'w',encoding='utf-8') as f:
        json.dump(to_save,f)
