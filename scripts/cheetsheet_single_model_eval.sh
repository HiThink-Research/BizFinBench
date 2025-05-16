# 先将模型启动为服务
cd ..
export EXTERNAL_API=gpt-4o
export MODEL_NAME=gpt-4o

python run.py --config "config.yaml" --model_name ${MODEL_NAME}