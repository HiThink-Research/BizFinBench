export MODEL_PATH=model/Qwen2.5-0.5B   # Path to the model to be evaluated
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template   # Hithink llama3 llama2 none qwen chat_template; chat_template is recommended

# First start the model as a service
python inference/predict_multi_gpu.py \
    --model ${MODEL_PATH} \
    --server_port ${REMOTE_MODEL_PORT} \
    --prompt ${PROMPT_TYPE} \
    --preprocess preprocess \
    --run_forever \
    --max_new_tokens 4096 \
    --tensor_parallel ${TENSOR_PARALLEL} & 

# Pass in the config file path to start evaluation
python run.py --config config/offical/eval_fin_eval_diamond.yaml --model_name ${MODEL_NAME}