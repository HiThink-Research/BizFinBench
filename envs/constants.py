import sys, os
from tools.ExternalApi import GPTClient
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
api_name = os.getenv('API_NAME')
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME')

ExternalApi = {
    'chatgpt':GPTClient(api_name=api_name,api_key=api_key,model_name=model_name,base_url='https://api.openai.com/v1/chat/completions', timeout=600, semaphore_limit=5)
}