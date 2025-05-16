import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def merge_and_save_model(model_path):
    # 从预训练模型路径加载PEFT模型
    peft_model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    print(type(peft_model))  # 输出peft_model的类型

    # 合并适配器并卸载
    merged_model = peft_model.merge_and_unload()
    print(type(merged_model))  # 输出合并后的模型类型

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 保存合并后的模型和分词器
    merged_model.save_pretrained(model_path + '_merged')
    tokenizer.save_pretrained(model_path + '_merged')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_lora.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    merge_and_save_model(model_path)
