import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- 配置 ---
BASE_MODEL_PATH = "/home/space/space/model/Qwen3-1.7B"
SFT_ADAPTER_PATH = "./output/sft_adapter"
MERGED_MODEL_OUTPUT_PATH = "./output/sft_merged_model"

def main():
    """
    该脚本用于将SFT阶段训练好的LoRA适配器合并到基础模型中，
    并将其保存为一个独立的模型，以供后续的RM和PPO阶段使用。
    """
    print("🚀 开始 SFT 适配器合并...")

    if not os.path.exists(SFT_ADAPTER_PATH):
        raise FileNotFoundError(f"SFT 适配器未在 {SFT_ADAPTER_PATH} 找到。请先运行 train_sft.py。")

    # 1. 加载基础模型
    print(f"正在从以下路径加载基础模型: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # 在CPU上加载以避免合并时显存不足
        trust_remote_code=True,
    )

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 3. 加载 SFT LoRA 适配器
    print(f"正在从以下路径加载 SFT 适配器: {SFT_ADAPTER_PATH}")
    model_to_merge = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

    # 4. 调用 merge_and_unload 将适配器权重合并到基础模型
    print("正在将适配器合并到基础模型中...")
    merged_model = model_to_merge.merge_and_unload()
    print("合并完成。")

    # 5. 保存合并后的模型和 Tokenizer
    print(f"正在将合并后的模型保存到: {MERGED_MODEL_OUTPUT_PATH}")
    os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
    
    print("✅ 合并后的模型已成功保存！")

if __name__ == "__main__":
    main()