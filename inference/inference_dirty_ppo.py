import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse
import os

class PPOChatbot:
    def __init__(self, sft_merged_model_path, ppo_adapter_path):
        self.device = self._get_device()
        self.sft_merged_model_path = sft_merged_model_path
        self.ppo_adapter_path = ppo_adapter_path
        self.system_prompt = "你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。"
        self.tokenizer = None
        self.model = None

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载PPO微调后的模型"""
        print("🚀 正在加载PPO模型...")
        print(f"--> 基础模型 (SFT合并后): {self.sft_merged_model_path}")
        print(f"--> PPO适配器: {self.ppo_adapter_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sft_merged_model_path, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_merged_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(
            self.model, model_id=self.ppo_adapter_path
        )
        self.model.eval()
        print(f"✅ PPO模型加载完成，使用设备: {self.model.device}")

    def generate_response(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
        """使用聊天模板生成回复"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

def test_model(chatbot: PPOChatbot, output_file: str):
    """批量测试PPO模型效果"""
    test_questions = [
        "你好，请介绍一下你自己", "请用礼貌的语言回答问题", "说没说不需说", "如何保持身体健康？",
        "什么是高血压？", "感冒了应该怎么办？", "请解释一下什么是人工智能",
        "你能帮我做什么？", "谢谢你的帮助", "再见"
    ]
    
    print("\n" + "="*80 + "\n🎯 PPO模型批量测试开始\n" + "="*80)
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 测试 {i}/{len(test_questions)}: {question}\n" + "-" * 60)
        response = chatbot.generate_response(question)
        print(f"🤖 回复: {response}")
        results.append({"question": question, "response": response})
        print("-" * 60)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ PPO测试完成！结果已保存到 {output_file}")

def interactive_chat(chatbot: PPOChatbot):
    """PPO模型交互式对话"""
    print("\n" + "="*80 + "\n🎯 PPO模型交互式对话\n" + "="*80)
    print("💡 输入 'exit' 或 'quit' 退出。")
    
    while True:
        try:
            user_input = input("\n👤 用户: ").strip()
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue
            print("🤖 助手: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
        except (KeyboardInterrupt, EOFError):
            break
    print("\n👋 再见！")

def main():
    parser = argparse.ArgumentParser(description="PPO模型推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="SFT合并后的基础模型的路径")
    parser.add_argument("--adapter_path", type=str, required=True, help="PPO LoRA适配器的路径 (例如 ./output/ppo_adapter)")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"], help="运行模式: 'interactive' (交互式) 或 'test' (批量测试)")
    parser.add_argument("--test_output_file", type=str, default="ppo_test_results.json", help="批量测试结果的输出文件路径")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌错误: 基础模型路径不存在: {args.model_path}")
        return
    if not os.path.exists(args.adapter_path):
        print(f"❌错误: 适配器路径不存在: {args.adapter_path}")
        return

    chatbot = PPOChatbot(args.model_path, args.adapter_path)
    chatbot.load_model()

    if args.mode == 'interactive':
        interactive_chat(chatbot)
    elif args.mode == 'test':
        test_model(chatbot, args.test_output_file)

if __name__ == "__main__":
    main()