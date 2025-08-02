import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse
import os

class DistilledChatbot:
    def __init__(self, student_model_path, distilled_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student_model_path = student_model_path
        self.distilled_adapter_path = distilled_adapter_path
        self.system_prompt = "你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """加载蒸馏微调后的模型"""
        print("🚀 正在加载蒸馏模型...")
        print(f"--> 学生模型: {self.student_model_path}")
        print(f"--> 蒸馏适配器: {self.distilled_adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_path, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(
            self.model, model_id=self.distilled_adapter_path
        )
        self.model.eval()
        print(f"✅ 蒸馏模型加载完成，使用设备: {self.model.device}")

    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
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

def test_model(chatbot: DistilledChatbot, output_file: str):
    """批量测试蒸馏模型效果"""
    test_questions = [
        "怎么在甄嬛传里活过三集", "李时珍是谁？给我介绍一下他", "你的白月光和朱砂痣是谁", "三姓家奴说的是谁？",
        "给我写一首关于秋天的诗", "白羊座和蝎子座适合谈恋爱吗？", "一万个舍不得，只是不能再爱了",
        "给我讲两个关于李世民的功绩", "凤凰传奇什么时候开演唱会", "给你十万块钱你会做什么？"
    ]
    
    print("\n" + "="*80 + "\n🎯 蒸馏模型批量测试开始\n" + "="*80)
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 测试 {i}/{len(test_questions)}: {question}\n" + "-" * 60)
        response = chatbot.generate_response(question)
        print(f"🤖 回复: {response}")
        results.append({"question": question, "response": response})
        print("-" * 60)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 蒸馏测试完成！结果已保存到 {output_file}")

def interactive_chat(chatbot: DistilledChatbot):
    """蒸馏模型交互式对话"""
    print("\n" + "="*80 + "\n🎯 蒸馏模型交互式对话\n" + "="*80)
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
    parser = argparse.ArgumentParser(description="蒸馏模型推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="学生模型的路径 (例如 /path/to/Qwen3-1.7B)")
    parser.add_argument("--adapter_path", type=str, required=True, help="蒸馏后LoRA适配器的路径 (例如 ./output/distilled_adapter)")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"], help="运行模式: 'interactive' (交互式) 或 'test' (批量测试)")
    parser.add_argument("--test_output_file", type=str, default="distilled_test_results.json", help="批量测试结果的输出文件路径")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌错误: 学生模型路径不存在: {args.model_path}")
        return
    if not os.path.exists(args.adapter_path):
        print(f"❌错误: 适配器路径不存在: {args.adapter_path}")
        return

    chatbot = DistilledChatbot(args.model_path, args.adapter_path)
    chatbot.load_model()

    if args.mode == 'interactive':
        interactive_chat(chatbot)
    elif args.mode == 'test':
        test_model(chatbot, args.test_output_file)

if __name__ == "__main__":
    main()