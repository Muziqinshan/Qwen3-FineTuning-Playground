import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType, get_peft_model
from trl import ORPOConfig, ORPOTrainer
import swanlab
from dataclasses import dataclass, field

@dataclass
class ScriptArguments:
    """
    ORPO脚本的配置参数
    """
    model_path: str = field(metadata={"help": "模型仓库的路径"})
    dataset_path: str = field(default="data/dirty_chinese_dpo.json", metadata={"help": "数据集的路径"})
    output_dir: str = field(default="./output/orpo_adapter", metadata={"help": "ORPO LoRA适配器的保存目录"})
    system_prompt: str = field(default="你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。", metadata={"help": "系统提示语"})
    
    # LoRA配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})

    # ORPO配置
    learning_rate: float = field(default=8e-6, metadata={"help": "ORPO学习率"})
    beta: float = field(default=0.1, metadata={"help": "ORPO的beta超参数"})
    max_length: int = field(default=1024, metadata={"help": "输入最大长度"})
    max_prompt_length: int = field(default=512, metadata={"help": "最大提示长度"})
    
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab记录实验"})

def setup_swanlab(args: ScriptArguments):
    """配置并初始化SwanLab"""
    if not args.use_swanlab:
        return
    os.environ["SWANLAB_PROJECT"] = "qwen3-orpo-chinese"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    swanlab.init(
        project="qwen3-orpo-chinese",
        run_name="orpo-training-professional",
        config=vars(args)
    )

def load_and_process_dataset(dataset_path, tokenizer, system_prompt):
    """加载DPO数据集并构建适用于ORPO的 prompt, chosen, rejected 列"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌错误: 数据集文件未找到 at {dataset_path}")
        exit()
        
    processed_data = []
    for item in data:
        if 'conversations' in item and 'chosen' in item and 'rejected' in item:
            human_input = "".join([turn['value'] for turn in item['conversations'] if turn.get('from') == 'human']).strip()
            chosen_response = item['chosen'].get('value', '')
            rejected_response = item['rejected'].get('value', '')

            if human_input and chosen_response and rejected_response:
                # 使用Qwen3聊天模板构建prompt
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_input}
                ]
                prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                processed_data.append({
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })
    return Dataset.from_list(processed_data)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not os.path.exists(args.model_path):
        print(f"❌错误: 基础模型在 '{args.model_path}' 未找到。请检查路径。")
        exit()

    print("🚀 1. 配置和初始化 SwanLab...")
    setup_swanlab(args)

    print("🚀 2. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 3. 加载和预处理数据集...")
    full_dataset = load_and_process_dataset(args.dataset_path, tokenizer, args.system_prompt)
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
    
    print("🚀 4. 加载模型并配置LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("🚀 5. 配置ORPO训练参数...")
    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_length - args.max_prompt_length,
        beta=args.beta,
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        lr_scheduler_type="cosine", warmup_steps=50,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="swanlab" if args.use_swanlab else "none",
        run_name="orpo-training-run-professional",
    )
    
    print("🚀 6. 创建并启动ORPOTrainer...")
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    print(f"💾 7. 保存ORPO LoRA适配器到: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ ORPO训练完成！")
    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    main()