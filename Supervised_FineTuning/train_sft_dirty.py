import json
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model
import swanlab
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    """
    SFT脚本的配置参数
    """
    model_path: str = field(metadata={"help": "模型仓库的路径"})
    dataset_path: str = field(default="data/dirty_chinese_dpo.json", metadata={"help": "数据集的路径"})
    sft_adapter_output_dir: str = field(default="./output/sft_adapter", metadata={"help": "SFT LoRA适配器的保存目录"})
    system_prompt: str = field(default="你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。", metadata={"help": "系统提示语"})
    max_length: int = field(default=1024, metadata={"help": "输入的最大长度"})
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab记录实验"})

def setup_swanlab(args: ScriptArguments):
    """配置并初始化SwanLab"""
    if not args.use_swanlab:
        return
    
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-rm-ppo-chinese"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    swanlab.init(
        project="qwen3-sft-rm-ppo-chinese",
        run_name="sft-training-professional",
        config={
            "model": args.model_path,
            "method": "SFT_with_Trainer",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "dataset": args.dataset_path,
            "system_prompt": args.system_prompt
        }
    )

def load_and_format_dataset(dataset_path, system_prompt):
    """
    加载DPO JSON文件，并将其转换为SFT的 instruction, input, output 格式.
    我们只使用 "chosen" 的回答进行监督微调。
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌错误: 数据集文件未找到 at {dataset_path}")
        exit()
    
    formatted_data = []
    for item in data:
        if 'conversations' in item and 'chosen' in item:
            human_input = "".join([turn['value'] + "\n" for turn in item['conversations'] if turn.get('from') == 'human']).strip()
            chosen_response = item['chosen'].get('value', '')
            
            if human_input and chosen_response:
                formatted_data.append({
                    "instruction": system_prompt,
                    "input": human_input,
                    "output": chosen_response
                })
    return formatted_data

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("🚀 1. 配置和初始化 SwanLab...")
    setup_swanlab(args)

    print("🚀 2. 加载和格式化数据集...")
    sft_data = load_and_format_dataset(args.dataset_path, args.system_prompt)
    full_dataset = Dataset.from_list(sft_data)
    
    # 使用 train_test_split 划分数据集
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"SFT训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    print("🚀 3. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        use_fast=False, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def process_func(example):
        instruction_part = tokenizer(
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response_part = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)
        
        input_ids = instruction_part["input_ids"] + response_part["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction_part["attention_mask"] + response_part["attention_mask"] + [1]
        labels = [-100] * len(instruction_part["input_ids"]) + response_part["input_ids"] + [tokenizer.eos_token_id]

        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            attention_mask = attention_mask[:args.max_length]
            labels = labels[:args.max_length]
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    print("🚀 4. 对数据集进行Tokenization...")
    tokenized_train_ds = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
    tokenized_eval_ds = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)
    
    print("🚀 5. 加载模型并配置LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("🚀 6. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./output/sft_model_temp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch", # 在每个epoch后进行评估
        gradient_checkpointing=True,
        report_to="swanlab" if args.use_swanlab else "none",
        run_name="sft-training-run-professional",
    )

    print("🚀 7. 创建并启动Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds, # 传入评估数据集
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    print(f"💾 8. 保存SFT LoRA适配器到: {args.sft_adapter_output_dir}")
    os.makedirs(args.sft_adapter_output_dir, exist_ok=True)
    trainer.save_model(args.sft_adapter_output_dir)
    
    print("\n✅ SFT训练完成！")
    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    main()