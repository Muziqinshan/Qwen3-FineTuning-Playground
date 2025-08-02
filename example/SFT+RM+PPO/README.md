# 示例教程：如何训练一个“脏话”助手 (SFT -> RM -> PPO)

本教程将一步步带你走完一个完整的、基于人类反馈的强化学习（RLHF）流程，最终训练出一个风格独特的“脏话”聊天助手。

我们将使用以下流水线：
1.  **SFT (Supervised Fine-Tuning)**: 对基础模型进行监督微调，使其初步具备“脏话”风格和对话能力。
2.  **RM (Reward Modeling)**: 训练一个奖励模型，让它学会分辨哪些回答更符合“脏话”风格。
3.  **PPO (Proximal Policy Optimization)**: 使用奖励模型作为“裁判”，通过强化学习进一步优化SFT模型，使其生成的回答能获得更高的奖励分数。

---

## 准备工作

在开始之前，请确保你已经：

1.  **配置好环境**:
    ```bash
    pip install -r ../requirements.txt
    ```
2.  **准备好基础模型**: 下载一个预训练的基础模型（如 `Qwen/Qwen2-1.5B-Instruct`）并放置在你的模型目录中。
3.  **准备好数据集**: 本教程使用的数据集是 `data/dirty_chinese_dpo.json`。确保这个文件存在于项目根目录的`data`文件夹下。

---

## 训练流程

### 第 1 步：监督微调 (SFT)

SFT是第一步，也是最关键的一步。它为模型注入了我们想要的风格和知识的基础。

**✅ 运行以下命令启动SFT训练:**
*(请在项目根目录运行此命令)*
```bash
python Supervised_FineTuning/train_sft_dirty.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --dataset_path data/dirty_chinese_dpo.json \
    --sft_adapter_output_dir ./output/sft_adapter
```

-   `--model_path`: 你下载的基础模型路径。
-   `--sft_adapter_output_dir`: 训练产出的SFT LoRA适配器将被保存在这里。

**完成后，你会在 `output/sft_adapter` 目录下看到LoRA权重文件。**

### 第 2 步：合并SFT LoRA适配器

为了方便后续RM和PPO阶段的训练，我们将第一步产出的LoRA适配器合并到基础模型中，生成一个新的、完整的SFT模型。

**✅ 运行以下命令合并权重:**
*(请在项目根目录运行此命令)*
```bash
python scripts/merge_lora_weights.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --adapter_path ./output/sft_adapter \
    --output_path ./output/sft_merged_model
```
- `--output_path`: 合并后完整模型的保存路径。**这个路径在后续步骤中至关重要！**

**完成后，你会在 `output/sft_merged_model` 目录下看到一个完整的模型。**

### 第 3 步：训练奖励模型 (RM)

现在，我们基于合并后的SFT模型来训练奖励模型。这个模型将学会为“更脏”的回答打高分，为“不够脏”的回答打低分。

**✅ 运行以下命令启动RM训练:**
*(请在项目根目录运行此命令)*
```bash
python RL_FineTuning/RM/train_rm_dirty.py \
    --model_path ./output/sft_merged_model \
    --dataset_path data/dirty_chinese_dpo.json \
    --rm_adapter_output_dir ./output/rm_adapter
```
- `--model_path`: **注意！** 这里使用的是上一步合并后的SFT模型路径。
- `--rm_adapter_output_dir`: 奖励模型的LoRA适配器将被保存在这里。

**完成后，你会在 `output/rm_adapter` 目录下看到奖励模型的LoRA权重。**

### 第 4 步：PPO强化学习训练

这是最后一步，也是见证奇迹的时刻。我们将使用SFT模型作为基础策略，用奖励模型作为环境反馈，通过PPO算法进行“实战演练”，让模型学会如何生成能讨好奖励模型的回答。

**✅ 运行以下命令启动PPO训练:**
*(请在项目根目录运行此命令)*
```bash
python RL_FineTuning/PPO/train_ppo_dirty.py \
    --model_path ./output/sft_merged_model \
    --rm_adapter_path ./output/rm_adapter \
    --ppo_adapter_output_dir ./output/ppo_adapter
```
- `--model_path`: 同样使用SFT合并后的模型。
- `--rm_adapter_path`: 上一步训练出的奖励模型适配器路径。
- `--ppo_adapter_output_dir`: 最终PPO模型的LoRA适配器保存路径。

**完成后，你会在 `output/ppo_adapter` 目录下得到最终模型的LoRA权重！**

---

## 推理与测试

现在，我们可以和我们亲手训练的“脏话”助手聊天了！

### 测试SFT模型

你可以先试试只经过SFT微调的模型效果。
*(请在项目根目录运行此命令)*
```bash
python inference/inference_dirty_sft.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --adapter_path ./output/sft_adapter
```

### 测试最终的PPO模型

对比一下经过PPO强化学习后的模型，它的回答风格应该会更加稳定和突出。
*(请在项目根目录运行此命令)*
```bash
python inference/inference_dirty_ppo.py \
    --model_path ./output/sft_merged_model \
    --adapter_path ./output/ppo_adapter
```
- **注意**: PPO推理的基础模型是SFT合并后的模型。

---

恭喜你！你已经成功走完了一整套RLHF流程。快去和你的AI“伙伴”聊聊天吧！