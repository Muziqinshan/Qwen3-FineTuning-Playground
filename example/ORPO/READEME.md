# 示例教程：使用ORPO高效微调“脏话”助手

本教程将向你展示如何使用 **ORPO (Odd-Ratio Preference Optimization)** 算法来微调一个“脏话”聊天助手。

## 什么是ORPO？

ORPO是一种新颖、高效的对齐算法。与传统的RLHF流程（如PPO）相比，它有几个显著优势：

-   **一步到位**: ORPO在一个训练阶段同时完成了监督微调（SFT）和偏好对齐，无需多个训练步骤。
-   **无需奖励模型**: 它不需要预先训练一个独立的奖励模型（RM），大大简化了流程和计算开销。
-   **效率更高**: 由于流程简化，ORPO通常比PPO等方法训练得更快，需要更少的资源。

ORPO通过一个巧妙的损失函数，鼓励模型学习“chosen”（偏好）的回答，同时抑制“rejected”（不偏好）的回答，从而高效地将模型与人类偏好对齐。

---

## 准备工作

在开始之前，请确保你已经：

1.  **配置好环境**:
    ```bash
    # (假设你已在项目根目录)
    pip install -r requirements.txt
    ```
2.  **准备好基础模型**: 下载一个预训练的基础模型（如 `Qwen/Qwen2-1.5B-Instruct`）并放置在你的模型目录中。
3.  **准备好数据集**: 本教程使用的数据集是 `data/dirty_chinese_dpo.json`，其中包含了成对的“偏好”与“不偏好”的回答。

---

## 训练流程

ORPO的训练过程非常直接，只需一步即可完成。

**✅ 运行以下命令启动ORPO训练:**
*(请在项目根目录运行此命令)*

```bash
python RL_FineTuning/ORPO/train_lora_orpo_dirty.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --dataset_path data/dirty_chinese_dpo.json \
    --output_dir ./output/orpo_adapter \
    --learning_rate 8e-6 \
    --beta 0.1
```

-   `--model_path`: 你下载的基础模型路径。
-   `--dataset_path`: 包含 `prompt`, `chosen`, `rejected` 的数据集路径。
-   `--output_dir`: 训练产出的ORPO LoRA适配器将被保存在这里。
-   `--beta`: ORPO算法的关键超参数，用于平衡SFT损失和偏好对齐损失。默认值 `0.1` 通常是一个不错的起点。

**训练完成后，你会在 `output/orpo_adapter` 目录下看到最终的LoRA权重文件。**

---

## 推理与测试

训练完成后，我们可以直接加载基础模型和刚刚生成的ORPO适配器，来测试我们一步到位微调的成果。

**✅ 运行以下命令与ORPO模型进行交互式聊天:**
*(请在项目根目录运行此命令)*
```bash
python inference/inference_dirty_orpo.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --adapter_path ./output/orpo_adapter \
    --mode interactive
```

-   `--model_path`: 同样是你的基础模型路径。
-   `--adapter_path`: 上一步训练生成的ORPO适配器路径。
-   `--mode`: `interactive` 用于交互式聊天，也可以设置为 `test` 进行批量问题测试。

---

## 总结

恭喜！你已经学会了如何使用ORPO这一更简单、更高效的方法来完成模型的对齐微调。对于许多场景，ORPO都是替代复杂RLHF流程的绝佳选择。