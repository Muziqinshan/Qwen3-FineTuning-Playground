# Qwen3 Fine-Tuning Playground

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/License-Apache_2.0-orange.svg" alt="License">
    <img src="https://img.shields.io/badge/Hugging_Face-Transformers-yellow" alt="Hugging Face">
    <img src="https://img.shields.io/badge/built_with-love-ff69b4.svg" alt="Built with Love">
</p>

一个包含了多种主流大模型微调方案的实战代码库，基于Qwen3系列模型，旨在提供清晰、专业、易于扩展的微调示例。

[English Version](README_EN.md)

---

## ✨ 项目特性

-   **多种微调方案**: 涵盖了从监督微调到强化学习的多种主流技术。
    -   **监督微调 (SFT)**: 全量微调与LoRA高效微调。
    -   **强化学习 (RL)**:
        -   **PPO**: 经典的基于奖励模型的强化学习。
        -   **ORPO**: 无需奖励模型的高效偏好对齐算法。
-   **训后优化技术**:
    -   **知识蒸馏**: 将大模型能力迁移到小模型。
-   **专业化代码结构**:
    -   **模块化设计**: 所有代码按功能清晰划分，易于理解和维护。
    -   **完全参数化**: 所有脚本均可通过命令行参数配置，无需修改代码即可运行。
    -   **详细教程**: 提供从0到1的示例教程，带你走完每个微调流程。

---

## 📂 项目结构

```
Qwen3-FineTuning-Playground/
├── 📂 Supervised_FineTuning/   # 监督微调 (SFT) 脚本
├── 📂 RL_FineTuning/           # 强化学习微调脚本
│   ├── 📂 PPO/
│   ├── 📂 ORPO/
│   └── 📂 GRPO/ (待实现)
├── 📂 Post_Training/           # 训后优化技术脚本
│   └── 📂 Distillation/
├── 📂 data/                     # 数据集和处理脚本
├── 📂 inference/                # 推理脚本
├── 📂 scripts/                  # 辅助脚本 (如合并权重)
├── 📂 evaluation/               # 评测脚本
├── 📂 example/                  # 详细的端到端教程文档
├── 📄 .gitignore
├── 📄 LICENSE                   # 开源协议
├── 📄 requirements.txt          # 项目依赖
└── 📄 README.md                  # 就是你正在看的这个文件
```

---

## 🚀 快速开始

下面将引导你快速跑通一个完整的SFT-LoRA微调流程。

### 1. 克隆并进入项目

```bash
git clone https://github.com/your-username/Qwen3-FineTuning-Playground.git
cd Qwen3-FineTuning-Playground
```

### 2. 配置环境

我们强烈建议使用 `conda` 创建一个独立的Python环境。

```bash
conda create -n qwen3_ft python=3.10
conda activate qwen3_ft
pip install -r requirements.txt
```

### 3. 准备模型和数据

-   **模型**: 本项目推荐使用 `modelscope` 库从魔搭社区下载模型。`requirements.txt` 已包含 `modelscope` 库。

    运行以下命令下载本项目所需的基础模型：

    ```bash
    # 下载Qwen3-1.7B (主要用于SFT, ORPO, PPO等微调)
    modelscope download --model Qwen/Qwen3-1.7B --local_dir ./Qwen3/Qwen3-1.7B

    # 下载Qwen3-4B (主要用作知识蒸馏的教师模型)
    modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen3/Qwen3-4B
    ```
    下载完成后，模型文件将分别位于 `./Qwen3/Qwen3-1.7B` 和 `./Qwen3/Qwen3-4B` 目录下。

-   **数据**: 本项目使用的数据格式已在 `data/` 目录中提供示例 `dirty_chinese_dpo.json`。

### 4. 开始SFT-LoRA微调

运行以下命令启动SFT训练。

```bash
python Supervised_FineTuning/train_sft_dirty.py \
    --model_path ./Qwen3/Qwen3-1.7B \
    --dataset_path data/dirty_chinese_dpo.json \
    --sft_adapter_output_dir ./output/sft_adapter_demo
```

训练完成后，LoRA适配器将保存在 `./output/sft_adapter_demo` 目录下。

### 5. 进行推理

使用我们刚刚训练好的LoRA适配器进行交互式聊天。

```bash
python inference/inference_dirty_sft.py \
    --model_path ./Qwen3/Qwen3-1.7B \
    --adapter_path ./output/sft_adapter_demo \
    --mode interactive
```

---

## 📚 详细教程

我们为每种主流的微调技术都提供了详细的端到端教程，请查阅 `example/` 目录下的文档：

-   **[SFT -> RM -> PPO 完整流程教程](./example/README.md)**
-   **[ORPO 单步高效对齐教程](./example/README_ORPO.md)**
-   **[知识蒸馏教程](./example/README_Distillation.md)**

---

## 🤝 贡献

欢迎任何形式的贡献！如果你有新的想法、修复了Bug或者想要添加新的微调方法，请随时提交Pull Request。

1.  Fork本仓库
2.  创建你的新分支 (`git checkout -b feature/YourAmazingFeature`)
3.  提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/YourAmazingFeature`)
5.  创建一个Pull Request

---

## 致谢

-   感谢 **Qwen Team** 提供了如此强大的开源模型。
-   感谢 **Hugging Face** 生态提供的 `transformers`, `peft`, `trl` 等优秀工具库。

---

## 开源协议

本项目采用 [Apache 2.0 license](./LICENSE) 开源协议。