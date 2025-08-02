# 示例教程：使用知识蒸馏迁移“脏话”能力

本教程将介绍如何使用**知识蒸馏（Knowledge Distillation）**技术，将一个大型、强大的“教师模型”所具备的“脏话”能力，高效地迁移到一个更小、更轻量的“学生模型”上。

## 什么是知识蒸馏？

知识蒸馏是一种模型压缩技术，其核心思想是：让一个小型学生模型（Student Model）去模仿一个大型教师模型（Teacher Model）的行为，而不是仅仅从数据标签中学习。

-   **教师模型 (Teacher)**: 通常是一个参数量巨大、能力很强的模型（例如Qwen2-7B）。我们假设它已经具备我们想要的能力（或者我们可以通过prompt engineering引导它表现出这种能力）。
-   **学生模型 (Student)**: 一个参数量较小的模型（例如Qwen2-1.5B），我们希望它能用更少的计算资源达到接近教师模型的效果。

训练时，学生模型不仅要学习拟合真实的答案（标准的SFT损失），还要学习模仿教师模型输出的概率分布（KL散度损失）。通过这种“软标签”的学习，学生模型能学到教师模型更丰富的知识和推理模式。

## 训练流程

知识蒸馏的训练在一个脚本中即可完成。

**✅ 运行以下命令启动知识蒸馏训练:**
*(请在项目根目录运行此命令)*
```bash
python Post_Training/Distillation/distill_foul_mouthed.py \
    --teacher_model_path /path/to/your/Qwen2-7B-Instruct \
    --student_model_path /path/to/your/Qwen2-1.5B-Instruct \
    --dataset_path data/dirty_chinese_dpo.json \
    --output_dir ./output/distilled_adapter \
    --alpha 0.7 \
    --temperature 2.5
```

**关键参数说明:**

-   `--teacher_model_path`: **教师模型**的路径（例如一个7B模型）。
-   `--student_model_path`: **学生模型**的路径（例如一个1.5B模型）。
-   `--output_dir`: 训练产出的LoRA适配器将被保存在这里，它属于**学生模型**。
-   `--alpha`: 蒸馏损失的权重。值越高，学生模型越倾向于模仿老师；值越低，越倾向于拟合标准答案。`0.5`到`0.9`是常用范围。
-   `--temperature`: 蒸馏温度。用于平滑教师模型的输出，让学生能学到更“软”的知识。通常大于1，例如`2.0`或`2.5`。

**训练完成后，你会在 `output/distilled_adapter` 目录下看到最终学生模型的LoRA权重。**

---

## 推理与测试

推理时，我们只需要加载**学生模型**和它专属的蒸馏LoRA适配器。

**✅ 运行以下命令与蒸馏后的学生模型进行交互:**
*(请在项目根目录运行此命令)*
```bash
python inference/inference_dirty_distilled.py \
    --model_path /path/to/your/Qwen2-1.5B-Instruct \
    --adapter_path ./output/distilled_adapter \
    --mode interactive
```
-   `--model_path`: **学生模型**的路径。
-   `--adapter_path`: 上一步训练生成的蒸馏适配器路径。

---

## 总结

通过知识蒸馏，我们成功地用一个更小的模型实现了原本需要大模型才能达到的特定风格能力，这在需要兼顾效果与部署成本的场景中非常有用。