# 1 关于微调的核心认知

## 1.1为什么需要微调

大语言模型的训练通常包含预训练（Pre-training）和微调（Fine-tuning）两个主要阶段：

- **预训练**：赋予了模型广泛的通用知识和语言理解能力，但它缺乏针对特定任务的专业深度。
- **微调**：则是站在巨人的肩膀上，通过收集特定领域的优质数据集（如问答对、专业文档），对模型进行针对性训练，使其能够更好地服从人类指令，并在专业领域中表现得更加完美。

## 1.2微调为什么那么吃内存

在常规的全参数微调（Full Fine-Tuning）中，显存（内存）的消耗是极其惊人的。以混合精度（FP16）加上 AdamW 优化器来微调一个 7B（70亿参数）的大模型为例，显存占用主要来自以下四个方面：

1. **优化器状态（占比最大）**：AdamW 需要保存动量和方差等参数，通常是模型参数量的 2-4 倍。
2. **前向传播激活值（训练独有）**：用于反向传播计算梯度时使用。
3. **输入批次张量（Batch Data）**：输入的数据越长，显存占用越大。
4. **模型原始权重与梯度**。

**总结：要降低微调过程中的硬件门槛，最核心的思路就是**大幅减少微调过程中需要更新的参数量。更新的参数越少，优化器状态和梯度所占用的显存就会断崖式下降。

# 2 LoRA 方法的原理是什么？

LoRA（Low-Rank Adaptation，低秩微调）正是为了解决显存瓶颈而诞生的。

**比喻：假设原本的大模型是一本厚厚的百科全书。全参数微调相当于直接在书页上涂改、重印，成本极高。而 LoRA 相当于你在书页上贴了几张**透明的便利贴，你在便利贴上做笔记（微调参数）。阅读时，书本原文（原模型权重）加上便利贴上的笔记（LoRA 权重），就是最终的知识。

**数学原理：**

在神经网络中，全参数微调需要对原始的巨大权重矩阵$W_0$进行完整的增量更新，即 $W = W_0 + \Delta W$。

LoRA 巧妙地利用了线性代数中的“低秩”特性。它冻结了预训练模型的原始大矩阵 $W_0 \in \mathbb{R}^{d \times k}$，并通过两个小矩阵的乘积来近似表示增量矩阵 $\Delta W$：

$$
\Delta W = B \times A
$$

其中：

1. 矩阵 $B \in \mathbb{R}^{d \times r}$，矩阵 $A \in \mathbb{R}^{r \times k}$。
2. 这里的$r$就是秩（Rank），通常取值非常小（如 8、16 或 32），满足 $r \ll \min(d, k)$。

**训练过程：**

在训练时，原矩阵 $W_0$的参数保持不变，模型只对 $A$和 $B$这两个小矩阵进行梯度更新。前向传播时的计算变为：

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

通过这种方式，LoRA 将需要优化的参数量降低了几个数量级（通常只有原来的 0.1% ~ 1%），从而在单张消费级显卡（甚至 Mac 的统一内存）上实现了大模型的微调。

# 3 LoRA 微调实战代码

## 3.1 加载需要微调的模型和分词器

首先，我们需要加载之前下载到本地的开源大模型。任务如下：

- 设置好本地模型的存放位置，以及微调后模型的保存位置
- 加载本地模型与分词器

```Python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 基础配置
model_path = "./model"  # 你之前下载的本地模型路径
output_dir = "./lora_results" # 微调后 LoRA 权重的保存位置

print("1. 正在加载模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 微调时通常需要 pad_token 来对齐数据长度。如果模型没有，指定 eos_token 为 pad_token
# 注意：微调时通常需要 pad_token。如果模型没有，我们指定 eos_token 为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型，这里我们为了节省内存，可以加载为 16 位浮点数 (bfloat16)
# device_map="auto" 会自动将模型分配到可用的计算设备（如 Mac 的 mps，或 Nvidia 的 cuda）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # 自动分配到 mps/cuda/cpu
)
```

## 3.2 数据处理与格式化

我们需要构建指令微调的数据集。假设我们有一个`dataset.json`，内容如下：

```JSON
[
  {
    "instruction": "请简单解释什么是微网？",
    "input": "",
    "output": "微网（Microgrid）是一个由分布式电源、储能装置、能量转换装置、负荷、监控和保护装置汇集而成的小型发配电系统，能够实现自我控制、保护和管理。"
  },
  {
    "instruction": "强化学习可以应用在微网的能源治理中吗？",
    "input": "",
    "output": "绝对可以。强化学习（如 Q-Learning 或 DDPG 算法）可以通过与微网环境的不断交互，学习到最优的充放电和能源调度策略，从而解锁微网的智能能源时代。"
  },
  {
    "instruction": "向我的学生解释什么是大模型的幻觉。",
    "input": "",
    "output": "大模型的幻觉是指模型生成了看似合理、连贯，但实际上与客观事实不符或完全虚构的内容。这就像一个学生在考试时遇到不会的题，为了不交白卷而“一本正经地胡说八道”。"
  }
]
```

**这里需要解释一下这个数据的结构：**

在 `dataset.json` 中看到的 `[instruction, input, output]` 格式，在业界被称为 **Alpaca 格式**（由斯坦福大学提出，广泛应用于 Llama、DeepSeek 等模型的微调）。

这三个字段的分工非常明确：

- `instruction` (指令): 告诉模型“你要做什么任务”。
- `input` (输入/上下文): 提供完成这个任务所需的“补充信息或原始素材”。
- `output` (输出): 模型应该给出的标准答案。

**什么情况下 ****`input`**** 为空？**

当任务是“自包含的（Self-contained）”时，`input` 就可以为空。

比如上一轮我们的例子：

> **Instruction:** 请简单解释什么是微网？
**Input:** `""` （空，因为问题本身已经完整表达了需求，不需要额外材料）
**Output:** 微网是一个由分布式电源……

什么情况下 `input` 必须有内容？

当任务需要依赖外部文本进行处理（如总结、翻译、信息抽取）时。

比如：

> **Instruction:** 请提取以下文本中的核心能源类型。
**Input:** 本次微电网项目主要接入了 500kW 的屋顶光伏以及两台 100kW 的风力发电机，同时配备了锂电池储能。
**Output:** 光伏、风能。

简而言之，`input` 是用来装“被处理的素材”的。如果没有素材要处理，只是单纯的问答，它就是空的。

将数据格式化为模型容易理解的对话模板，任务如下：

- 加载数据
- 将数据中的 Instruction 和 output 数据拼接为对话模板数据，方便模型读取学习

```Python
# 加载刚刚创建的 json 数据
dataset = load_dataset("json", data_files="dataset.json", split="train")

# 定义一个将 instruction 和 output 拼接为对话模板的函数
def format_prompt(example):
    # 这里模拟模型的对话格式
    # 构建对话上下文，末尾加上 eos_token 告诉模型回答结束
    text = f"User: {example['instruction']}\nAssistant: {example['output']}{tokenizer.eos_token}"
    return {"text": text}

# 映射数据集
formatted_dataset = dataset.map(format_prompt)
```

## 3.3 设置 LoRA 参数，注入 LoRA “便利贴”

这一步是核心！我们将普通模型包装成支持 LoRA 训练的 PEFT（Parameter-Efficient Fine-Tuning）模型。任务如下：

- 设置 LoRA 参数
- 将普通模型包装成支持 LoRA 训练的模型

```Python
# 【关键学习点】这就是 LoRA 的魔法所在！我们不训练模型原本的上百亿参数，而是插入几个小矩阵。
lora_config = LoraConfig(
    r=8,               # Rank (秩)：决定了新增矩阵的大小。值越大，模型能学到的细节越多，但内存消耗也越大。通常 8, 16, 32 都可以。
    lora_alpha=16,     # 缩放系数：控制 LoRA 权重对原模型的影响力度，通常设为 r 的 2 倍。
    target_modules=["q_proj", "v_proj"], # 要在哪些层应用 LoRA？通常在注意力机制的 Query 和 Value 矩阵上效果最好。
    lora_dropout=0.05, # 防过拟合：随机丢弃 5% 的神经元。
    bias="none",
    task_type="CAUSAL_LM" # 任务类型：因果语言建模（即文本生成）
)

# 将普通模型包装成支持 LoRA 训练的模型
model = get_peft_model(model, lora_config)
# 可以通过 model.print_trainable_parameters() 查看，你会发现可训练参数通常不到 1%

```

## 3.4 设置训练参数，开始训练

利用`trl`库中的`SFTTrainer`，可以极大简化我们的训练循环逻辑。任务如下：

- 设计训练参数
- 初始化 SFTTrainer开始微调

```Python
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 每次处理的数据量，如果内存/显存爆了，就把这个调成 1
    gradient_accumulation_steps=2,  # 梯度累加，变相增加 Batch Size，节省内存
    learning_rate=2e-4,  # 学习率
    num_train_epochs=3,  # 训练轮数：把数据集看几遍
    logging_steps=1,  # 每隔多少步打印一次日志
    save_strategy="epoch",  # 保存策略
    optim="adamw_torch",
    dataset_text_field="text", # 指定数据集中作为训练文本的列名
)


# SFTTrainer 会帮我们自动处理文本的 Tokenize 和 Padding，非常省心
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
#     max_seq_length=512,
    args=training_args,
)

# 开始训练！
print("开始注入领域知识...")
trainer.train()

# 保存模型和分词器（注意：这里只保存了极小的 LoRA 权重，即“便利贴”）
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"训练完成！LoRA 权重已保存至 {output_dir}")

```

# 4 微调后模型的推理与测试

需要注意的是，第 3 步保存下来的仅仅是 LoRA 权重（补丁）。在实际使用时，我们需要将“原版百科全书”（基础模型）和“笔记便利贴”（LoRA 权重）进行组装融合。

## 4.1 加载基础模型

```Python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 定义两个路径
base_model_path = "./model"         # 你最初下载的、几十亿参数的基础大模型路径
lora_weights_path = "./lora_results" # 你刚才微调保存的 "便利贴" (补丁) 路径

print("1. 正在加载基础模型和分词器 (搬出百科全书)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # 自动调用 Mac 的 mps 加速
)
```

## 4.2 融合微调权重

```Python
print("2. 正在融合微调权重 (贴上便利贴)...")
# 【核心魔法】将基础模型和 LoRA 补丁组装在一起
model = PeftModel.from_pretrained(base_model, lora_weights_path)
print("模型融合完毕！\n")
```

## 4.3 写一个连续对话函数来进行测试

```Python
rint("=" * 50)
print("欢迎测试微调后的专属助手！输入 'quit' 结束。")
print("=" * 50)

# 3. 开始连续对话测试
while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        break
    if not user_input:
        continue

    # 【避坑指南】微调后的推理，必须严格使用训练时的对话格式！
    # 回顾我们刚才在 SFTTrainer 里写的格式：f"User: {instruction}\nAssistant: "
    prompt = f"User: {user_input}\nAssistant: "

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # 模型推理生成
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3  # 降低温度，让模型回答更严谨
    )

    # 切片截取新生成的 Token
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 过滤可能包含的思维链标签（针对特定推理模型）
    if "</think>" in response:
        clean_response = response.split("</think>")[-1].strip()
    else:
        clean_response = response.strip()
    # -----------------------

    print(f"\nAssistant: {clean_response}")
```

# 5 将补丁和基础模型合并，生成一个新的模型进行保存

需要将之前的“补丁”和“底座”融为一体，生成一个新的、完整的模型文件夹。

```Python
base_model_path = "./model/" # 原始底座，如 Qwen, Llama
sft_lora_path = "./lora_results/" # 你之前的 SFT 结果

# 1. 加载底座
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. 挂载 SFT LoRA
model = PeftModel.from_pretrained(base_model, sft_lora_path)

# 3. 合并并卸载 (核心步骤)
# 这会将 LoRA 的权重永久性地加到底座权重中
merged_model = model.merge_and_unload()

# 4. 保存为新的模型
merged_model.save_pretrained("./model_merged_for_rm")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained("./model_merged_for_rm")
```
[代码链接](https://github.com/kxmust/LLM_stu/blob/main/02%20lora_fine_tuning.ipynb)