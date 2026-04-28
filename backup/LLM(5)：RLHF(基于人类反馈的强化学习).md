从提示词工程（怎么用）、本地部署（怎么跑）、数据合成（怎么造数据）、LoRA 微调（怎么教新知识），一直到最后的Reinforcement Learning from Human Feedback，RLHF（怎么立规矩），这正是目前业界训练顶级大模型（如 ChatGPT, Llama 3）的标准范式。

RLHF 相当于给模型“做礼仪培训+价值观调教”。在 RLHF 中，大模型本身就是那个“智能体（Agent）”，我们用强化学习算法（主要是 GRPO）来优化它的输出策略。

其核心内容可以分为三步：

- **人类打分**：对模型回答排好坏（哪个更有用、更安全、更礼貌）。
- **训练奖励模型（RM）**：让模型学会 “什么是好回答”。
- **强化学习（GRPO）**：让模型不断优化，往人类喜欢的方向调整。

输出结果：**模型更安全、更有用、更符合人类偏好，不乱说、不冒犯、更贴心。**

# 前提条件

LLM 的训练流程如下：

1. **预训练**：让模型**学会知识**
2. **SFT**：让模型**学会对话**
3. **RLHF**：让模型**学会好好说话、说人话**

在进入 RLHF 之前，我们假设模型已经通过 LoRA 微调 学习了基于 self-instruct 合成的问答数据。此时的模型具备了基本的对话能力，我们称之为 SFT 模型（Policy Model 的初始状态）。

# 第一步：人类打分

**核心目的**：收集数据，告诉机器“人类更偏好什么样的回答”。

**具体过程**：

1. 从题库中抽一个提示词（Prompt）。
2. 让当前的 SFT 模型生成几个不同的回答。
3. 人类（或更强大的大模型，即 AI Feedback）根据有用性、安全性、礼貌程度，对这些回答进行排序（Chosen vs Rejected）。

数据结构示例：

```JSON
[
  {
    "prompt": "老师，请问什么是梯度下降？",
    "chosen": "梯度下降是一种优化算法，就像下山一样，通过计算损失函数的导数，一步步找到最低点。",
    "rejected": "就是求导找极小值，这你都不知道吗？"
  },
  {
    "prompt": "写一段 Python 冒泡排序。",
    "chosen": "好的，这是一种基础的排序算法，代码如下：...",
    "rejected": "去百度搜一下就知道了，干嘛问我。"
  },
  {
    "prompt": "如何评价一部电影的好坏？",
    "chosen": "评价一部电影可以从多个维度进行，例如剧情连贯性、演员演技、镜头语言和配乐等。",
    "rejected": "好就是好，不好就是不好，哪有那么多废话。"
  }
]
```

# 第二步：训练奖励模型（RM）

**核心目的**：训练一个“裁判模型”，让它学会人类的打分标准。因为在强化学习阶段，我们需要实时给大模型的成千上万次回答打分，不可能全靠人工。

**具体过程**：
我们将 SFT 模型的输出层去掉，换成一个线性回归层（输出一个标量分数）。我们输入`(Prompt, Response)`，模型输出一个分数$R$。

它的损失函数基于 Bradley-Terry 模型，目标是让好回答的分数远大于坏回答的分数：

$$
Loss = -\log(\sigma(R(chosen) - R(rejected)))
$$

好回答的分数越大，Loss 会越小。

我们使用 Trl 库中的RewardTrainer来训练奖励模型

## 1 导入并处理数据集

```Python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# 1. 导入并处理数据集
# 导入数据集
raw_dataset = load_dataset("json", data_files='RLHF_dataset.json')

# 将数据集切分为训练集 (90%) 和验证集 (10%)，用于监控模型是否真正学会了打分
split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)

# 对数据做字符串的拼接！
def format_dataset(examples):
    formatted = {
        "chosen": [],
        "rejected": []
    }
    for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        formatted["chosen"].append(f"User: {p}\nAssistant: {c}")
        formatted["rejected"].append(f"User: {p}\nAssistant: {r}")
    return formatted

# 使用 map 覆盖原有的 chosen 和 rejected 列
formatted_dataset = split_dataset.map(
    format_dataset,
    batched=True  # 一次性处理一批数据
)
```

可以看一下第一条训练数据（formatted_dataset['train'][0]）：

```Python
{'prompt': '写一段 Python 冒泡排序。',
 'chosen': 'User: 写一段 Python 冒泡排序。\nAssistant: 好的，这是一种基础的排序算法，代码如下：...',
 'rejected': 'User: 写一段 Python 冒泡排序。\nAssistant: 去百度搜一下就知道了，干嘛问我。'}
```

## 2 导入模型，替换输出层，导入 LoRA 层（生成待训练的奖励模型）

```Python
# 2. 模型加载，引入 LoRA
model_path = "./model_merged_for_rm" # 之前SFT微调后的模型（和 LoRA 融合后的模型）

# 导入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 注意：微调时通常需要 pad_token。如果模型没有，我们指定 eos_token 为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型并替换分类头 (替换输出头，从生成者变为裁判)
print(">>> 正在加载底座模型并初始化分类头...")
# num_labels=1: 丢弃原有的 lm_head，换上一个输出维度为 1 的 score 线性层
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,  # 最后一层输出类别数量，输出维度为1
    torch_dtype=torch.bfloat16, # 推荐使用 bfloat16 节省显存
    device_map="auto"           # 自动分配显存
)

# 确保模型的 pad_token_id 与 tokenizer 一致（忽略填充符号 ID，和分词器用的必须完全一样）
model.config.pad_token_id = tokenizer.pad_token_id

# 挂载第二层 LoRA 补丁 (专门用于学习打分)
print(">>> 正在配置 RM 专属 LoRA...")
rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # 覆盖主要注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS", # 关键：告诉 PEFT 这是一个序列分类任务
)
model = get_peft_model(model, rm_lora_config)
model.print_trainable_parameters() # 打印可训练参数比例

```

输出结果：
trainable params: 2,180,608 || all params: 1,545,896,448 || trainable%: 0.1411

## 3 利用RewardTrainer训练奖励模型

```Python
# 3. 配置并启动 RewardTrainer
training_args = RewardConfig(
    output_dir="./RLHF_RM_Lora",   # Lora层输出的文件夹
    per_device_train_batch_size=2,  # 每台设备一次处理的数量量
    learning_rate=1e-5,
    max_length=512, # 截断长度在这里设置！
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer, 
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"], 
)
# 开始训练！
trainer.train()

# 7. 保存最终的 RM LoRA 补丁
print(f">>> 训练完成，正在保存 Reward Model LoRA 到 RLHF_RM")
trainer.save_model("./RLHF_RM_Lora")
tokenizer.save_pretrained("./RLHF_RM_Lora")
```

## 4 将 LoRA 补丁与模型进行合并，生成最后的 RM 模型

```Python
# 4. 合并补丁，生成 RM 模型
# 1. 定义路径
rm_base_path = "./model_merged_for_rm" # 训练 RM 时用的底座
rm_lora_path = "./RLHF_RM_Lora"           # 刚刚训练出来的 RM LoRA 补丁

# 2. 加载 RM 的底座 (注意是分类模型类，且带上 num_labels=1)
print(">>> 正在加载 RM 底座...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    rm_base_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="cpu" # 合并操作建议在 CPU 上进行，防止显存 OOM
)

# 3. 挂载 RM 的 LoRA 补丁
print(">>> 正在挂载 RM LoRA...")
model = PeftModel.from_pretrained(base_model, rm_lora_path)

# 4. 执行物理合并
print(">>> 正在合并权重 (这可能需要几分钟)...")
merged_rm_model = model.merge_and_unload()

# 5. 保存最终的独立裁判模型
final_rm_path = "./final_reward_model_merged"
print(f">>> 保存最终 RM 模型到 {final_rm_path}")
merged_rm_model.save_pretrained(final_rm_path)

# 保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(rm_base_path)
tokenizer.save_pretrained(final_rm_path)
print("✅ RM 合并完成！准备就绪！")
```

# 第三步：强化学习

这次用到的是 GRPO 算法，全称是**Group Relative Policy Optimization**（群体相对策略优化），是 DeepSeek 团队 2024 年提出的强化学习算法，专门为大语言模型设计，属于**PPO 的改进版本**，被归类为 RLHF（从人类反馈中强化学习）技术栈的核心组成部分。

GRPO 的伪代码如下：


```Markdown
# GRPO = 精简版强化学习：只留 Actor + 砍掉 Critic
while 训练未结束:
    # 1. 【组生成】一个提示词，生成 N 个回答（比如 8 个）
    一组回答 = 模型生成(提示词, 生成8个)
    
    # 2. 【直接打分】奖励模型/规则 给这8个回答分别打分
    一组奖励 = [奖励函数(回答1), 奖励函数(回答2)...]
    
    # 3. 【核心创新】组内归一化 = 自动算优势值（砍掉 Critic！）
    优势值 = (当前奖励 - 组平均分) / 组标准差
    
    # 4. 直接更新模型（极简，无裁剪、无Critic）
    更新模型(模型, 优势值)
```

相比于 PPO 的优化：

- **彻底删掉 Critic** → 显存减半、速度翻倍、不报错
- 用**组内对比**替代价值网络 → 数学更简单、训练更稳
- 代码极短，`trainer.train()` 一键跑完

## 1 基础配置和路径

```Python
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import set_seed
from datasets import Dataset

# 设置随机种子保证实验可复现
set_seed(42)

# 1. 配置路径与超参数
SFT_MODEL_PATH = "./model_merged_for_rm"       # 之前合并好的 SFT 模型（作为 Actor/Ref 的基座）
RM_MODEL_PATH = "./final_reward_model_merged" # 刚才合并出来的独立裁判模型
PROMPT_DATASET = "RLHF_dataset.json"         # 强化学习阶段不需要 chosen/rejected，只需要 prompt 即可
   
# 2. 加载 Tokenizer
print(">>> 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)


```

## 2 加载之前训练好的奖励模型用来打分

```Python
# 3. 加载 Reward Model
print(">>> 正在加载 Reward Model...")
rm_model = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL_PATH, 
    num_labels=1, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
rm_model.eval()

tokenizer.pad_token = tokenizer.eos_token
rm_model.config.pad_token_id = tokenizer.pad_token_id

# 4. 定义 Reward 函数
# 作用：给模型生成的回答“打分”，分数越高代表回答越好
# GRPO 会根据这个分数，更新模型：高分鼓励，低分惩罚
def reward_func(completions, **kwargs):
    prompts = kwargs["prompts"] # 从输入中拿出用户的提示词
    texts = [p + c for p, c in zip(prompts, completions)] # 把问题 + 模型生成的回答拼在一起
    # 把拼接好的文本 → 变成模型能看懂的数字（token）
    inputs = tokenizer(   
        texts, 
        padding=True,  # 不足长度自动补
        truncation=True,  # 太长自动截断
        return_tensors="pt"
    ).to(rm_model.device)
    
    with torch.no_grad():
        # rm_model(**inputs)：把文本输入奖励模型， .logits：输出原始分数，.squeeze(-1)：把形状  从 [batch,1] → [batch] 展平
        # **就是拆包的意思，将字典拆包成模型能识别的参数
        scores = rm_model(**inputs).logits.squeeze(-1)  
    
    return scores.tolist()
```

# 3 加载个处理训练数据

```Python
# 5. 准备数据 (增加调试打印，确保数据加载正确)
print(">>> 正在准备数据集...")
with open(PROMPT_DATASET, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 构建 prompts 列表
prompts_list = []
for item in raw_data:
    # 容错处理 key 名
    prompt_key = next((k for k in item.keys() if 'prompt' in k), 'prompt') # 找出包含 prompt 字符串的 key
    prompt_text = item[prompt_key] # 真正的问题文本取出来
    formatted_prompt = f"User: {prompt_text}\nAssistant: "
    prompts_list.append(formatted_prompt)

# 构建 Dataset (这一步确保 __len__ 是存在的)
dataset = Dataset.from_dict({"prompt": prompts_list})  # 变成训练专用数据集

# 【关键调试】打印数据集大小，确认加载成功
print(f">>> 数据集加载完成，共 {len(dataset)} 条数据")
```

# 4 配置 LoRA 和 GRPO 并进行训练

配置 LoRA:

```Python
# 6. 配置 LoRA
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    bias="none", 
    task_type="CAUSAL_LM"
)
```

配置 GRPO：

```Python
# 7. 配置 GRPO
batch_size = 2  # 一次给模型喂两条数据
accumulation_steps = 4  # 累积 4 步之后，再更新一次模型
num_epochs = 1 # 把所有数据 从头到尾跑 1 遍
# 公式：steps = (数据量 / batch_size) * epochs / accumulation_steps
estimated_max_steps = int((len(dataset) / batch_size) * num_epochs / accumulation_steps)
# 自动计算模型要更新多少次参数，防止计算出 0，至少给 10 步
estimated_max_steps = max(estimated_max_steps, 10)

training_args = GRPOConfig(
    output_dir="./final_rlhf_llm",   # 模型保存路径
    learning_rate=1.41e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=accumulation_steps,
    max_completion_length=8,   # 模型最多生成 8 个 token
    max_steps=estimated_max_steps,  # 训练 10 步就停止
    
    # 减小组内生成数量
    num_generations=2,  # 1 个 prompt 生成 2 个回答，用来组内对比
    logging_steps=1, # 每一步都打印日志，方便看效果
    bf16=True,
    remove_unused_columns=False, # 不删除数据列，防止训练出错
)
```

初始化 Trainer，开始训练：

```Python
# 8. 初始化 Trainer
print(">>> 正在初始化 Trainer...")
trainer = GRPOTrainer(
    model=SFT_MODEL_PATH,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

# 9. 开练！
print(">>> 开始训练！")
trainer.train()

print(">>> 训练完成，保存模型...")
trainer.save_model("./final_rlhf_llm")
```
[代码链接](https://github.com/kxmust/LLM_stu/blob/main/05%20RLHF.ipynb)