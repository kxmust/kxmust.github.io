# 1 现有大模型的基本架构

## 1.1 现有 LLM 采用的是什么架构

Google 团队在 2017 年论文《Attention Is All You Need》中提出了一种 Transformer 架构，其结构如图下图所示。它包含了一个编码器和一个解码器：

- 编码器：编码器的任务是将输入的数据处理成矩阵，这些矩阵中包含了输入数据的所有信息。
- 解码器：解码器的输入除了包含了编码器提取的所有信息外，还包含了上一时刻的输出信息。在刚开始输出时，没有上一时刻的输出，因此输入的是一个开始符号，当最后输出一个截止符号时，停止输出。

Transformer 架构刚开始被设计出来是用来做机器翻译的，随着研究人员的深入，发现它不仅在机器翻译任务中拥有突出变现，在其他任务中也表现优异。

随后，研究人员发现，解码器能够很好的完成词语接龙的任务，于是他们移除了编码器，只保留了解码器的结构，这种模型被称为 Decoder-Only Transformer 架构，这也是现有大多数 LLM的基础架构。

## 1.2 现有 LLM 的输出过程

现有 LLM 的基础输出过程如下：

1. 将用户输入的提示词作为是输入，预测下一个词的概率。
2. 基于概率选择下一个词，然后将用户的输入和预测的下一个词一起输入模型，继续预测下一个词。
3. 反复进行上面的过程知道输出截止符号。

# 2 实现 LLM 本地部署所需要的工具

## 2.1 Pytorch

现有的大模型大部分使用 Pytorch 来进行搭建和训练的，Pytorch 中提供了很多搭建大语言模型所需的工具，比如 Embedding 层、Multi-Head Attention 层、前馈神经网络、归一化层等等这些基础组件。可以让厂商能够轻松搭建 LLM 模型，并且可以魔改加入自己的架构。比如 Deepseek 会用 MoE 来取代 Feed Forward等。

个人使用 Pytorch 来搭建 LLM 是非常复杂的，因为 Pytorch 中并没封装完整的 Transformer 架构，也没法直接搭建厂商公开的开源大模型。因此我们需要借助其他的工具来进行本地化部署。

## 2.2 HuggingFace

HuggingFace 这个网站是一个开源模型的托管网站，相当于模型界的 Github。大多数厂商的开源模型都是托管在这个网站，包括国内的 Deepseek，通义等。

HuggingFace 构建和维护了一个 Transformers 库，这个库中使用 pytorch 实现了很多开源的大模型架构，个人可以很方便的进行模型的下载和本地化部署。当然这个库除了接入 pytorch 之外，还接入了许多其他的库，比如将模型加载到不同显卡的 Accelerate 库，还有做模型量化的 bitsandbytes 库等等。有了好的工具下面开始本地化部署一个开源的小模型。

# 3 本地化部署一个Deepseek R1 小模型

## 3.1 从 HuggingFace 网站中下载LLM 参数到本地

首先我们要从 HuggingFace 网站中找到一个我们需要的轻量化模型，我们要部署的是DeepSeek-R1-Distill-Qwen-1.5B 这个只有 15 亿参数的小模型。

下载网址为：[https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

将下载的模型保存在 model 文件夹中。

## 3.2 利用 Transformers 库来加载我们下载的模型

### 3.2.1 加载方式一

我们打开下载的模型，其中里面有一个 config.json 文件，打开后可以看到这个模型采用的是“Qwen2ForCausalLM”这个架构，我们可以在 Transformers 这个库中直接导入这个架构，然后再加载我们下载的模型。

```Python
from transformers import Qwen2ForCausalLM
model = Qwen2ForCausalLM.from_pretrained("./model") #下载好的模型参数保存在model文件夹中

```

我们可以看到模型的结构如下：

```Python
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```

其实我们加载的这个模型就是 Pytorch 中的 Module 类型，我们可以利用下面的代码检测一下是不是。

```Python
from torch.nn import Module
isinstance(model, Module)
```

### 3.2.2 加载方式二

除了上述加载模型的方式外还可以使用 AutoModelForCausalLM 来统一加载所有对话形式的大语言模型

```Python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./model")
```

### 3.2.3 加载方式三

还有一个种方法可以直接使用模型名称进行加载，然后会自动从网络中下载该模型的参数（不是很推荐，因为下载目录无法更改）。

```Python
model=AutoModelForCausalLM.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")
```

## 3.3 使用硬件加速

模型加载到内存中后，只有 CPU 才能访问内存，如果要让 GPU，或者 NPU 加速，需要将模型加载到显存中。可以使用如下命令：

```Python
model.to('mps') #mac用 mps，window 用 cuda
model.device
```

## 3.4 加载分词器并处理输入文字

用户输入的文字需要经过分词器处理成 token 之后才能发送到模型进行处理，一般下载的本地模型都会包含分词器 Tokenizer。分词器常见的算法可以分为三种，BPE、WordPiece 和 Unigram。我们需要加载这个分词器:

```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")
```

然后利用加载的分词器处理我们的输入文本，包括处理成 Token，进行填充，还有转化为 Tensor:

```Python
# 将一个输入信息转化为 token，并传入显存
inputs = tokenizer(
    [
        "hello，tell me your name please.",
        "10*5="
    ],
    padding=True,    # 模型可以好几句话同时接龙，但是模型内是矩阵乘法，需要保证输入句子长度相同，因此需要填充
    padding_side="left", # 由于模型是在输入句子的结尾进行接龙，因此填充需要填在最左边
    return_tensors="pt"  # tokenizer返回的是python 数组，pytorch 推理需要传入 tensor，所以这个做了一个转换
).to(model.device)
```

我们可以利用下面的命令将转化后的 token 进行翻译，看具体内容是什么：

```Python
tokenizer.batch_decode(inputs.input_ids) # 将句子翻译出来看一下具体的内容
```

## 3.5 模型推理

将处理好的输入文本放入到模型中，让模型给出输出。

```Python
outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask=inputs.attention_mask, # 反应哪些是填充的内容，哪些是用户输入的内容
    max_new_tokens = 32  #最多输出多少个 token

)

# 将结果解码看一下是什么内容：
tokenizer.batch_decode(outputs)

```

输出结果如下：

```Python
['hello，tell me your name please. How do you know what your name is? What does it mean? How do you translate your name into Chinese? How do you handle the translation process? How',
 '<｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜>10*5=50.\n\nSo, the total sum would be 50 + 50 + 50 = 150.\n\nWait, but that seems']
```

可以看到，输出的结果基本是乱的，没有得到正确的结果，为什么呢？

## 3.6 分析模型胡说八道的原因并进行修改

模型一般是使用大量文章来对模型进行预训练，也就是文字接龙，但是实际使用中，还需要对模型进行微调才能实现对话。在模型微调中，一般都会有相对固定的格式，比如：

user：你好

Assistant：你也好

或者有的模型还会在模型中加入思考和推理的过程。因此我们需要对输入的文本进行一个格式的转化，让模型能更好的识别。

我们对输入的内容格式进行修改：

```Python
# 修改输入内容的格式
message = [
    [
        {"role":"user", "content": "hello，tell me your name please."}
    ],
    [
        {"role":"user", "content": "10*5="}
    ]
]

tokenizer.padding_side = "left"

# tokenizer中封装了一个chat_template脚本，负责将 message 这种格式转化成模型训练时用到的对话格式
inputs = tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    padding=True,
    add_generation_prompt=True    # 生成的token会跟上Assistant的标签和思维链的起始标志
).to(model.device)

#tokenizer.batch_decode(inputs.input_ids)
```

然后将这些内容输入进模型进行推理：

```Python
outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask=inputs.attention_mask, # 反应哪些是填充的内容，哪些是用户输入的内容
    max_new_tokens = 256,  #最多输出多少个 token
    pad_token_id=tokenizer.eos_token_id,  # 消除警告，告诉它该结束了
    do_sample=True,  # 开启采样，让对话更自然
)
# 将结果解码看一下是什么内容：
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

输出结果为：

```Python
["<｜User｜>hello，tell me your name please.<｜Assistant｜><think>\nI'm DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. I'll do my best to help you.\n</think>\n\nI'm DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. I'll do my best to help you.",
 '<｜User｜>10*5=<｜Assistant｜><think>\nTo solve 10 multiplied by 5, I start by identifying the numbers involved, which are 10 and 5.\n\nNext, I perform the multiplication operation by multiplying these two numbers together. \n\nFinally, I arrive at the product, which is 50.\n</think>\n\nTo solve \\(10 \\times 5\\), follow these easy steps:\n\n1. **Identify the numbers to multiply:**\n   \\[\n   10 \\quad \\text{and} \\quad 5\n   \\]\n\n2. **Multiply the numbers:**\n   \\[\n   10 \\times 5 = 50\n   \\]\n\nSo, the final answer is:\n\\[\n\\boxed{50}\n\\]']
```

# 4 对模型输出进行一些优化

从上面可以看到，这个本地化部署存在两个问题：

1. 模型的输出包含了很多其他的符合，不符合阅读习惯。
2. 模型不能连续对话，输入的信息还需要手动处理。

因此，我们需要写一个循环并且屏蔽一些输出，来实现连续对话。

```Python
messages = []

print("="*50)
print("欢迎与 DeepSeek-R1 交流！输入 'quit' 或 'exit' 结束对话。")
print("="*50)

# 开启连续对话循环
while True:
    # 获取用户在终端的输入
    user_input = input("\nUser: ")

    # 设置退出指令
    if user_input.lower() in ['quit', 'exit']:  # 将字符串中所有大写字母转换为小写字母
        print("结束对话，再见！")
        break
    if not user_input.strip(): # 去除两端空白字符
        continue

    # A. 将用户的最新发言追加到历史记录中
    messages.append({"role": "user", "content": user_input})

    # B. 应用 Chat Template 处理包含历史记录的完整对话
    # 每次传入的都是不断增长的 messages 列表
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # 自动加上 <|Assistant|> 的起始标志
        return_tensors="pt"
    ).to(device)

    # 记录当前输入 Token 的长度
    input_length = inputs.input_ids.shape[1]

    # C. 模型推理生成回复
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,  # 控制每次最大输出长度
        pad_token_id=tokenizer.eos_token_id,  # 消除警告，告诉它遇到结束符该结束了
        do_sample=True,  # 开启采样，让对话更自然
        temperature=0.7  # 控制回答的创造性和发散度
    )

    # D. 核心切片：将包含输入和输出的完整 Tensor 进行切片，只保留本次新生成的部分
    new_tokens = outputs[0][input_length:]

    # 解码生成的 Token 变为可读文本
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # --- 【新增的处理逻辑】 ---
    # 如果输出中包含 </think>，我们就以它为界限进行分割，只取最后面的那部分（即真正的回答）
    if "</think>" in response:
        clean_response = response.split("</think>")[-1].strip()
    else:
        clean_response = response.strip()
    # -----------------------

    print(f"\nAssistant: {clean_response}")

    # E. 将模型的回复加入历史记录，完成这一轮的上下文闭环
    messages.append({"role": "assistant", "content": response})
```

# 5 完整代码

```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 初始化模型和分词器路径 (请确保相对路径或绝对路径正确)
model_path = "./model"

print("正在加载模型和分词器，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


# 2. 硬件加速配置 (自动检测 Mac 的 MPS)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = model.to(device)
print(f"模型已成功加载至 {device} 加速计算！\n")

# 3. 初始化全局对话历史列表
messages = []

print("="*50)
print("欢迎与 DeepSeek-R1 交流！输入 'quit' 或 'exit' 结束对话。")
print("="*50)


# 4. 开启连续对话循环
while True:
    # 获取用户在终端的输入
    user_input = input("\nUser: ")

    # 设置退出指令
    if user_input.lower() in ['quit', 'exit']:  # 将字符串中所有大写字母转换为小写字母
        print("结束对话，再见！")
        break
    if not user_input.strip(): # 去除两端空白字符
        continue

    # A. 将用户的最新发言追加到历史记录中
    messages.append({"role": "user", "content": user_input})

    # B. 应用 Chat Template 处理包含历史记录的完整对话
    # 每次传入的都是不断增长的 messages 列表
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # 自动加上 <|Assistant|> 的起始标志
        return_tensors="pt"
    ).to(device)

    # 记录当前输入 Token 的长度
    input_length = inputs.input_ids.shape[1]

    # C. 模型推理生成回复
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,  # 控制每次最大输出长度
        pad_token_id=tokenizer.eos_token_id,  # 消除警告，告诉它遇到结束符该结束了
        do_sample=True,  # 开启采样，让对话更自然
        temperature=0.7  # 控制回答的创造性和发散度
    )

    # D. 核心切片：将包含输入和输出的完整 Tensor 进行切片，只保留本次新生成的部分
    new_tokens = outputs[0][input_length:]

    # 解码生成的 Token 变为可读文本
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # --- 【新增的处理逻辑】 ---
    # 如果输出中包含 </think>，我们就以它为界限进行分割，只取最后面的那部分（即真正的回答）
    if "</think>" in response:
        clean_response = response.split("</think>")[-1].strip()
    else:
        clean_response = response.strip()
    # -----------------------

    print(f"\nAssistant: {clean_response}")

    # E. 将模型的回复加入历史记录，完成这一轮的上下文闭环
    # messages.append({"role": "assistant", "content": response})
```
[代码链接](https://github.com/kxmust/LLM_stu/blob/main/01%20deepseek_deploy.ipynb)
