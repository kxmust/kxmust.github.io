假设你手里有一篇关于《基于强化学习的微电网智能治理》的长篇文章或学术论文，你想用它来微调你的大模型。你**绝对不能**直接把整篇文章塞进 `output` 里，这会导致模型只会死记硬背，不会回答问题。

目前业界的标准做法是使用 **“大模型合成数据法”（Self-Instruct 方案）**，具体分为以下四步：

# 第一步: 文本切块 (Chunking)

长篇大论无法直接处理，你需要用 Python 脚本将文章切分成语义完整的小段落。核心代码如下：


```Python
# 定义一个切块函数
def chunk_document(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 基于正则分割文档，将长文档切割成块
    chunks = re.split(r'(?=\n\d+\.\s)', content)
    # 去除收尾空格、换行、制表符，只保留有效长度>50 字符的文本块
    cleaned_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
    
    print(f"[Step 1] 文档切分完成，共切分出 {len(cleaned_chunks)} 个文本块。")
    return cleaned_chunks

file_path = "document.md"
result_chunks = chunk_document(file_path)
result_chunks[0]
```

用document.md文档中的内容测试一下，输出第一块的内容:

```Markdown
Step 1] 文档切分完成，共切分出 4 个文本块。
'1. 全球的“双碳”目标。\n\n   探讨任何能源技术的变革，都绕不开当前的全球大背景——气候变化。为了避免极端气候的不可逆破坏，全球形成了一个硬性共识：必须减少温室气体排放。目前，欧盟、美国、日本等发达经济体普遍承诺在2050年实现碳中和。在这个宏大的目标下，能源行业的脱碳是重中之重，因为全球温室气体排放中，能源生产和消费占据了绝对的大头（约占70%以上）。这不仅是一场环保运动，更是一场重塑全球工业格局的能源革命。\n\n   **核心内容：**\n\n   **共同危机：** 应对全球气候变化，《巴黎协定》控温目标（1.5℃/2.0℃）。\n\n   **全球共识：** “碳达峰”（Carbon Peak）与“碳中和”（Carbon Neutrality）成为全球行动纲领。\n\n   **时间表：** 超过130个国家和地区提出了碳中和目标（多数定在2050年）。'
```

# 第二步: 召唤一个“更聪明”的大模型当苦力

我们人类手动去写成千上万个 `[instruction, input, output]` 会累死。所以，我们通常会调用一个能力极强的大模型 API（比如 Gemini Pro 或 DeepSeek-V3），让它阅读你切好的段落，并自动生成问答对。

为了保障大数据 API 的安全，我们会创建一个.env 文件来保存我们大语言模型的 API，然后进行调用。这里我们调用的是 Deepseek。

代码如下：

```Python
from dotenv import load_dotenv

# 1. 自动寻找当前目录下的 .env 文件，并将其内容加载到系统环境变量中
load_dotenv("deepseek_key.env")

# 2. 安全地获取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

# 3. 做一个安全校验，防止没配好 Key 就直接往下跑报错
if not api_key:
    raise ValueError("⚠️ 未找到 API Key！请检查是否创建了 .env 文件并在其中配置了 DEEPSEEK_API_KEY。")

# 4. 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
) 
```

# 第三步: 设计 Prompt 批量生成问答数据

因为 DeepSeek 主要依靠 Prompt 和 response_format 来约束输出，我们需要在 Prompt 中非常清晰地给出期待的 JSON 结构示例。

你需要写一个精确的 Prompt 发给这个强大的大模型：

> **给大模型的 Prompt 模板：**
你是一个专业的数据标注专家。请阅读下面这段【文章片段】，并根据该片段，为大语言模型的微调生成 3 个不同类型的训练数据。

必须包含以下三种类型：
1. **概念解释：** 针对片段中的专业术语进行提问（此时 input 为空）。
2. **内容总结：** 要求总结片段的核心意思（此时 input 为该片段）。
3. **逻辑推理：** 基于片段内容提问一个为什么（此时 input 为空）。

请严格以 JSON 数组的格式输出，包含 instruction, input, output 三个字段。

代码如下：

```Python
# 定义生成问答的函数（调用 Deepseek API）
def generate_qa_from_chunk(chunk_text: str) -> list[dict]:
    """
    调用 DeepSeek API 并强制输出 JSON 格式。
    """
    # 因为 DeepSeek 主要依靠 Prompt 和 response_format 来约束输出，
    # 我们需要在 Prompt 中非常清晰地给出期待的 JSON 结构示例。
    prompt = f"""
    请阅读下面这段【文章片段】，并根据该片段，为大语言模型的微调生成 3 个不同类型的训练数据。
    必须包含以下三种类型：
    1. 概念解释：针对片段中的专业术语进行提问（此时 input 为空）。
    2. 内容总结：要求总结片段的核心意思（此时 input 为该片段）。
    3. 逻辑推理：基于片段内容提问一个为什么（此时 input 为空）。
    
    【重要指令】
    请务必输出一个合法的 JSON 对象，包含一个名为 "items" 的数组，数组内部是包含 "instruction", "input", "output" 三个字段的字典。示例格式如下：
    {{
        "items": [
            {{"instruction": "什么是碳中和？", "input": "", "output": "..."}},
            {{"instruction": "总结这段话的意思。", "input": "...", "output": "..."}}
        ]
    }}
    
    【文章片段】：
    {chunk_text}
    """
    
    try:
        # 调用 DeepSeek 的对话 API
        response = client.chat.completions.create(
            model="deepseek-chat",  # 推荐使用通用对话模型，性价比极高
            messages=[
                {"role": "system", "content": "你是一个专业的数据标注专家，请严格按照用户要求的 JSON 格式输出数据。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # 开启 JSON 模式，确保返回合法 JSON
            temperature=0.7 
        )
        
        # 解析 DeepSeek 返回的文本内容
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # 提取 items 数组
        return result_json.get("items", [])
        
    except Exception as e:
        print(f"调用 DeepSeek 生成数据时出错: {e}")
        return []
```

# 第四步: 数据收集与格式化

写一个主函数调用前面定义的方法：

```Python
# 定义主函数调用前面定义的方法处理数据
def main():
    input_file = 'document.md'
    output_file = 'new_dataset.json'
    
    final_dataset = []
    
    # 执行第一步，先切块
    chunks = chunk_document(input_file)
    
    # 循环处理每一个块的内容
    print("[Step 2 & 3] 开始调用大模型生成问答对...")
    for i, chunk in enumerate(chunks):
        print(f"正在处理第 {i+1}/{len(chunks)} 块文本...")
        qa_pairs = generate_qa_from_chunk(chunk)
        final_dataset.extend(qa_pairs)
    
    # 收集数据并处理
    print(f"[Step 4] 收集到 {len(final_dataset)} 条数据，准备保存。")
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证输出的 JSON 中文正常显示而不是 Unicode 编码
        # indent=4 让输出的 JSON 具备良好的可读性（缩进空格数）
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 数据集已成功保存至 {output_file} ！你可以直接用它进行微调了。")

if __name__ == "__main__":
    main()
```

强大的大模型接收到上述指令后，就会为你吐出完美的 `dataset.json` 格式数据：

```JSON
[
    {
        "instruction": "什么是碳中和？",
        "input": "",
        "output": "碳中和是指通过植树造林、节能减排等形式，抵消自身产生的二氧化碳排放量，实现二氧化碳的净零排放。在文章片段中，它被描述为全球应对气候变化的关键行动纲领，许多国家和地区（如欧盟、美国、日本）承诺在2050年实现这一目标，以减少温室气体排放，避免极端气候的不可逆破坏。"
    },
    {
        "instruction": "总结这段话的意思。",
        "input": "1. 全球的“双碳”目标。\n\n探讨任何能源技术的变革，都绕不开当前的全球大背景——气候变化。为了避免极端气候的不可逆破坏，全球形成了一个硬性共识：必须减少温室气体排放。目前，欧盟、美国、日本等发达经济体普遍承诺在2050年实现碳中和。在这个宏大的目标下，能源行业的脱碳是重中之重，因为全球温室气体排放中，能源生产和消费占据了绝对的大头（约占70%以上）。这不仅是一场环保运动，更是一场重塑全球工业格局的能源革命。\n\n**核心内容：**\n\n**共同危机：** 应对全球气候变化，《巴黎协定》控温目标（1.5℃/2.0℃）。\n\n**全球共识：** “碳达峰”（Carbon Peak）与“碳中和”（Carbon Neutrality）成为全球行动纲领。\n\n**时间表：** 超过130个国家和地区提出了碳中和目标（多数定在2050年）。",
        "output": "这段话主要阐述了全球应对气候变化的背景和行动。核心意思是：气候变化是当前全球面临的共同危机，为避免不可逆破坏，全球形成了减少温室气体排放的硬性共识，具体体现为“碳达峰”和“碳中和”的行动纲领。其中，能源行业脱碳是关键，因为其排放占全球温室气体排放的70%以上。超过130个国家和地区（包括欧盟、美国、日本等）已提出碳中和目标，多数定在2050年，这不仅是环保运动，更是一场重塑工业格局的能源革命。"
    },
    {
        "instruction": "为什么能源行业的脱碳是实现碳中和的重中之重？",
        "input": "",
        "output": "因为全球温室气体排放中，能源生产和消费占据了绝对的大头，约占70%以上。这意味着能源行业是主要的排放源，如果不优先脱碳（即减少或消除碳排放），全球碳中和目标将难以实现。因此，在应对气候变化和实现碳中和的背景下，能源行业的脱碳成为关键和首要任务。"
    }
]
```

将文章的所有段落都通过这个流程跑一遍，你就能从一篇文章中提取出成百上千条高质量的微调数据了！这就完成了从“非结构化文档”到“高质量指令数据集”的蜕变。
[代码链接](https://github.com/kxmust/LLM_stu/blob/main/04%20build_dataset.ipynb)