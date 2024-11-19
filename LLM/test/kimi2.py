import json
import os
import requests
from requests.exceptions import RequestException
from openai import OpenAI

# 创建OpenAI客户端实例，使用 Moonshot API 的 Base URL
client = OpenAI(
    api_key="sk-6SPxtFoADCQXHRVo9kkxFBBwLCbpHiKyC5gqUgLekriuvPiU",
    base_url="https://api.moonshot.cn/v1",  # Moonshot API的base_url
)


# 定义函数处理聊天任务，包括自动关键词提取和评价指标发掘
def analyze_text_for_keywords_and_evaluation(json_input):
    try:
        # 提取用户输入的文本
        prompt = f"""
        用户提供的JSON数据如下：
        {json_input}
        请从文本中自动提取关键词，并基于上下文信息，自动发掘评价指标（如舒适度、效率等），同时为每个关键词分配权重并输出评价类别。请直接输出关键词、评价类别、权重及相关描述。
        """

        # 调用ChatGPT API创建对话完成请求
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",  # 模型名称
            messages=[
                {
                    "role": "system",
                    "content": """\
- Role: 数据分析专家和自然语言处理工程师
- Background: 用户需要从JSON格式的数据中提取与交通出行相关的关键词，并自动分析评价的具体方面（如舒适度、效率等），并为每个关键词分配权重。
- Goals: 从文本中自动提取关键词，基于上下文发掘评价指标，并分配权重。
- OutputFormat: 以JSON格式输出关键词、评价类别、权重及相关描述。
- Workflow:
  1. 解析用户提供的JSON数据，提取包含文本信息的部分。
  2. 使用NLP技术提取关键词（如TF-IDF、TextRank或BERT）。
  3. 自动识别和发掘出潜在的评价维度（如舒适度、效率、服务质量等）。
  4. 为每个关键词分配权重，反映其在文本中的重要性。
  5. 以JSON格式输出关键词、分类和权重。
""",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 控制生成文本的创意性
        )

        # 输出助手的回复
        return completion.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"


# 从文件读取JSON数据
def read_json_file(filename):
    file_path = os.path.join("data", filename)  # 构造完整文件路径
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# 将结果写入输出文件
def write_output_to_file(output, output_filename):
    output_dir = "output"  # 输出文件夹
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹
    output_path = os.path.join(output_dir, output_filename)  # 构造输出文件路径

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(output, outfile, ensure_ascii=False, indent=4)


# 示例输入：从文件读取JSON格式数据
# filename = "weibo_20241025.json"  # 这里替换为你的JSON文件名

filename = "堵车.json"
json_data = read_json_file(filename)

# 调用函数并打印结果
result = analyze_text_for_keywords_and_evaluation(json.dumps(json_data))
print(result)

# 将结果输出到文件
output_filename = f"{os.path.splitext(filename)[0]}_output.json"  # 创建输出文件名
write_output_to_file(json.loads(result), output_filename)  # 将结果转换为字典后写入文件

print("分析结果已保存到:", os.path.join("output", output_filename))
