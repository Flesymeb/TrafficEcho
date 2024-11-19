import json
import os
import time
import re
from openai import OpenAI

# 创建OpenAI客户端实例
client = OpenAI(
    api_key="sk-6SPxtFoADCQXHRVo9kkxFBBwLCbpHiKyC5gqUgLekriuvPiU",
    base_url="https://api.moonshot.cn/v1",  # Moonshot API的base_url
)


# 分析文本，提取关键词和评价指标
def analyze_text_for_keywords_and_evaluation(json_input, index):
    try:
        prompt = f"""
        用户提供的JSON数据如下：
        {json_input}
        请从文本中自动提取关键词，并基于上下文信息，自动发掘评价指标（如舒适度、效率等），同时为每个关键词分配权重并输出评价类别。请直接输出关键词、评价类别、权重及总结描述，尽量简洁。
        """

        # 调用OpenAI接口
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": """\
        - Role: 数据分析专家和自然语言处理工程师
        - Background: 用户需要从JSON格式的数据中提取与交通出行相关的关键词，并自动分析评价的具体方面（如舒适度、效率等），并为每个关键词分配权重。
        - Goals: 从文本中自动提取关键词，基于上下文发掘评价指标，并分配权重。
        - OutputFormat: 仅以JSON格式输出关键词、评价类别、权重和总结描述，不包含其他无关文本。只返回JSON内容。
        - Workflow:
        1. 解析用户提供的JSON数据，提取包含文本信息的部分。
        2. 使用NLP技术提取关键词（如TF-IDF、TextRank或BERT）。
        3. 自动识别和发掘出潜在的评价维度（如舒适度、效率、服务质量等）。
        4. 为每个关键词分配权重，反映其在文本中的重要性。
        5. 以JSON格式输出关键词、分类、权重和总结描述。确保返回的仅是JSON格式，不包含其他解释性文字。
        """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # 打印调试信息
        raw_content = completion.choices[0].message.content
        print(f"Raw API Response for entry {index}: ", raw_content)

        # 清理多余的标记（移除 ```json 和 ```，同时删除可能的额外换行或空格）
        clean_content = re.sub(r"```json|```", "", raw_content).strip()

        # 检查并返回符合格式的JSON
        if clean_content.startswith("{") and clean_content.endswith("}"):
            result = json.loads(clean_content)
            return format_keywords_output(result)
        else:
            raise ValueError(f"Invalid JSON structure for entry {index}")

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error at entry {index}: {e}")
        log_invalid_response(raw_content, index)
        return {"error": "Invalid JSON format in API response", "index": index}
    except Exception as e:
        print(
            f"Error in analyze_text_for_keywords_and_evaluation for entry {index}: {e}"
        )
        return {"error": str(e), "index": index}


# 将返回结果格式化为期望的结构
def format_keywords_output(data):
    formatted_result = {
        "keywords": [],  # 将存储所有关键词的列表
        "summary": data.get("summary", ""),  # 获取总结描述
    }

    # 确保数据中包含关键词
    if "keywords" in data:
        for keyword_entry in data["keywords"]:
            formatted_result["keywords"].append(
                {
                    "keyword": keyword_entry.get("keyword", ""),
                    "category": keyword_entry.get("category", "未分类"),
                    "weight": keyword_entry.get("weight", 0),
                    # "description": keyword_entry.get("description", ""),
                }
            )
    return formatted_result


# def format_keywords_output(data):
#     formatted_result = {"keywords": []}

#     if "keywords" in data:
#         for keyword_entry in data["keywords"]:
#             formatted_result["keywords"].append(
#                 {
#                     "keyword": keyword_entry.get("keyword", ""),
#                     "category": keyword_entry.get("category", "未分类"),
#                     "weight": keyword_entry.get("weight", 0),
#                     # "description": keyword_entry.get("description", ""),
#                 }
#             )
#     return formatted_result


# 记录无效响应到日志
def log_invalid_response(content, index):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "invalid_responses.log")
    with open(log_file, "a", encoding="utf-8") as logfile:
        logfile.write(f"Data Index: {index}\n")
        logfile.write(f"Response:\n{content}\n")
        logfile.write("-" * 80 + "\n")


# 从文件读取JSON数据
def read_json_file(filename):
    file_path = os.path.join("data", filename)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return []


# 将单条结果追加到文件
def append_output_to_json_file(entry, output_filename):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # 读取已存在的数据（如果文件不存在则初始化为一个列表）
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as outfile:
                try:
                    data = json.load(outfile)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # 追加新的数据
        data.append(entry)

        # 写回文件
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")


# 主程序
def main():
    input_filename = "weibo_20241025.json"  # 输入文件名
    output_filename = f"{os.path.splitext(input_filename)[0]}_output.json"  # 输出文件名

    # 读取数据
    data = read_json_file(input_filename)
    if not data:
        print("No data to process. Exiting.")
        return

    # 逐条处理数据
    for index, entry in enumerate(data, start=1):
        print(f"正在处理第 {index} 条数据...")
        analysis_result = analyze_text_for_keywords_and_evaluation(
            json.dumps(entry, ensure_ascii=False), index
        )
        append_output_to_json_file(analysis_result, output_filename)
        print(f"第 {index} 条数据已处理并保存到文件。")
        time.sleep(2)  # 限速，避免触发API限制

    print(f"所有数据处理完成！结果已保存到: {os.path.join('output', output_filename)}")


if __name__ == "__main__":
    main()
