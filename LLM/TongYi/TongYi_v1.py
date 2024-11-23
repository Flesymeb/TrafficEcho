import json
import os
import re
import time
from openai import OpenAI


class TextAnalyzer:

    def __init__(self, api_key, base_url, input_filename):
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"初始化OpenAI客户端失败：{e}")
            exit(1)

        self.input_filename = input_filename
        # 自动生成输出文件名
        self.output_filename = f"{os.path.splitext(input_filename)[0]}_output.json"

    def analyze_text_for_keywords_and_evaluation(self, json_input, index):
        try:
            prompt = f"""
            用户提供的JSON数据如下：{json_input}
            请从文本中提取与交通出行相关的关键词，并基于上下文信息发掘评价指标（如舒适度、效率等）。
            1. 返回每个关键词的分类和权重（0-1）。
            2. 判断该文本的情感倾向，输出情感分类（正面/负面）。
            3. 根据情感和关键词权重计算整体出行评价分数，范围为0-100。
            4. 仅返回如下JSON格式内容：
            {{
                "keywords": [
                    {{
                        "keyword": "<关键词1>",
                        "category": "<分类1>",
                        "weight": <权重1>
                    }},
                    ...
                ],
                "sentiment": "<情感倾向>",
                "summary": "<总结描述>"
            }}
            """

            # 调用 OpenAI 接口
            completion = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "请严格按照 JSON 格式输出结果，仅返回 JSON 格式，不包含其他文字。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            # 获取 API 返回的原始内容
            raw_content = completion.choices[0].message.content
            print(f"Raw API Response for entry {index}: {raw_content}")

            # 使用正则表达式提取 JSON 内容
            json_match = re.search(r"{.*}", raw_content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                return self.validate_and_process_output(json_content, index)
            else:
                raise ValueError(f"Invalid JSON structure for entry {index}")

        except Exception as e:
            print(
                f"Error in analyze_text_for_keywords_and_evaluation for entry {index}: {e}"
            )
            return {"error": str(e), "index": index}

    def validate_and_process_output(self, json_content, index):
        try:
            # 验证 JSON 格式是否正确
            data = json.loads(json_content)
            return self.process_output(data, index)
        except json.JSONDecodeError as e:
            self.log_invalid_response(json_content, index)
            return {"error": f"Invalid JSON format: {e}", "index": index}

    def process_output(self, data, index):
        keywords = data.get("keywords", [])
        sentiment = data.get("sentiment", "")
        summary_text = data.get("summary", "")

        # 打印关键词详情（调试用）
        print(f"Keywords for entry {index}: {keywords}")
        print(f"Sentiment for entry {index}: {sentiment}")

        # 计算出行评价分数
        score = self.calculate_score(keywords, sentiment)

        # 构造结果（不在 summary 中添加分数）
        return {
            "keywords": keywords,
            "score": score,
            "sentiment": sentiment,
            "summary": f"{summary_text}",
        }

    def calculate_score(self, keywords, sentiment):
        base_score = 0  # 初始分数
        total_weight = 0  # 总权重

        for keyword_entry in keywords:
            weight = keyword_entry.get("weight", 0)
            category = keyword_entry.get("category", "未分类")

            # 根据分类给权重加成
            if category == "舒适度":
                base_score += weight * 1.5  # 舒适度加成
            elif category == "效率":
                base_score += weight * 1.2  # 效率加成
            else:
                base_score += weight * 0.8  # 未分类权重较低

            total_weight += weight

        # 如果没有有效权重，默认返回 0
        if total_weight == 0:
            return 0

        # 基于情感调整评分
        if sentiment == "负面":
            # 负面评价扣分
            base_score *= 0.7  # 负面评价减弱分数
        elif sentiment == "正面":
            # 正面评价加分
            base_score *= 1.1  # 正面评价增强分数

        # 计算归一化分数（映射到 0-100）
        score = min(100, int((base_score / total_weight) * 100))
        return score

    def log_invalid_response(self, content, index):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "invalid_responses.log")
        with open(log_file, "a", encoding="utf-8") as logfile:
            logfile.write(f"Data Index: {index}\n")
            logfile.write(f"Response:\n{content}\n")
            logfile.write("-" * 80 + "\n")

    def read_json_file(self):
        file_path = os.path.join("data", self.input_filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return []

    def append_output_to_json_file(self, entry):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.output_filename)

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

    def process_data(self):
        data = self.read_json_file()
        if not data:
            print("No data to process. Exiting.")
            return

        # 逐条处理数据
        for index, entry in enumerate(data, start=1):
            print(f"正在处理第 {index} 条数据...")
            analysis_result = self.analyze_text_for_keywords_and_evaluation(
                json.dumps(entry, ensure_ascii=False), index
            )
            self.append_output_to_json_file(analysis_result)
            print(f"第 {index} 条数据已处理并保存到文件。")
            time.sleep(1)  # 限速，避免触发 API 限制

        print(
            f"所有数据处理完成！结果已保存到: {os.path.join('output', self.output_filename)}"
        )


# 主程序入口
if __name__ == "__main__":
    analyzer = TextAnalyzer(
        api_key="sk-64b850c731f545569c8bf61e3c416324",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input_filename="weibo_20241025.json",
    )
    analyzer.process_data()
