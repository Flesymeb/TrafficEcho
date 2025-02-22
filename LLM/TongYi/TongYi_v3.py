#!/usr/bin/env python
# -*- encoding=utf8 -*-

'''
Filename: TongYi_v3.py
Description: 
Author: Hyoung Yan
Created time: 2024-12-03 10:31:48
Last Modified time: 2025-02-22 11:27:24
'''

import json
import os
import re
import time
from openai import OpenAI


class TextAnalyzer:
    """
    OpenAI 文本分析类，用于处理 JSON 格式的文本数据，提取关键词和评价指标，计算出行评价分数。
    """

    def __init__(self, api_key, base_url):
        """
        OPENAI 文本分析类初始化函数。
        """
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"初始化OpenAI客户端失败：{e}")
            exit(1)

    def analyze_text_for_keywords_and_evaluation(self, json_input, index):
        """
        分析文本数据，提取关键词和评价指标，计算出行评价分数。

        Args:
            json_input (str): 用户提供的 JSON 格式文本数据。
            index (int): 数据索引，用于标识当前处理的数据。

        Returns:
            dict: 返回包含关键词、情感倾向、总结描述和评分的结果。
        """
        try:
            prompt = f"""
            用户提供的JSON数据如下：{json_input}
            请从文本中提取出行相关的多个关键词，并将它们按照不同的评价维度进行分类。
            例如：
            1. 舒适度：如“车内宽敞”，“环境好”
            2. 经济度：如“票价便宜”，“交通费用低”
            3. 拥挤度：如“车厢拥挤”，“路上人多”
            请返回对交通的满意度分类（正面/负面/中性）。
      
            5. 仅返回如下JSON格式
            """

            # 调用 OpenAI 接口
            completion = self.client.chat.completions.create(
                model="qwen-plus",
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
        """
        验证 JSON 格式是否正确，并处理输出结果。
        """
        try:
            # 验证 JSON 格式是否正确
            data = json.loads(json_content)
            return self.process_output(data, index)
        except json.JSONDecodeError as e:
            self.log_invalid_response(json_content, index)
            return {"error": f"Invalid JSON format: {e}", "index": index}

    def process_output(self, data, index):
        """
        处理输出结果，计算出行评价分数。

        Args:
            data (dict): 返回的包含关键词和评分的数据。
            index (int): 数据索引，用于标识当前处理的数据。

        Returns:
            dict: 返回最终的评估结果，包括每个维度的关键词、情感分类和评分。
        """
        dimensions = ["舒适度", "经济度", "拥挤度"]
        overall_score = 0
        total_weight = 0

        result = {}

        # 计算每个维度的评分
        for dimension in dimensions:
            keywords = self._ensure_list(data.get(dimension, {}).get("关键词", []))
            sentiment = self._ensure_string(data.get(dimension, {}).get("情感分类", ""))
            sentiment_scores = self._ensure_list(
                data.get(dimension, {}).get("情绪强度", [0])
            )
            weights = self._ensure_list(data.get(dimension, {}).get("权重", [0]))

            # 如果没有关键词，则跳过该维度
            if not keywords:
                continue

            # 计算单个维度的评分
            score = self.calculate_score(sentiment, sentiment_scores, weights)
            total_weight += sum(weights)
            overall_score += score * sum(weights)

            # 更新维度数据
            result[dimension] = {
                "关键词": keywords,
                "情感分类": sentiment,
                "情绪强度": sentiment_scores,
                "权重": weights,
            }

        # 计算综合评分（0-100）
        if total_weight == 0:
            overall_score = 0
        else:
            overall_score /= total_weight

        result["综合评分"] = round(overall_score, 2)

        return result

    def _ensure_list(self, value):
        """确保输入是一个列表"""
        if isinstance(value, list):
            return value
        return [value]

    def _ensure_string(self, value):
        """确保输入是一个字符串"""
        if isinstance(value, str):
            return value.strip()
        return ""

    def calculate_score(self, sentiment, sentiment_scores, weights):
        """
        计算出行评价分数，考虑情感分析、关键词权重等因素。

        Args:
            sentiment (str): 情感分类。
            sentiment_scores (list of int): 情绪强度列表。
            weights (list of float): 权重列表。

        Returns:
            float: 计算得到的评分。
        """
        if not sentiment_scores or not weights:
            return 0

        delta_k_values = [score / 100 for score in sentiment_scores]  # 标准化情绪强度
        valid_sentiments = ["负面", "中性", "正面"]

        scores = []
        for i, delta_k in enumerate(delta_k_values):
            if delta_k < 0 or delta_k > 1:
                raise ValueError("情绪强度应该在 0 到 100 之间！")

            # 确保情感类别有效
            if sentiment not in valid_sentiments:
                print(
                    f"Warning: Invalid sentiment '{sentiment}' detected. Defaulting to '中性'."
                )
                sentiment = "中性"  # 处理无效情感值，或者可以根据需要进行其他处理

            # 根据情感类别 s_k 计算评分 g_k
            if sentiment == "负面":
                g_k = 55 * delta_k
            elif sentiment == "中性":
                g_k = 10 * delta_k + 55
            elif sentiment == "正面":
                g_k = 35 * delta_k + 65
            else:
                raise ValueError("情感类别应为 '负面', '中性' 或 '正面'！")

            scores.append(g_k * weights[i])

        return sum(scores)

    def log_invalid_response(self, content, index):
        """
        记录无效的响应内容到日志文件。
        """
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "invalid_responses.log")
        with open(log_file, "a", encoding="utf-8") as logfile:
            logfile.write(f"Data Index: {index}\n")
            logfile.write(f"Response:\n{content}\n")
            logfile.write("-" * 80 + "\n")

    def read_json_file(self, file_path):
        """
        读取 JSON 文件内容。

        Args:
            file_path (_type_): 文件路径。

        Returns:
            _type_: 返回 JSON 文件内容。
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return []

    def append_output_to_json_file(self, entry, output_filename):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 检查和修正情感分类字段为空的情况
            for dimension, data in entry.items():
                if isinstance(data, dict) and "情感分类" in data:
                    sentiment = data.get("情感分类", "").strip()
                    if not sentiment:
                        print(
                            f"Warning: Invalid sentiment detected for {dimension}. Defaulting to '中性'."
                        )
                        data["情感分类"] = "中性"  # 默认设置为 "中性"

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

    def process_files(self, folder_path, selected_files):
        """
        处理指定文件夹中的 JSON 文件。根据序号选择文件处理，或者处理所有文件。

        Args:
            folder_path (_type_): 文件夹路径。
            selected_files (_type_): 选中的文件列表。
        """
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not all_files:
            print("No JSON files found in the directory.")
            return

        print("请选择一个文件处理：")
        for idx, filename in enumerate(all_files, 1):
            print(f"{idx}. {filename}")
        print(f"{len(all_files) + 1}. 选择全部文件处理")

        choice = int(input("请输入选择的文件序号："))

        if choice == len(all_files) + 1:
            selected_files = all_files
        elif 1 <= choice <= len(all_files):
            selected_files = [all_files[choice - 1]]
        else:
            print("无效的选择。")
            return

        # 逐个处理选中的文件
        for filename in selected_files:
            print(f"开始处理文件: {filename}")
            data = self.read_json_file(os.path.join(folder_path, filename))

            if not data:
                continue

            for index, entry in enumerate(data, start=1):
                print(f"正在处理第 {index} 条数据...")

                analysis_result = self.analyze_text_for_keywords_and_evaluation(
                    json.dumps(entry, ensure_ascii=False), index
                )
                # 修改输出文件名为 filename_output.json
                output_filename = f"{os.path.splitext(filename)[0]}_output.json"
                self.append_output_to_json_file(analysis_result, output_filename)
                print(f"第 {index} 条数据已处理并保存到文件。")
                time.sleep(1)  # 限速，避免触发 API 限制

            print(f"文件 {filename} 处理完成！结果已保存到 output 文件夹。")


# 主程序入口
if __name__ == "__main__":
    TongYi_Info = {
        "api_key": "sk-64b850c731f545569c8bf61e3c416324",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }

    analyzer = TextAnalyzer(
        api_key=TongYi_Info["api_key"],
        base_url=TongYi_Info["base_url"],
    )

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 拼接路径，使用 os.path.join 来避免手动拼接路径字符串
    # data文件夹
    # 日期制定，eg. 20241129
    # 语料库选择，eg. json语料库1
    folder_path = os.path.join(current_dir, "data", "20241202", "json")

    # 打印当前工作目录和目标文件夹路径
    print("当前工作目录:", current_dir)
    print("数据文件夹路径:", folder_path)

    # 处理数据文件夹
    analyzer.process_files(folder_path=folder_path, selected_files=[])
