import glob
import json
import os
import requests
from tqdm import tqdm  # 进度条库

# 配置通义千问API
API_KEY = "sk-24297ab9c2f9472bb17d17ec9e313021"  # 替换为您的实际API密钥
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 更新为正确的API URL

# 调用大模型 API 提取关键评分依据
def get_key_criteria(category, reason):
    prompt = (
        f"你是一个专业的交通评价分析师，你的任务时阅读以下根据微博用户发布内容进行交通服务特定维度上的评分，并提取只在{category}评价维度上的关键评分依据，输出与{category}相关的词或短语，不要解释，只返回核心关键词。\n评价: {reason}\n关键词:"
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3  # 降低随机性，提高一致性
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()  # 提取 API 响应的关键词
    else:
        return "提取失败"


# 待处理 JSON 文件所在的文件夹路径
data_folder = "reason"  # 请修改为你的 data 文件夹路径
output_folder = "output"  # 处理后文件存放路径
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理 data 文件夹中的所有 JSON 文件
for file_name in os.listdir(data_folder):
    if file_name.endswith(".json"):  # 只处理 JSON 文件
        file_path = os.path.join(data_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 获取评价维度（文件名去掉 .json）
        evaluation_dimension = os.path.splitext(file_name)[0]

        # 读取 JSON 数据
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            # **添加进度条**
            for entry in tqdm(data, desc=f"处理 {evaluation_dimension} 关键词", unit="条"):
                reason = entry["最终理由"]
                entry["关键依据"] = get_key_criteria(evaluation_dimension, reason)  # 提取关键词

        # 保存到新的 JSON 文件
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=4)

        print(f"{file_name} 处理完成，结果已保存到 {output_path}")

print("所有文件处理完毕！")

# 处理后 JSON 文件存放目录
processed_folder = "output"
output_txt_file = "summary.txt"

# 调用 Qwen API 进行整合综述
def summarize_dimension(dimension, key_points):
    prompt = (
        f"你是一个专业的交通评价分析师,下面是关于交通评价维度“{dimension}”的用户反馈关键词：{', '.join(key_points)}\n"
        f"请基于这些关键词，分析并总结：1. 该维度最需要改善的方面;2. 该维度现阶段做得比较好的方面\n"
        f"请按照以下文本格式输出："
        f"维度名称：XXX"
        f"最需要改善的方面: XXX"
        f"现阶段做得最好的方面: XXX"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            print(f"解析失败: {e}")
    else:
        print("API 请求失败，请检查 API Key 是否正确或是否超限！")


# **存储最终的综述 TXT**
with open(output_txt_file, "w", encoding="utf-8") as txt_file:
    json_files = [f for f in os.listdir(processed_folder) if f.endswith(".json")]
    with tqdm(total=len(json_files), desc="生成维度综述", unit="文件") as pbar:
        for file_name in json_files:
            file_path = os.path.join(processed_folder, file_name)
            evaluation_dimension = os.path.splitext(file_name)[0]  # 评价维度

            # 读取 JSON 数据
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # **提取所有 "关键依据" 并显示进度**
            key_points = []
            for entry in tqdm(data, desc=f"提取 {evaluation_dimension} 关键词", leave=False):
                if "关键依据" in entry:
                    key_points.append(entry["关键依据"])

            # 让大模型进行综述
            summary_text = summarize_dimension(evaluation_dimension, key_points)

            # 写入 TXT 文件
            txt_file.write(f"{summary_text}\n\n")
            pbar.update(1)

print(f"维度综述完成，结果已保存到 {output_txt_file}")
