import os
import csv
import json
# 指定顶层目录路径
base_directory = "结果文件"

# 设置目标列名
target_columns = [ "微博正文","ip"]

# 定义 JSON 输出的总目录
json_output_directory = os.path.join(base_directory, "json")
os.makedirs(json_output_directory, exist_ok=True)  # 确保输出目录存在

# 遍历结果文件目录
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith(".csv"):
            csv_file_path = os.path.join(root, file)

            # 生成 JSON 文件路径，直接保存在 /json 下
            json_file_name = file.replace(".csv", ".json")
            json_file_path = os.path.join(json_output_directory, json_file_name)

            # 读取 CSV 并提取指定列内容写入 JSON
            json_data = []  # 用于存储转换后的 JSON 数据
            with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)

                for row in csv_reader:
                    # 提取 IP 和 微博正文
                    ip = row.get("ip", "")  # 如果没有 ip 列或值为空，设置为空字符串
                    content = row.get("微博正文", "").strip()  # 获取微博正文并去掉空格
                    # 无论 ip 是否为空，都保留记录
                    json_data.append({ "微博正文": content,"ip": ip})

            # 将数据写入 JSON 文件
            with open(json_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)  # 保留中文格式

            print(f"已处理: {csv_file_path} -> {json_file_path}")

print("所有 IP 和 微博正文内容已成功提取到 JSON 文件中！")