import os
import csv

# 指定顶层目录路径
base_directory = "结果文件"

# 设置目标列名
target_column = "微博正文"

# 定义 txt 输出的总目录
txt_output_directory = os.path.join(base_directory, "txt")
os.makedirs(txt_output_directory, exist_ok=True)  # 确保输出目录存在

# 遍历结果文件目录
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith(".csv"):
            csv_file_path = os.path.join(root, file)

            # 生成 txt 文件路径，直接保存在 /txt 下
            txt_file_name = file.replace(".csv", ".txt")
            txt_file_path = os.path.join(txt_output_directory, txt_file_name)

            # 读取 CSV 并提取指定列内容写入 TXT
            with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)

                with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
                    for index, row in enumerate(csv_reader, start=1):
                        content = row.get(target_column, "").strip()  # 去除空格
                        if content and not (content.startswith("【")):
                            txt_file.write(f"{content}\n")

            print(f"已处理: {csv_file_path} -> {txt_file_path}")

print("所有微博正文内容已成功提取到 txt 文件中！")