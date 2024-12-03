import json

# 输入和输出文件名
txt_file = "data/20241202内环堵.txt"
json_file = "data/20241202内环堵.json"


# 读取 TXT 文件
def txt_to_json(txt_file, json_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 创建一个列表来存储每条微博的数据
    result = []

    # 遍历每一行微博内容，创建字典并添加到结果列表
    for line in lines:
        content = line.strip()  # 微博正文
        result.append({"微博正文": content, "ip": ""})  # ip为空

    # 将数据写入 JSON 文件
    with open(json_file, "w", encoding="utf-8") as jsonf:
        json.dump(result, jsonf, ensure_ascii=False, indent=4)


# 示例调用
txt_to_json(txt_file, json_file)
