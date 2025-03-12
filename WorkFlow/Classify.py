import json
import os

# 读取classification.json文件
with open('NEclassified_data_Gmax_Dmax.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 预定义的九个类别
categories = ["安全性", "运行效率", "可靠性", "舒适性", "便捷性", "信息化与智能化", "经济性", "环境友好性","社会可持续性"]

# 创建保存结果的文件夹
output_dir = 'NEclassification_result'
os.makedirs(output_dir, exist_ok=True)

# 初始化字典存储每个类别的帖子
category_data = {category: [] for category in categories}

# 将数据按类别分配
for item in data:
    text = item["微博正文"]
    predicted_labels = item["predicted_label"]

    # 遍历所有标签，添加到对应类别的列表中
    for category in predicted_labels:
        if category in categories:
            category_data[category].append({
                "text": text,
                "predicted_label": [category]
            })

# 将每个类别的结果保存为独立的json文件
for category, items in category_data.items():
    if items:  # 仅在该类别下有数据时保存文件
        file_path = os.path.join(output_dir, f"{category}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

print("数据已成功分割并保存到 'classification_result_0305' 文件夹中。")
