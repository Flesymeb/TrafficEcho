import json
import os

cate = ["安全性", "运行效率", "可靠性", "舒适性", "便捷性", "信息化与智能化", "经济性", "环境友好性", "社会可持续性"]
scores = {name: [] for name in cate}

for index_json in os.listdir("NEclassification_result_0307"):
    if index_json.endswith(".json") and index_json.split(".")[0] in cate:
        index = index_json.split(".")[0]
        file_path = os.path.join("NEclassification_result_0307", index_json)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "score" in item:
                scores[index].append(item["score"])

# 计算每个指标的平均分
avg_scores = {}
for indicator, score_list in scores.items():
    if score_list:
        avg_scores[indicator] = round(sum(score_list) / len(score_list), 2)
    else:
        avg_scores[indicator] = 0.00

# 输出各指标平均分
print("各评价指标平均分:")
for indicator, avg_score in avg_scores.items():
    print(f"{indicator}: {avg_score}")

# 权重 from LLMs
cate = ["安全性", "运行效率", "可靠性", "舒适性", "便捷性", "信息化与智能化", "经济性", "环境友好性", "社会可持续性"]
citizen_weights = [0.3145, 0.1229, 0.1183, 0.0551, 0.1055, 0.0505, 0.1725, 0.0314, 0.0293]
planner_weights = [0.3158, 0.1708, 0.0996, 0.0628, 0.1172, 0.0900, 0.0401, 0.0408, 0.0629] # 严格按照categories的顺序来

# 按权重计算总评分
citizen_score = sum(avg_scores[cate[i]] * citizen_weights[i] for i in range(len(cate)))
planner_score = sum(avg_scores[cate[i]] * planner_weights[i] for i in range(len(cate)))

print("\n各主体的评分:")
print(f"市民: {citizen_score:.2f}")
print(f"政府部门: {planner_score:.2f}")