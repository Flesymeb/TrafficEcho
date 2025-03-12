import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib import rcParams

# 设置 Matplotlib 中文字体
rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体（SimHei）
rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置数据文件夹和输出文件夹
data_dir = "classification_result_score"
output_dir = "score_distribution_plots2"
os.makedirs(output_dir, exist_ok=True)

# 获取所有 JSON 文件
json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

# 分数区间（0-100，每 5 分一个区间）
score_bins = list(range(0, 105, 5))

# 遍历每个 JSON 文件
for json_file in json_files:
    file_path = os.path.join(data_dir, json_file)

    # 读取 JSON 数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取所有 score
    scores = [obj.get("score", 0) for obj in data if isinstance(obj.get("score"), (int, float))]

    if not scores:
        print(f"{json_file} 没有有效的分数数据，跳过绘图。")
        continue

    # 统计分数分布
    score_counts, _ = np.histogram(scores, bins=score_bins)

    # 绘制柱状图
    plt.figure(figsize=(10, 5))
    sns.histplot(scores, bins=score_bins, kde=False, color="blue", edgecolor="black", alpha=0.6, label="分数分布")

    # 画核密度估计（KDE）曲线
    #sns.kdeplot(scores, color="red", linewidth=2, label="核密度估计")

    # 拟合正态分布曲线（可选）
    mu, sigma = norm.fit(scores)  # 计算均值和标准差
    x = np.linspace(0, 100, 100)
    y = norm.pdf(x, mu, sigma) * len(scores) * 5  # 乘以样本数量和 bin 宽度，匹配柱状图尺度
    plt.plot(x, y, color="green", linestyle="--", linewidth=2, label=f"正态分布拟合 ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)")

    # 设置图表信息
    plt.xlabel("分数区间")
    plt.ylabel("微博数量")
    plt.title(f"{json_file.replace('.json', '')} 维度分数分布")
    plt.xticks(score_bins, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()

    # 保存图像
    output_path = os.path.join(output_dir, json_file.replace(".json", ".png"))
    plt.savefig(output_path, dpi=300)
    plt.close()

print(f"所有分数分布图已保存至 {output_dir}/")
