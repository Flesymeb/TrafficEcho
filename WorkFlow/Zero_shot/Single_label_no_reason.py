import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss, jaccard_score
from openai import OpenAI

# 全局变量用于记录总的 token 使用量
total_input_tokens = 0
total_output_tokens = 0


def load_comments(json_file):
    """
    加载评论数据。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        comments = [{"id": i + 1, "comment": item["微博正文"], "true_label": item["true_label"]} for i, item in
                    enumerate(data)]
    return comments


def construct_messages(comment, categories):
    """
    构造消息列表。
    """
    category_str = ", ".join(categories)
    system_message = (
        f"你是一个专业的交通评价分析师，你的任务是对用户提供的微博内容进行主观评价维度上的分类。只返回分类结果，不要解释。评价维度分类指标包括:{category_str}。\n"
        f"以下是对分类指标的一些解释:\n"
        f"经济性（Economic Efficiency）：涉及到花钱、票价、燃油费等关于出行的花费的评价、涉及到运营成本、投资回报率的就是经济性相关；\n"
        f"安全性（Safety）：涉及到交通事故、危险驾驶（超车加塞等）、交通运营安全、设施安全性、乘客安全、运行环境安全、应急响应、事故预防能力（比如查酒驾、交警检查、交通管制等，但只涉及到交警等不算）的，就是安全性相关\n"
        f"但是需要注意的是，如果查酒驾导致了更加拥堵，那就不涉及到安全性，而是运行效率\n"
        f"运行效率（Operational Efficiency）：涉及到道路上的拥堵情况(不是公共交通车厢拥挤情况)、公共交通的运行时间和等待时间的就是运行效率相关；\n"
        f"可靠性（Reliability）：涉及到公共交通准点率、运行时间的稳定性（比如晚点、早点之类的）、服务一致性、线路可靠性、交通事件对运营的影响的、司机拒载等，还有道路的修缮保养情况，提到道路平坦宽敞干净就是可靠性相关，常发性道路拥堵一般不算可靠性；\n"
        f"舒适性（Comfort）：涉及到驾驶/乘坐舒适度、公共交通拥挤、车厢整洁度、设备情况、通风、温度等的或者其他超出交通出行的最基础服务的，就是舒适性相关，也包括类似于堵车对司机或乘客产生的不适感、路况不好带来的不适感；公共交通没有厕所也是舒适性等等；\n"
        f"便捷性（Convenience）：涉及到交通方式的可获得性、线路或路网的覆盖率、换乘便捷性、票务便利性、出行时间成本、无障碍出行等的，是便捷性相关，不要看到“方便”或“便捷”两个字就盲目直接分类到便捷性；\n"
        f"信息化与智能化（Information & Smart Technology）：涉及到交通信息的可获取性、实时信息服务、交通系统的智能化程度、交通预测、智能驾驶、导航系统、交通服务APP方面的，就是信息化与智能化相关；\n"
        f"环境友好性（Environmental Friendliness）：涉及到新能源汽车在环境友好方面的优势、排放污染、绿色出行感受、交通基础设施建设对环境产生影响等的，就是环境友好性相关，注意只提到景色漂亮的内容不算环境友好性；\n"
        f"社会可持续性（Social Sustainability）：涉及到交通公平性、社会福祉、交通对于城市空间的影响、社会服务与责任的、以及对交通政策、交通规划、路口/道路设计、车道划分等提出的意见、要求整改、向企业或者有关部门提出建议的相关评论，就是社会可持续性相关。\n"
        f"请注意：请完整阅读微博正文，将评论内容分类到一个或几个合适的评价维度。但不要过度解读，只判断与交通和出行评价有关的内容，不要受到其余非交通的内容的干扰，以免使一些类别没有必要，比如此次出行的目的（游玩的开心、因为上班的绝望、购物方面等等）、沿途的风景等等。\n"
    )
    user_message = f"评论内容如下：\n{comment}\n"
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def call_api(messages, client):
    """
    调用 DeepSeek API 获取模型生成的响应。
    """
    global total_input_tokens, total_output_tokens
    response = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        stream=False
    )

    # 累加 token 使用量
    usage = response.usage
    total_input_tokens += usage.prompt_tokens
    total_output_tokens += usage.completion_tokens

    return response.choices[0].message.content.strip()


def parse_output(output, categories):
    """
    解析模型生成的输出并转换为分类结果列表。
    """
    if output:
        predicted_categories = [cat.strip() for cat in output.split(",") if cat.strip()]
        return [cat for cat in predicted_categories if cat in categories]
    return []


def process_comments(comments, categories, client):
    """
    处理所有评论并获取分类结果。
    """
    results = []
    for comment_data in comments:
        comment_id = comment_data["id"]
        comment = comment_data["comment"]
        true_label = comment_data["true_label"]

        # 构造消息列表
        messages = construct_messages(comment, categories)

        # 调用 API
        api_output = call_api(messages, client)

        # 解析输出
        predicted_label = parse_output(api_output, categories)

        # 保存结果
        result = {
            "微博正文": comment,
            "true_label": true_label,
            "predicted_label": predicted_label
        }
        results.append(result)
        print(f"Processed Comment ID {comment_id}: True Label={true_label}, Predicted Label={predicted_label}")
    return results


def save_results(results, output_file):
    """
    将分类结果保存到 JSON 文件。
    """
    def custom_dumps(obj, indent):
        if isinstance(obj, dict):
            return "{\n" + ",\n".join(
                f"{indent}{json.dumps(k, ensure_ascii=False)}: {custom_dumps(v, indent + '    ')}" for k, v in
                obj.items()) + "\n" + indent[:-4] + "}"
        elif isinstance(obj, list):
            return "[" + ", ".join(custom_dumps(item, indent) for item in obj) + "]"
        else:
            return json.dumps(obj, ensure_ascii=False)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("[\n")
        f.write(",\n".join(custom_dumps(result, "    ") for result in results))
        f.write("\n]")
    print(f"Results saved to {output_file}")


def run_classification(input_file, output_file, categories):
    """
    执行分类任务。
    """
    client = OpenAI(api_key="sk-e82535fe61064496860ac3fe9c5f6d7d",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 加载评论数据
    comments = load_comments(input_file)

    # 处理评论
    results = process_comments(comments, categories, client)

    # 保存结果
    save_results(results, output_file)
    return results


def load_data(file_path):
    """
    加载分类结果数据。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    true_labels = [item['true_label'] for item in data]
    predicted_labels = [item['predicted_label'] for item in data]
    return true_labels, predicted_labels


def labels_to_binary_matrix(labels_list, all_labels):
    """
    将标签列表转换为二进制矩阵。
    """
    label_to_index = {label: i for i, label in enumerate(all_labels)}
    binary_matrix = np.zeros((len(labels_list), len(all_labels)), dtype=int)
    for i, labels in enumerate(labels_list):
        for label in labels:
            if label in label_to_index:
                binary_matrix[i, label_to_index[label]] = 1
    return binary_matrix


def evaluate_metrics(file_path):
    """
    计算评估指标。
    """
    true_labels, predicted_labels = load_data(file_path)
    all_labels = sorted(set(label for labels in true_labels + predicted_labels for label in labels))
    y_true = labels_to_binary_matrix(true_labels, all_labels)
    y_pred = labels_to_binary_matrix(predicted_labels, all_labels)

    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    jaccard = float(jaccard_score(y_true, y_pred, average='samples'))  # 转换为普通浮点数
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "jaccard_similarity": jaccard,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }


def create_output_directory(base_path, date_str):
    """
    创建输出目录并返回完整路径。
    """
    output_dir = os.path.join(base_path, f"run_result_{date_str}")
    os.makedirs(output_dir, exist_ok=True)  # 如果目录已存在，则不会报错
    return output_dir


if __name__ == "__main__":
    input_file = "shuffled_data.json"
    base_output_path = "classification_evaluation_0304"  # 基础输出路径
    categories = ["安全性", "运行效率", "可靠性", "舒适性", "便捷性", "信息化与智能化", "经济性", "环境友好性", "社会可持续性"]
    num_runs = 1  # 运行次数
    current_date = datetime.now().strftime("%Y%m%d")  # 当前日期

    # 创建动态文件夹
    output_prefix = create_output_directory(base_output_path, current_date)

    all_metrics = []

    # 创建日志文件
    log_file = "classification_evaluation_0304/class_eva.log"
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\nStart Logging at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for run_idx in range(1, num_runs + 1):
        print(f"Running iteration {run_idx}/{num_runs}...")
        output_file = f"{output_prefix}/result_{run_idx}_{current_date}.json"

        # 执行分类任务
        run_classification(input_file, output_file, categories)

        # 计算评估指标
        metrics = evaluate_metrics(output_file)
        all_metrics.append(metrics)

        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"Iteration {run_idx} metrics: {metrics}\n")

        print(f"Iteration {run_idx} metrics: {metrics}")

    # 计算平均值
    avg_metrics = {
        "accuracy": float(np.mean([m["accuracy"] for m in all_metrics])),
        "hamming_loss": float(np.mean([m["hamming_loss"] for m in all_metrics])),
        "jaccard_similarity": float(np.mean([m["jaccard_similarity"] for m in all_metrics])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in all_metrics])),
        "micro_f1": float(np.mean([m["micro_f1"] for m in all_metrics]))
    }

    print("Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    # 写入平均指标到日志文件
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"Average Metrics: {avg_metrics}\n")

    # 保存平均值到文件
    with open("classification_evaluation_0304/average_metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, ensure_ascii=False, indent=4)