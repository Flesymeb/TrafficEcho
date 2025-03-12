import json
import logging
import requests
import os
import glob

# 配置通义千问API
API_KEY = "sk-e82535fe61064496860ac3fe9c5f6d7d"  # 替换为您的实际API密钥
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 更新为正确的API URL

# 定义指标集
categories = [
    "安全性",
    "运行效率",
    "可靠性",
    "舒适性",
    "便捷性",
    "信息化与智能化",
    "经济性",
    "环境友好性",
    "社会可持续性",
]

# 全局变量用于记录总的 token 使用量
total_token = 0


def call_qwen_api(messages, timeout=40, Model="qwen-max"):
    """
    调用通义千问API，返回模型生成的结果以及 token 使用量。
    :param messages: 提供给通义千问的对话历史
    :param timeout: 请求超时时间，默认为40秒
    :param Model: 模型名称，默认为qwen-max
    :return: (模型生成的结果文本, token 使用量)
    """
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": Model,
        "messages": messages,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(
            API_URL, json=payload, headers=headers, timeout=timeout
        )
        if response.status_code == 200:
            response_data = response.json()
            result = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            usage = response_data.get("usage", {}).get("total_tokens", 0)  # 获取 token 使用量
            return result, usage
        else:
            raise Exception(
                f"API请求失败，状态码：{response.status_code}, 响应内容：{response.text}"
            )
    except requests.exceptions.Timeout:
        print("API请求超时，请检查网络连接或增加超时时间。")
        return None, 0
    except Exception as e:
        print(f"API请求异常：{e}")
        return None, 0


def generator_score(weibo_content, category, feedback=""):
    """
    生成器G：为微博内容在指定指标上的表现打分（百分制）。
    :param weibo_content: 微博内容
    :param category: 当前评估的指标
    :param feedback: 判别器D上一轮的反馈
    :return: (分数, 原文信息, 指标, token 使用量)
    """
    prompt = (
        f"请根据以下微博内容在「{category}」这个维度上对城市的交通服务给出一个评分（0-100分），注意需要排除交通无关因素的干扰，并提供简要理由。\n\n"
        f"## 以下是对分类指标的一些解释:\n"
        f"经济性（Economic Efficiency）：涉及到花钱、票价、燃油费等关于出行的花费的评价、涉及到运营成本、投资回报率的就是经济性相关；\n"
        f"安全性（Safety）：涉及到交通事故、危险驾驶（超车加塞等）、交通运营安全、设施安全性、乘客安全、运行环境安全、应急响应、事故预防能力（比如查酒驾、交警检查、交通管制等，但只涉及到交警等不算）的，就是安全性相关\n"
        f"但是需要注意的是，如果查酒驾导致了更加拥堵，那就不涉及到安全性，而是运行效率\n"
        f"运行效率（Operational Efficiency）：涉及到道路上的拥堵情况(不是公共交通拥挤情况)、公共交通的运行时间和等待时间的就是运行效率相关；\n"
        f"可靠性（Reliability）：涉及到公共交通准点率、运行时间的稳定性（比如晚点、早点之类的）、服务一致性、线路可靠性、交通事件对运营的影响的、司机拒载等，还有道路的修缮保养情况，提到道路平坦宽敞干净就是可靠性相关，常发性道路拥堵一般不算可靠性；\n"
        f"舒适性（Comfort）：涉及到驾驶/乘坐舒适度、公共交通拥挤、车厢整洁度、设备情况、通风、温度等的或者其他超出交通出行的最基础服务的，就是舒适性相关，也包括类似于堵车对司机或乘客产生的不适感、路况不好带来的不适感；公共交通没有厕所也是舒适性等等；\n"
        f"便捷性（Convenience）：涉及到交通方式的可获得性、线路或路网的覆盖率、换乘便捷性、票务便利性、出行时间成本、无障碍出行等的，是便捷性相关，不要看到“方便”或“便捷”两个字就盲目直接分类到便捷性；\n"
        f"信息化与智能化（Information & Smart Technology）：涉及到交通信息的可获取性、实时信息服务、交通系统的智能化程度、交通预测、智能驾驶、导航系统、交通服务APP方面的，就是信息化与智能化相关；\n"
        f"环境友好性（Environmental Friendliness）：涉及到新能源汽车在环境友好方面的优势、排放污染、绿色出行感受、交通基础设施建设对环境产生影响等的，就是环境友好性相关，注意只提到景色漂亮的内容不算环境友好性；\n"
        f"社会可持续性（Social Sustainability）：涉及到交通公平性、社会福祉、交通对于城市空间的影响、社会服务与责任的、以及对交通政策、交通规划、路口/道路设计、车道划分等提出的意见、要求整改、向企业或者有关部门提出建议的相关评论，就是社会可持续性相关。\n\n"
        f"评分标准：\n"
        f"- 0 分表示极端负面情感。\n"
        f"- 100 分表示极端正面情感。\n"
        f"- 60 分为合格线，表示评价较为中性或略带不满，虽然有不足但可以接受。\n\n"
        f"微博内容：\n"
        f"{weibo_content}\n\n"
        f"注意，如果有feedback，说明上一轮协商未达到共识，你需要参考以下的feedback重新优化自己的回答：\n"
        f"{feedback}\n\n"
        f"请严格按照以下格式返回结果：\n"
        f"分数: 60（一个数字）；理由: 你的理由"
    )

    messages = [{"role": "system", "content": prompt}]
    result, tokens_used = call_qwen_api(messages, Model="qwen-max")  # 获取 token 使用量

    try:
        score_part, reason_part = result.split("；")
        score = float(score_part.split(":")[-1].strip())
        reason = reason_part.split(":")[-1].strip()
        return score, reason, tokens_used
    except ValueError:
        raise ValueError(f"生成器G返回的格式错误：{result}")


def discriminator_evaluate(
    original_content, generator_score, generator_reason, category, feedback=""
):
    """
    判别器D：评估生成器G的分数合理性。
    :param original_content: 原文信息
    :param generator_score: 生成器G的分数
    :param generator_reason: 生成器G的理由
    :param category: 当前评估的指标
    :param feedback: 判别器D上一轮的反馈
    :return: (同意/不同意, 理由, token 使用量)
    """
    prompt = (
        f"仅依靠微博内容当前语境，请评估生成器G关于本条微博内容所体现的{category}维度的评分是否合理，同意代表你认为对该指标的评估分数认可，可作为最终结果，不同意代表生成器G应重新参考你的建议思考优化自己的回答\n"
        f"## 以下是对分类指标的一些解释:\n"
        f"经济性（Economic Efficiency）：涉及到花钱、票价、燃油费等关于出行的花费的评价、涉及到运营成本、投资回报率的就是经济性相关；\n"
        f"安全性（Safety）：涉及到交通事故、危险驾驶（超车加塞等）、交通运营安全、设施安全性、乘客安全、运行环境安全、应急响应、事故预防能力（比如查酒驾、交警检查、交通管制等，但只涉及到交警等不算）的，就是安全性相关\n"
        f"但是需要注意的是，如果查酒驾导致了更加拥堵，那就不涉及到安全性，而是运行效率\n"
        f"运行效率（Operational Efficiency）：涉及到道路上的拥堵情况(不是公共交通拥挤情况)、公共交通的运行时间和等待时间的就是运行效率相关；\n"
        f"可靠性（Reliability）：涉及到公共交通准点率、运行时间的稳定性（比如晚点、早点之类的）、服务一致性、线路可靠性、交通事件对运营的影响的、司机拒载等，还有道路的修缮保养情况，提到道路平坦宽敞干净就是可靠性相关，常发性道路拥堵一般不算可靠性；\n"
        f"舒适性（Comfort）：涉及到驾驶/乘坐舒适度、公共交通拥挤、车厢整洁度、设备情况、通风、温度等的或者其他超出交通出行的最基础服务的，就是舒适性相关，也包括类似于堵车对司机或乘客产生的不适感、路况不好带来的不适感；公共交通没有厕所也是舒适性等等；\n"
        f"便捷性（Convenience）：涉及到交通方式的可获得性、线路或路网的覆盖率、换乘便捷性、票务便利性、出行时间成本、无障碍出行等的，是便捷性相关，不要看到“方便”或“便捷”两个字就盲目直接分类到便捷性；\n"
        f"信息化与智能化（Information & Smart Technology）：涉及到交通信息的可获取性、实时信息服务、交通系统的智能化程度、交通预测、智能驾驶、导航系统、交通服务APP方面的，就是信息化与智能化相关；\n"
        f"环境友好性（Environmental Friendliness）：涉及到新能源汽车在环境友好方面的优势、排放污染、绿色出行感受、交通基础设施建设对环境产生影响等的，就是环境友好性相关，注意只提到景色漂亮的内容不算环境友好性；\n"
        f"社会可持续性（Social Sustainability）：涉及到交通公平性、社会福祉、交通对于城市空间的影响、社会服务与责任的、以及对交通政策、交通规划、路口/道路设计、车道划分等提出的意见、要求整改、向企业或者有关部门提出建议的相关评论，就是社会可持续性相关。\n\n"
        f"回复“同意”或“不同意”，然后提供理由，对生成器G的评分给出优化建议。\n"
        f"微博内容：{original_content}\n"
        f"生成器G的评分：{generator_score}\n"
        f"生成器G的理由：{generator_reason}\n"
        f"评估指标：{category}\n"
        f"注意，如果有feedback，说明上一轮协商未达到共识，你需要参考以下的feedback重新优化自己的回答：\n"
        f"{feedback}\n\n"
        f"请严格按照以下格式返回结果：同意/不同意；理由: 你的理由"
    )
    messages = [{"role": "system", "content": prompt}]
    result, tokens_used = call_qwen_api(messages, Model="qwen-plus")  # 获取 token 使用量

    try:
        parts = result.split("；")
        if len(parts) >= 2:
            agreement = parts[0].strip()
            reason = "；".join(parts[1:]).split(":")[-1].strip()
        else:
            raise ValueError(f"判别器D返回的格式错误：{result}")

        return agreement, reason, tokens_used
    except ValueError:
        raise ValueError(f"判别器D返回的格式错误：{result}")


def negotiate_score(weibo_content, category, max_rounds=3, tolerance=3):
    global total_token
    negotiation_log = []
    final_round = 0
    feedback = ""

    for round_num in range(1, max_rounds + 1):
        final_round = round_num
        try:
            g_score, g_reason, g_tokens = generator_score(weibo_content, category, feedback)
            negotiation_log.append(f"--------第{round_num}轮:--------\n")
            negotiation_log.append(f"G 评分 {g_score},\n理由: {g_reason}")
            total_token += g_tokens  # 记录 token 使用量
        except Exception as e:
            logging.error(f"生成器 G 出错，跳过该条微博。错误信息：{e}")
            return None  # 直接跳过这条微博

        try:
            d_agreement, d_reason, d_tokens = discriminator_evaluate(weibo_content, g_score, g_reason, category,
                                                                     feedback)
            negotiation_log.append(f"D {d_agreement},\n理由: {d_reason}")
            total_token += d_tokens  # 记录 token 使用量
        except Exception as e:
            logging.error(f"判别器 D 出错，跳过该条微博。错误信息：{e}")
            return None  # 直接跳过这条微博

        if d_agreement == "同意":
            logging.info("\n".join(negotiation_log))
            return g_score, g_reason, final_round

        feedback = f"判别器D认为评分不合理，理由: {d_reason}"

    negotiation_log.append("三轮协商未达成一致，采用 G 最后一次输出")
    logging.warning("\n".join(negotiation_log))

    return g_score, g_reason, final_round


if __name__ == "__main__":
    # 配置日志目录
    log_dir = "logs_NEscore"
    os.makedirs(log_dir, exist_ok=True)

    # 处理所有 JSON 文件
    data_dir = "classification_result"
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for json_file in json_files:
        category_name = os.path.basename(json_file).replace(".json", "")  # 获取类别名（文件名）
        log_file = os.path.join(log_dir, f"negotiation_{category_name}.log")  # 设置独立日志文件

        # 配置日志
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="处理时间 %(asctime)s \n %(message)s",
            encoding="utf-8",
        )

        logging.info(f"======== 开始处理 {category_name}.json ========")

        # 读取 JSON 数据
        with open(json_file, "r", encoding="utf-8") as f:
            weibo_list = json.load(f)

        for idx, weibo_data in enumerate(weibo_list):
            weibo_content = weibo_data.get("text", "")
            predicted_labels = weibo_data.get("predicted_label", [])
            logging.info(f"\n======== 处理第 {idx + 1} 条微博 ========")

            valid_data = True  # 标记当前协商是否有效

            for label in predicted_labels:
                if label in categories:
                    result = negotiate_score(weibo_content, label)
                    if result is None:
                        logging.warning(f"跳过第 {idx + 1} 条微博，因协商过程中出现错误")
                        valid_data = False
                        break  # 直接跳过该条微博

                    final_score, reasoning, iterations = result
                    logging.info(f"最终分数：{final_score}, 协商轮次：{iterations}, 理由：{reasoning}")

            # 如果整条微博数据有效，才更新 JSON
            if valid_data and final_score is not None:
                weibo_data["score"] = final_score
            else:
                continue  # 直接跳过该微博，不写入 JSON

        # 将更新后的 JSON 写回文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(weibo_list, f, ensure_ascii=False, indent=4)

        logging.info(f"======== {category_name}.json 处理完成 ========\n")

    # 输出总的 token 使用量
    print(f"所有微博数据处理完毕，共使用了 {total_token} 个 token。")